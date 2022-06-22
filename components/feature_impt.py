from __future__ import print_function

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from components import losses, metrics

import operator as op
from functools import reduce

from tqdm import tqdm

import pdb

DEVICE = torch.device('cuda')


def forward(opt, model, tokenizer, objs, qns, q_ori):
    if opt.model_type != 'lxmert':
        words, logits, attended_objs, ans_idxs = model(qns, objs)
        return words, logits, attended_objs, ans_idxs
    else: # for lxmert
        features = objs[:, :, :2048]
        spatials = objs[:, :, 2048:]

        # run lxmert(s)
        inputs = tokenizer(
            q_ori,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        output = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=spatials,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        logits = output['question_answering_score']
        ansidx = torch.argsort(logits, dim=1, descending=True)
        return None, logits, None, ansidx

def apply_mask(opt, inputs, mask):
    assert(inputs.shape[0] == mask.shape[0])
    assert(inputs.shape[1] == mask.shape[1])
    assert(inputs.dim() == mask.dim())
    
    if opt.replace_func == '0s':
        return inputs * mask
    elif opt.replace_func == 'negative_ones':
        # mask inputs
        new_inputs = inputs * mask
        # get replacement
        mask_reverse = (mask==0).float()
        all_neg_ones = -torch.ones(inputs.size(), device=DEVICE)
        replacement = all_neg_ones * mask_reverse
        assert(replacement.shape==new_inputs.shape)
        return new_inputs + replacement
    elif opt.replace_func == 'random_sample':
        reverse_mask = (mask==0.0).float()
        # shuffle batch - dim=0
        random_inputs = inputs.permute((2,0,1)).flatten(start_dim=1) # [2048, Batch*num_objs]
        idx = torch.randperm(random_inputs.shape[1])
        random_inputs = random_inputs[:, idx]
        random_inputs = random_inputs.reshape(inputs.shape[2], inputs.shape[0], inputs.shape[1])
        random_inputs = random_inputs.permute((1,2,0))
        assert(random_inputs.shape==inputs.shape)
        return inputs * mask + random_inputs * reverse_mask
    elif opt.replace_func == 'gaussian-singla':
        ## mask inputs
        impt_input = inputs * mask
        ## get replacement
        mask_reverse = (mask==0).float()
        # get mean/std
        means = torch.zeros(inputs.size(), device=DEVICE)
        std = torch.std(inputs.clone())
        stds = std * torch.ones(inputs.size(), device=DEVICE)
        noise = torch.normal(means, stds) 
        noise = noise * mask_reverse # apply mask
        assert(noise.shape==impt_input.shape)
        return inputs + noise
    elif opt.replace_func == 'gaussian-ours':
        ## mask inputs
        impt_input = inputs * mask
        ## get replacement
        mask_reverse = (mask==0).float()
        # get mean/std
        means = torch.mean(inputs.clone()) * torch.ones(inputs.size(), device=DEVICE)
        std = torch.std(inputs.clone())
        stds = std * torch.ones(inputs.size(), device=DEVICE)
        replacement = torch.normal(means, stds) 
        replacement = replacement * mask_reverse # apply mask
        assert(replacement.shape==impt_input.shape)
        return impt_input + replacement
    elif opt.replace_func == 'shuffle':
        # shuffle feats within bbox, then shuffle bbox, in the same sample
        inputs = inputs.clone()
        for i in range(len(inputs)):
            non_zero_idx = torch.nonzero(mask[i].squeeze().long())
            if len(non_zero_idx) != 0:
                non_zeros = inputs[i][non_zero_idx.squeeze(1)]
                non_zeros = non_zeros[torch.randperm(non_zeros.size(0)),:]
                non_zeros = non_zeros[:,torch.randperm(non_zeros.size(1))]
                inputs[i][non_zero_idx.squeeze(1)] = non_zeros
        return inputs.requires_grad_()
    else: 
        raise ValueError(f"unsupported replace function {opt.replace_func}")
        

def generate_masks(num_samples, input_dim):
    # sample rows of data X. first row is always all ones
    X = np.zeros((num_samples, input_dim)) 
    X[0,:] = np.ones(input_dim, dtype=np.int32)
    already_sampled = set()
    already_sampled.add(str(np.ones(input_dim, dtype=np.int32)))
    already_sampled.add(str(np.zeros(input_dim, dtype=np.int32)))
    for i in range(1, num_samples):
        proposal = np.random.binomial(n=1, p=.5, size=input_dim)
        if str(proposal) not in already_sampled:
            X[i,:] = proposal
            already_sampled.add(str(proposal))
        else: # if already sampled, resample
            counter = 0
            while str(proposal) in already_sampled:
                proposal = np.random.binomial(n=1, p=.5, size=input_dim)
                counter += 1
                if counter > 100:
                    break
            X[i,:] = proposal
            already_sampled.add(str(proposal))
    return torch.from_numpy(X)
        

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom
def SHAP_weighting(opt, x):
    d = opt.num_objects
    k = int(sum(x))
    mult = 20 # scales all weights up so they roughly match scale of constraint_weight
    return mult * (d-1) / (ncr(d, k) * k * (d - k))


def FI_forward(opt, model, tokenizer, objs, qns, answers, hint_scores, hint_flags, q_ori):
    # init
    if opt.random_suff or opt.random_unc or opt.random_inv_FI or opt.random_align:
        hint_scores_random = hint_scores[1]
        hint_scores = hint_scores[0]
    cur_batch_size = objs.size(0)
    # max num of batches
    if 'updn' in opt.model_type:
        MAX_BUDGET = 50
    elif opt.model_type == 'lxmert':
        MAX_BUDGET = 25
    
    ## support data augmentation
    # cases with one additional batch
    if opt.aug_type in ["suff-human", "suff-random", "invariance", "uncertainty-uniform", "saliency-guided"]:
        num_repeat = 2
        # repeat things
        qns_new = qns.repeat(num_repeat, 1)
        objs_new = objs.repeat(num_repeat, 1, 1)
        q_ori_new = q_ori * num_repeat
        # get mask
        if opt.aug_type == "suff-human": # Impt only
            mask = (hint_scores > opt.impt_threshold).float()
        elif opt.aug_type == "invariance": # impt + random nonimpt
            mask_impt = (hint_scores > opt.impt_threshold).float()
            mask_nonimpt = (hint_scores <= opt.impt_threshold).float()
            # randomly select nonimpt objs - uniform over num of selected objs 
            mask_count = mask_nonimpt.sum(dim=(1,2))
            mask_uniform_nonimpt = torch.zeros(hint_scores.size(), device=DEVICE)
            for index, count in enumerate(mask_count):
                prob = torch.randint(int(count)+1, (1,), device=DEVICE) / count
                mask_uniform_nonimpt[index] = (torch.rand((opt.num_objects,1), device=DEVICE) <= prob).float()
            # combine with impt objs
            mask = torch.logical_or(mask_impt, mask_uniform_nonimpt)
            # if all objs are non-impt, ignore
            mask_ignore = (mask_nonimpt.sum(dim=(1, 2)) != mask_nonimpt.size(1)).float()
            mask = mask * mask_ignore.unsqueeze(-1).unsqueeze(-1)
        elif opt.aug_type == "uncertainty-uniform": # NonImpt only
            mask = (hint_scores <= opt.impt_threshold).float()
        elif opt.aug_type == "suff-random": # random
            random_scores = torch.rand(hint_scores.size(), device=DEVICE) # [Batch, num_objs, 1]
            sparsity = random.choices([0.1, 0.25, 0.5], k=cur_batch_size)
            sparsity = torch.tensor(sparsity, device=DEVICE).unsqueeze(-1).unsqueeze(-1) # [Batch, 1, 1]           
            mask = (random_scores > sparsity).float().cuda()
        elif opt.aug_type == "saliency-guided": # mask based on gradcam
            # get vanilla gradient
            vqa_grad = torch.autograd.grad(
                (logits[:cur_batch_size] * (answers > 0).float()).sum(), objs, create_graph=True
            )[0]
            if opt.gradcam_type == "sum":
                vqa_grad = vqa_grad.sum(2)  # [b, num_objs]
            elif opt.gradcam_type == "l1":
                vqa_grad = torch.abs(vqa_grad).sum(2)  # [b, num_objs]
            elif opt.gradcam_type == "l2":
                vqa_grad = (vqa_grad**2).sum(2)  # [b, num_objs]
            vqa_grad_sort, vqa_grad_ind = vqa_grad.sort(1, descending=True)
            # get mask
            # sparsity = random.choices([0.1,0.25,0.5], k=cur_batch_size)
            sparsity = random.choices([0.9,0.75,0.5], k=cur_batch_size)
            sparsity = torch.tensor(sparsity, device=DEVICE).unsqueeze(-1) # [Batch, 1]
            sparsity = (sparsity * opt.num_objects).long()
            vqa_grad_cutoff = vqa_grad_sort.gather(1, sparsity) # [Batch, 1]
            mask = (vqa_grad > vqa_grad_cutoff).detach().unsqueeze(-1).float().cuda() # [Batch, 1, 1]
            
        batch_mask = (mask.sum(dim=(1, 2)) > 0).float()
        # apply mask
        objs_new[cur_batch_size:cur_batch_size*2] = apply_mask(opt, objs, mask)
        # forward
        _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
        logits_ori = logits_new[:cur_batch_size]
        predicted_ans = logits_ori.ge(logits_ori.max(-1, keepdim=True)[0]).detach() # stritly greater
        # concat
        logits = logits_new
        ans_idxs = ans_idxs_new
    
    # two additional batches
    elif opt.aug_type in ["uncertainty-NonImptOnly", "suff-uncertainty", "suff-uncertainty-Chang2021"]:  
        num_repeat = 3
        # repeat things
        qns_new = qns.repeat(num_repeat, 1)
        objs_new = objs.repeat(num_repeat, 1, 1)
        q_ori_new = q_ori * num_repeat
        # get mask
        if opt.aug_type == "uncertainty-NonImptOnly": # NonImpt batch & all zeros batch
            mask_1 = (hint_scores <= opt.impt_threshold ).float()  # NonImpt mask
            mask_2 = torch.zeros(hint_scores.size(), device=DEVICE).float() # all 0s mask
            batch_mask = (mask_1.sum(dim=(1, 2)) > 0).float()
        elif opt.aug_type in ["suff-uncertainty"]: # impt batch & nonimpt batch
            mask_1 = (hint_scores >opt.impt_threshold).float()  # Impt mask
            mask_2 = (hint_scores <=opt.impt_threshold).float()  # NonImpt mask
            batch_mask_1 = (mask_1.sum(dim=(1, 2)) > 0).float()
            batch_mask_2 = (mask_2.sum(dim=(1, 2)) > 0).float()
            batch_mask = torch.cat((batch_mask_1, batch_mask_2))
        # apply mask
        objs_new[cur_batch_size : cur_batch_size*2] = apply_mask(opt, objs, mask_1)
        objs_new[cur_batch_size*2: cur_batch_size * 3] = apply_mask(opt, objs, mask_2)
        # forward
        _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
        logits_ori = logits_new[:cur_batch_size]
        predicted_ans = logits_ori.ge(logits_ori.max(-1, keepdim=True)[0]).detach() # stritly greater
        # concat
        logits = logits_new
        ans_idxs = ans_idxs_new
    
    elif opt.aug_type == 'none':
        batch_mask = None
        ## forward - original
        words, logits, attended_objs, ans_idxs = forward(opt, model, tokenizer, objs, qns, q_ori)
        predicted_ans = logits.ge(logits.max(-1, keepdim=True)[0]).detach() # stritly greater
    else:
        raise ValueError("unsupported augmentation method")
    
    # flag to use eval mode to calc FI
    if opt.eval_FI == True:
        model.eval()
    
    ## calc FI methods
    # gradcam FI
    if opt.model_importance == "gradcam":
        # get vanilla gradient
        if opt.FI_predicted_class:
            vqa_grad = torch.autograd.grad(
                (logits[:cur_batch_size] * (predicted_ans > 0).float()).sum(), objs, create_graph=True
            )[0]
        else:
            vqa_grad = torch.autograd.grad(
                (logits[:cur_batch_size] * (answers > 0).float()).sum(), objs, create_graph=True
            )[0]
        # ways to combine feats of an obj
        if opt.gradcam_type == "sum":
            model_impt = vqa_grad.sum(2)  # [b, num_objs]
        elif opt.gradcam_type == "l1":
            model_impt = torch.abs(vqa_grad).sum(2)  # [b, num_objs]
        elif opt.gradcam_type == "l2":
            model_impt = (vqa_grad**2).sum(2)  # [b, num_objs]

    # expected gradient FI
    elif opt.model_importance == "expected_gradient":
        if opt.num_sample_eg <= MAX_BUDGET:
            # get new input
            qns_new = qns.repeat(opt.num_sample_eg, 1)
            objs_new = objs.repeat(opt.num_sample_eg, 1, 1).requires_grad_()
            q_ori_new = q_ori * opt.num_sample_eg
            # get baseline
            mask = torch.zeros(objs_new.size(), device=DEVICE)
            baseline = apply_mask(opt, objs_new, mask)
            # interpolate
            for j in range(opt.num_sample_eg):
                alpha = torch.rand((cur_batch_size, 1, 1), device=DEVICE)
                alpha = alpha.repeat(1, objs.size(1), objs.size(2))
                new_input = objs + alpha * (
                    baseline[cur_batch_size * j : cur_batch_size * (j + 1)] - objs
                )
                new_input = new_input.requires_grad_()
                objs_new[cur_batch_size * j : cur_batch_size * (j + 1)] = new_input
            # forward
            _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
            # backward
            if opt.FI_predicted_class:
                vqa_grad = torch.autograd.grad((logits_new * (predicted_ans.repeat((opt.num_sample_eg, 1)) > 0).float()).sum(), objs_new, create_graph=True)[0]
            else:
                vqa_grad = torch.autograd.grad((logits_new * (answers.repeat((opt.num_sample_eg, 1)) > 0).float()).sum(), objs_new, create_graph=True)[0]
            vqa_grad = vqa_grad.split(cur_batch_size)
            # get model impt
            model_impt = torch.zeros((objs.size(0), opt.num_objects), device=DEVICE)
            for j in range(opt.num_sample_eg):
                vqa_grad_j = vqa_grad[j]
                if opt.gradcam_type == "sum":
                    model_impt = model_impt + vqa_grad_j.sum(2)  # [b, num_objs]
                elif opt.gradcam_type == "l1":
                    model_impt = model_impt + torch.abs(vqa_grad_j).sum(2)  # [b, num_objs]
                elif opt.gradcam_type == "l2":
                    model_impt = model_impt + (vqa_grad_j**2).sum(2)  # [b, num_objs]
            model_impt = model_impt / opt.num_sample_eg 
        else: # over budget limit -> NOTE: only for calc metrics; not for training
            assert(opt.calc_dp_level_metrics)
            # forward/backward both count as 1 budget
            # 1 sample for Expected Gradient uses 2 budget
            budget_eg = MAX_BUDGET // 2 
            num_batch = opt.num_sample_eg // budget_eg
            assert num_batch*budget_eg == opt.num_sample_eg, "only support exact division"
            
            model_impt = torch.zeros((objs.size(0), opt.num_objects), device=DEVICE)
            for batch_index in range(num_batch):
                # get new input
                qns_new = qns.repeat(budget_eg, 1)
                objs_new = objs.repeat(budget_eg, 1, 1).requires_grad_()
                q_ori_new = q_ori * budget_eg
                # get baseline
                mask_zeros = torch.zeros(objs_new.size(), device=DEVICE)
                baseline = apply_mask(opt, objs_new, mask_zeros)
                # interpolate
                for j in range(budget_eg):
                    alpha = torch.rand((cur_batch_size, 1, 1), device=DEVICE)
                    alpha = alpha.repeat(1, objs.size(1), objs.size(2))
                    new_input = objs + alpha * (
                        baseline[cur_batch_size * j : cur_batch_size * (j + 1)] - objs
                    )
                    new_input = new_input.requires_grad_()
                    objs_new[cur_batch_size * j : cur_batch_size * (j + 1)] = new_input
                # forward
                _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
                # backward
                if opt.FI_predicted_class:
                    vqa_grad = torch.autograd.grad((logits_new * (predicted_ans.repeat((budget_eg, 1)) > 0).float()).sum(), objs_new, create_graph=True)[0]
                else:
                    vqa_grad = torch.autograd.grad((logits_new * (answers.repeat((budget_eg, 1)) > 0).float()).sum(), objs_new, create_graph=True)[0]
                # get FI 
                for j in range(budget_eg):
                    vqa_grad_j = vqa_grad.split(cur_batch_size)[j].detach() # detach to save memory
                    if opt.gradcam_type == "sum":
                        model_impt = model_impt + vqa_grad_j.sum(2)  # [b, num_objs]
                    elif opt.gradcam_type == "l1":
                        model_impt = model_impt + torch.abs(vqa_grad_j).sum(2)  # [b, num_objs]
                    elif opt.gradcam_type == "l2":
                        model_impt = model_impt + (vqa_grad_j**2).sum(2)  # [b, num_objs]
                        
            model_impt /= opt.num_sample_eg

    # leave-one-out FI
    elif opt.model_importance == "LOO":
        model_impt = torch.zeros((cur_batch_size, opt.num_objects), device=DEVICE)# size = [batch_size, 36]

        if opt.num_sample_omission>=opt.num_objects: # full sample
            # get new input
            qns_new = qns.repeat(opt.num_objects, 1)
            objs_new = objs.repeat(opt.num_objects, 1, 1)
            q_ori_new = q_ori * opt.num_objects
            for j in range(opt.num_objects):  # for each obj
                # get input for masking bbox #j
                mask = torch.ones((objs.size(0), objs.size(1), 1), device=DEVICE)
                mask[:, j : j + 1, :] = torch.zeros((objs.size(0), 1, 1), device=DEVICE)
                objs_new[cur_batch_size * j : cur_batch_size * (j + 1)] = apply_mask(opt, objs, mask)
        elif opt.num_sample_omission<opt.num_objects: # less than d samples
            # get new input
            qns_new = qns.repeat(opt.num_sample_omission, 1)
            objs_new = objs.repeat(opt.num_sample_omission, 1, 1)
            q_ori_new = q_ori * opt.num_sample_omission
            
            # get & apply mask
            mask_sum = torch.zeros((cur_batch_size, opt.num_objects), device=DEVICE) # count each position
            mask_full = [] # record all masks
            for j in range(opt.num_sample_omission):
                # get mask
                mask = torch.rand((cur_batch_size, opt.num_objects), device=DEVICE)
                mask = mask.lt(mask.max(-1, keepdim=True)[0]).float()
                while mask.sum() != cur_batch_size*(opt.num_objects-1):
                    mask = torch.rand((cur_batch_size, opt.num_objects), device=DEVICE)
                    mask = mask.lt(mask.max(-1, keepdim=True)[0]).float()
                mask_sum += (mask==0).float() # count how many times each feature is used
                mask_full.append(mask)
                # apply mask
                mask = mask.unsqueeze(-1) # reshape
                objs_new[cur_batch_size*j:cur_batch_size*(j+1)] = apply_mask(opt, objs, mask)

        # forward
        if opt.calc_dp_level_metrics:
            with torch.no_grad():
                _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
        else:
            _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
        

        # get model importance
        if opt.num_sample_omission>=opt.num_objects: # full sample
            for j in range(opt.num_objects):  # for each obj
                logits_batch = logits_new.split(cur_batch_size)[j] # [Batch, num_answers]
                if opt.FI_predicted_class:
                    model_impt[:, j] = ((logits[:cur_batch_size].detach() - logits_batch) * (predicted_ans > 0).float()).sum(1)
                else:
                    model_impt[:, j] = ((logits[:cur_batch_size].detach() - logits_batch) * (answers > 0).float()).sum(1)
        else: # less than d samples
            for j in range(opt.num_sample_omission): # for j-th sample
                # calc diff from baseline
                logits_batch = logits_new.split(cur_batch_size)[j] # [Batch, num_answers]
                if opt.FI_predicted_class:
                    diff = ((logits[:cur_batch_size].detach() - logits_batch) * (predicted_ans > 0).float()).sum(1)
                else:
                    diff = ((logits[:cur_batch_size].detach() - logits_batch) * (answers > 0).float()).sum(1)
                # add to model_impt
                mask = mask_full[j]
                assert(mask.dim() == diff.unsqueeze(-1).dim())
                assert(mask.shape == model_impt.shape)
                model_impt += (mask==0).float() * diff.unsqueeze(-1)
            
            # divide by num of samples
            mask_wherezeros = (mask_sum == 0).float()
            assert((mask_wherezeros * model_impt).sum()<1e-5)
            mask_sum = mask_sum + mask_wherezeros # change 0s to 1s
            assert(mask_sum.shape == model_impt.shape)
            model_impt = model_impt / mask_sum            
                    
    # keep-one-in FI
    elif opt.model_importance == "KOI":
        model_impt = torch.zeros((cur_batch_size, opt.num_objects), device=DEVICE) # size = [batch_size, 36]

        if opt.num_sample_omission>=opt.num_objects: # full sample
            # get new input
            num_repeat = opt.num_objects + 1
            qns_new = qns.repeat(num_repeat, 1)
            objs_new = objs.repeat(num_repeat, 1, 1)
            q_ori_new = q_ori * num_repeat
            # get each obj
            for j in range(opt.num_objects):  # for each obj
                # get input for masking bbox #j
                mask = torch.zeros(objs.size(), device=DEVICE)
                mask[:, j : j + 1, :] = torch.ones((objs.size(0), 1, objs.size(2)), device=DEVICE)
                objs_new[cur_batch_size*(j+1):cur_batch_size*(j+2)] = apply_mask(opt, objs, mask)
        else: # less samples
            # get new input
            num_repeat = opt.num_sample_omission + 1
            qns_new = qns.repeat(num_repeat, 1)
            objs_new = objs.repeat(num_repeat, 1, 1)
            q_ori_new = q_ori * num_repeat
            
            # get & apply mask
            mask_sum = torch.zeros((cur_batch_size, opt.num_objects), device=DEVICE) # count each position
            mask_full = [] # record all masks
            for j in range(opt.num_sample_omission):
                # get mask
                mask = torch.rand((cur_batch_size, opt.num_objects), device=DEVICE)
                mask = mask.ge(mask.max(-1, keepdim=True)[0]).float()
                while mask.sum() != cur_batch_size:
                    mask = torch.rand((cur_batch_size, opt.num_objects), device=DEVICE)
                    mask = mask.ge(mask.max(-1, keepdim=True)[0]).float()
                mask_sum += mask
                mask_full.append(mask)
                # apply mask
                mask = mask.unsqueeze(-1) # reshape
                objs_new[cur_batch_size*(j+1):cur_batch_size*(j+2)] = apply_mask(opt, objs, mask)

        # get baseline
        mask_zeros = torch.zeros(objs.size(), device=DEVICE)
        objs_new[:cur_batch_size] = apply_mask(opt, objs, mask_zeros)
        
        # forward
        if opt.calc_dp_level_metrics:
            with torch.no_grad():
                _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
        else:
            _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
        logits_ref = logits_new[:cur_batch_size].detach()
        
        if opt.num_sample_omission>=opt.num_objects: # full sample
            # get model importance
            for j in range(opt.num_objects):  # for each obj
                # get model importance
                logits_batch = logits_new.split(cur_batch_size)[j+1] # [Batch, num_answers]
                if opt.FI_predicted_class:
                    model_impt[:, j] = ((logits_batch - logits_ref.detach()) * (predicted_ans > 0).float()).sum(1)
                else:
                    model_impt[:, j] = ((logits_batch - logits_ref.detach()) * (answers > 0).float()).sum(1)
        else: # less samples
            # get model importance
            for j in range(opt.num_sample_omission): # for j-th sample
                # calc diff from baseline
                logits_batch = logits_new.split(cur_batch_size)[j+1] # [Batch, num_answers]
                if opt.FI_predicted_class:
                    diff = ((logits_batch - logits_ref.detach()) * (predicted_ans > 0).float()).sum(1)
                else:
                    diff = ((logits_batch - logits_ref.detach()) * (answers > 0).float()).sum(1)
                # add to model_impt
                mask = mask_full[j]
                assert(mask.shape == model_impt.shape)
                model_impt += mask * diff.unsqueeze(-1)
            # divide by num of samples
            mask_wherezeros = (mask_sum == 0).float()
            assert((mask_wherezeros * model_impt).sum()<1e-5)
            mask_sum = mask_sum + mask_wherezeros # change 0s to 1s
            assert(mask_sum.shape == model_impt.shape)
            model_impt = model_impt / mask_sum

    # SHAP FI
    elif opt.model_importance == "SHAP":
        model_impt = torch.zeros((cur_batch_size, opt.num_objects), device=DEVICE) # size = [batch_size, 36]
        
        if opt.num_sample_omission <= MAX_BUDGET:
            # get new input
            num_repeat = opt.num_sample_omission+1
            qns_new = qns.repeat(num_repeat, 1)
            objs_new = objs.repeat(num_repeat, 1, 1)
            q_ori_new = q_ori * num_repeat

            # batch 0 -> baseline
            mask_zeros = torch.zeros(objs.size(), device=DEVICE)
            objs_new[:cur_batch_size] = apply_mask(opt, objs, mask_zeros)

            # generate masks
            X = generate_masks(opt.num_sample_omission, opt.num_objects) # [num_sample, num_objs/num_features]
            # apply mask
            for j in range(opt.num_sample_omission):
                mask = X[j].unsqueeze(0).unsqueeze(-1).repeat(cur_batch_size,1,1).float().cuda()
                objs_new[cur_batch_size*(j+1):cur_batch_size*(j+2)] = apply_mask(opt, objs, mask)

            # forward
            _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
            logits_samples = logits_new[cur_batch_size:cur_batch_size*num_repeat] # [Batch * num_sample, num_ans]
            logits_zero = logits_new[:cur_batch_size]

            ## get y
            # subtract off model_at_null_input from y
            logits_samples = logits_samples - logits_zero.repeat(opt.num_sample_omission, 1)
            # get gt answer class, [batch * num_samples, ]
            if opt.FI_predicted_class:
                logits_samples = (logits_samples * (predicted_ans.repeat(opt.num_sample_omission,1) >0).float()).sum(1)
            else:
                logits_samples = (logits_samples * (answers.repeat(opt.num_sample_omission,1)>0).float()).sum(1)

            # get y, [batch, num_samples]
            y = logits_samples.reshape((cur_batch_size, opt.num_sample_omission))
        else: # over budget limit -> NOTE: default no gradient! not for training!
            assert(opt.calc_dp_level_metrics)
            num_batch = opt.num_sample_omission // MAX_BUDGET
            assert num_batch*MAX_BUDGET == opt.num_sample_omission, "only support exact division"
            
            X_full = generate_masks(opt.num_sample_omission, opt.num_objects) # [num_sample, num_objs/num_features]
            y_full = []
            
            # batch 0 -> baseline
            mask_zeros = torch.zeros(objs.size(), device=DEVICE)
            objs_new = apply_mask(opt, objs, mask_zeros)
            _, logits_zero, _, _ = forward(opt, model, tokenizer, objs_new, qns, q_ori)
            
            for batch_index in range(num_batch):
                # get new input
                num_repeat = MAX_BUDGET
                qns_new = qns.repeat(num_repeat, 1)
                objs_new = objs.repeat(num_repeat, 1, 1)
                q_ori_new = q_ori * num_repeat


                # get masks
                X = X_full.split(MAX_BUDGET)[batch_index] # [MAX_BUDGET, num_objs/num_features]
                # apply mask
                for j in range(MAX_BUDGET):
                    mask = X[j].unsqueeze(0).unsqueeze(-1).repeat(cur_batch_size,1,1).float().cuda()
                    objs_new[cur_batch_size*j:cur_batch_size*(j+1)] = apply_mask(opt, objs, mask)

                # forward
                with torch.no_grad():
                    _, logits_samples, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)

                ## get y
                # subtract off model_at_null_input from y
                logits_samples = logits_samples - logits_zero.repeat(MAX_BUDGET, 1)
                # get logit at answer class, [batch * num_samples, ]
                if opt.FI_predicted_class:
                    logits_samples = (logits_samples * (predicted_ans.repeat(MAX_BUDGET,1) >0).float()).sum(1)
                else:
                    logits_samples = (logits_samples * (answers.repeat(MAX_BUDGET,1)>0).float()).sum(1)

                # get y, [batch, num_samples]
                y = logits_samples.reshape((cur_batch_size, MAX_BUDGET))
                
                # record
                y_full.append(y.detach())
                
            X = X_full
            y = torch.cat(y_full, dim=1)
        
        # do a linear regression with weights on the data points
        # this is what an unweighted regression looks like
        #   cov = torch.mm(X.T, X)
        #   inv_cov = torch.inverse(cov)
        #   XY = torch.mm(X.T, y)
        #   beta = torch.mm(inv_cov, XY)
        # but we use data point weights W to enforce SHAP additivity constraint and kernel weighting

        W = torch.eye(opt.num_sample_omission).cuda()
        if opt.num_sample_omission < 10:
            constraint_weight = 5
        else:
            constraint_weight = min(max(int(.1 * opt.num_sample_omission), 10), 10) # =10 for 15 samples
        W[0,0] = constraint_weight
        
        ## SHAP_weighting
        for i in range(1, opt.num_sample_omission):
            W[i,i] = SHAP_weighting(opt, X[i])

        # if at least as many samples as input dim, should be able to do the regression
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if opt.num_sample_omission >= opt.num_objects:                
            WX = torch.mm(W, X)
            Wcov = torch.mm(X.T, WX)
            try: # fix for occasional numerical problem
                inv_Wcov = torch.inverse(Wcov + .001 * torch.eye(X.size(1), device=DEVICE))
            except: # fix for occasional numerical problem
                inv_Wcov = torch.inverse(Wcov + .01 * torch.eye(X.size(1), device=DEVICE))
            # XWY = torch.mm(WX.T, y)
            XWY = torch.mm(WX.T, y.T)
            beta = torch.mm(inv_Wcov, XWY)
            model_impt = beta.T # [batch_size, num_objs]
        else:
            assert(False)

    # average effect FI
    elif opt.model_importance == "avg_effect":
        model_impt = torch.zeros((cur_batch_size, opt.num_objects), device=DEVICE).cuda()  # size = [batch_size, 36]
        
        if opt.num_sample_omission <= MAX_BUDGET:
            # get new input
            num_repeat = opt.num_sample_omission
            qns_new = qns.repeat(num_repeat, 1)
            objs_new = objs.repeat(num_repeat, 1, 1)
            q_ori_new = q_ori * num_repeat

            # get masks
            if opt.num_sample_omission > 2:# [num_sample, num_objs/num_features]
                X = generate_masks(opt.num_sample_omission, opt.num_objects) 
            elif opt.num_sample_omission == 2:
                mask = np.random.binomial(n=1, p=.5, size=opt.num_objects).reshape(1,-1)
                random_idx = np.random.randint(opt.num_objects)
                cf_mask = mask.copy()
                if mask[0,random_idx]:
                    cf_mask[0,random_idx] = 0
                else:
                    cf_mask[0,random_idx] = 1
                X = np.concatenate((mask, cf_mask), axis=0)
                X = torch.from_numpy(X)
            else:
                raise ValueError("unsupported budget for AveEffect")

            # apply mask
            for j in range(opt.num_sample_omission):
                mask = X[j].unsqueeze(0).unsqueeze(-1).repeat(cur_batch_size,1,1).float().cuda() # [Batch, num_objs, 2048]
                objs_new[cur_batch_size*j:cur_batch_size*(j+1)] = apply_mask(opt, objs, mask)

            # forward
            _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
            ## get y
            # get gt answer class, [batch * num_samples, ]
            if opt.FI_predicted_class:
                logits_new = (logits_new * (predicted_ans.repeat(opt.num_sample_omission,1)>0).float()).sum(1)
            else:
                logits_new = (logits_new * (answers.repeat(opt.num_sample_omission,1)>0).float()).sum(1)
            # get y, [batch, num_samples]
            y = logits_new.reshape((cur_batch_size, opt.num_sample_omission))
        
        else: # over budget limit -> NOTE: default no gradient! not for training!
            assert(opt.calc_dp_level_metrics)
            num_batch = opt.num_sample_omission // MAX_BUDGET
            assert num_batch*MAX_BUDGET == opt.num_sample_omission, "only support exact division"
            
            X_full = generate_masks(opt.num_sample_omission, opt.num_objects) # [num_sample, num_objs/num_features]
            y_full = []
            
            for batch_index in range(num_batch):
                # get new input
                num_repeat = MAX_BUDGET
                qns_new = qns.repeat(num_repeat, 1)
                objs_new = objs.repeat(num_repeat, 1, 1)
                q_ori_new = q_ori * num_repeat
                
                X = X_full.split(MAX_BUDGET)[batch_index]
                
                # apply mask
                for j in range(MAX_BUDGET):
                    mask = X[j].unsqueeze(0).unsqueeze(-1).repeat(cur_batch_size,1,1).float().cuda() # [Batch, num_objs, 2048]
                    objs_new[cur_batch_size*j:cur_batch_size*(j+1)] = apply_mask(opt, objs, mask)

                # forward
                with torch.no_grad():
                    _, logits_new, _, ans_idxs_new = forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
                ## get y
                # get gt answer class, [batch * num_samples, ]
                if opt.FI_predicted_class:
                    logits_new = (logits_new * (predicted_ans.repeat(MAX_BUDGET,1)>0).float()).sum(1)
                else:
                    logits_new = (logits_new * (answers.repeat(MAX_BUDGET,1)>0).float()).sum(1)
                # get y, [batch, num_samples]
                y = logits_new.reshape((cur_batch_size, MAX_BUDGET))
                y_full.append(y.detach())
                
            X = X_full
            y = torch.cat(y_full, dim=1)

        # for each feature, compute avg difference
        for d in range(opt.num_objects):
            where_one = np.argwhere(X[:,d])
            where_zero = np.setdiff1d(np.arange(opt.num_sample_omission), where_one)
            where_one = where_one.squeeze(0).cuda()
            where_zero = torch.tensor(where_zero, device=DEVICE)
            if len(where_one)==0 or len(where_zero)==0:
                avg_diff = torch.zeros((cur_batch_size,), device=DEVICE)
            else:
                avg_diff = y[:, where_one].mean(1) - y[:, where_zero].mean(1)
            model_impt[:, d] = avg_diff
    
    ## updn att layer FI
    elif opt.model_importance == "updn_att":  # directly regulate updn att
        model_impt = attended_objs.squeeze()
    ## lxmert att layer FI
    elif opt.model_importance == "lxmert_att":  # directly regulate updn att
        return None    
    elif opt.model_importance == 'none':
        model_impt = None
    else:
        raise ValueError("unsupported FI method")
    
    if opt.eval_FI == True:
        model.train()
        
    return logits, ans_idxs, model_impt, batch_mask