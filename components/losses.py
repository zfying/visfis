import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

def instance_bce_with_logits(opt, logits, labels):
    assert logits.dim() == 2
    
    if opt.vqa_loss_type == 'sigmoid':
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss *= labels.size(1)
    elif opt.vqa_loss_type == 'softmax':
        answers_class = labels.argmax(1)
        loss = F.cross_entropy(logits, answers_class)

    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def get_prediction_scores(logits, labels):
    logits = torch.max(logits, 1)[1]
    return labels.gather(1, logits.unsqueeze(1)).data


def compute_score_with_k_logits(logits, labels, k=5):
    logits = torch.sort(logits, 1)[1].data  # argmax
    scores = torch.zeros((labels.size(0), k))

    for i in range(k):
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits[:, -i - 1].view(-1, 1), 1)
        scores[:, i] = (one_hots * labels).squeeze().sum(1)
    scores = scores.max(1)[0]
    return scores


def compute_scr_loss(
    opt, objs, answers, ans_idxs, logits, hint_flags, hint_scores, ans_cossim
):
    """Self-Critical Loss copied from https://github.com/jialinwu17/self_critical_vqa"""
    eps = 0.0000001
    if opt.vqa_loss_type == 'softmax':
        to_prob_func = nn.Softmax(dim=1)
    else:
        to_prob_func = nn.Sigmoid()

    bucket = opt.bucket
    aidx = answers.argmax(1).detach().cpu().numpy().reshape((-1))

    vqa_grad = torch.autograd.grad(
        (logits * (answers > 0).float()).sum(), objs, create_graph=True
    )[0]  # B x num_objs x 2048
    vqa_grad_cam = vqa_grad.sum(2)

    ### hint_loss
    loss_hint = torch.zeros(
        (vqa_grad_cam.size(0), opt.num_sub, opt.num_objects)
    ).cuda()  # B x 5 x num_obj

    hint_scores = hint_scores.squeeze()  # B x num_objs
    hint_sort, hint_ind = hint_scores.sort(1, descending=True)

    thresh = hint_sort[:, opt.num_sub : opt.num_sub + 1] - 0.00001
    thresh += (thresh < 0.2).float() * 0.1
    hint_scores = (hint_scores > thresh).float()

    for j in range(opt.num_sub):
        for k in range(opt.num_objects):
            if j == k:
                continue
            hint1 = hint_scores.gather(
                1, hint_ind[:, j : j + 1]
            ).squeeze()  # j-th hint score
            hint2 = hint_scores.gather(1, hint_ind[:, k : k + 1]).squeeze()

            vqa1 = vqa_grad_cam.gather(1, hint_ind[:, j : j + 1]).squeeze()  # j-th grad
            vqa2 = vqa_grad_cam.gather(1, hint_ind[:, k : k + 1]).squeeze()

            if j < k:
                mask = ((hint1 - hint2) * (vqa1 - vqa2 - 0.0001) < 0).float()
                loss_hint[:, j, k] = torch.abs(vqa1 - vqa2 - 0.0001) * mask
            else:
                mask = ((hint2 - hint1) * (vqa2 - vqa1 - 0.0001) < 0).float()
                loss_hint[:, j, k] = torch.abs(vqa2 - vqa1 - 0.0001) * mask

    hint_flag1 = (
        hint_flags.unsqueeze(1)
        .unsqueeze(2)
        .repeat(1, loss_hint.shape[1], loss_hint.shape[2])
        .detach_()
        .cuda()
        .float()
    )
    loss_hint *= opt.scr_hint_loss_weight
    loss_hint *= hint_flag1
    loss_hint = loss_hint.sum(2)  # b num_sub
    loss_hint += (
        (loss_hint.sum(1).unsqueeze(1) > eps).float() * (loss_hint < eps).float()
    ) * 10000

    loss_hint, loss_hint_ind = loss_hint.min(1)  # loss_hint_ind b
    loss_hint_mask = (loss_hint > eps).float()
    loss_hint = (loss_hint * loss_hint_mask).sum() / (loss_hint_mask.sum() + eps)
    gt_logits = logits.gather(1, answers.argmax(1).view((-1, 1)))
    prob = to_prob_func(gt_logits).view(-1)
    ### end of hint_loss

    ### scr loss
    loss_compare = torch.zeros((logits.size(0), bucket)).cuda()
    loss_reg = torch.zeros((logits.size(0), bucket)).cuda()
    comp_mask = torch.zeros((logits.size(0), bucket)).cuda()
    for j in range(bucket):
        logits_pred = logits.gather(1, ans_idxs[:, j : j + 1])
        prob_pred = to_prob_func(logits_pred).squeeze()

        vqa_grad_pred = torch.autograd.grad(
            logits.gather(1, ans_idxs[:, j : j + 1]).sum(), objs, create_graph=True
        )[0]
        vqa_grad_pred_cam = vqa_grad_pred.sum(2)  # b * num_objs
        gradcam_diff = vqa_grad_pred_cam - vqa_grad_cam

        pred_aidx = ans_idxs[:, j].detach().cpu().numpy().reshape((-1))
        if opt.apply_answer_weight:
            ans_diff = (
                torch.from_numpy(1 - ans_cossim[aidx, pred_aidx].reshape((-1)))
                .cuda()
                .float()
            )
        prob_diff = prob_pred - prob
        prob_diff_relu = prob_diff * (prob_diff > 0).float()

        if opt.apply_answer_weight:
            loss_comp1 = (
                prob_diff_relu.unsqueeze(1)
                * gradcam_diff
                * ans_diff.unsqueeze(1)
                * hint_scores
            )
        else:
            loss_comp1 = prob_diff_relu.unsqueeze(1) * gradcam_diff * hint_scores
        loss_comp1 = loss_comp1.gather(1, loss_hint_ind.view(-1, 1)).squeeze()  # sum(1)
        loss_comp1 *= opt.scr_compare_loss_weight
        loss_compare[:, j] = loss_comp1
        comp_mask[:, j] = (prob_diff > 0).float().squeeze()
        
        if opt.reg_loss_weight!=0:
            if opt.apply_answer_weight:
                loss_reg[:, j] = (
                    torch.abs(vqa_grad_pred_cam * ans_diff.unsqueeze(1) * (1 - hint_scores))
                ).sum(1)
            else:
                loss_reg[:, j] = (torch.abs(vqa_grad_pred_cam * (1 - hint_scores))).sum(1)
    
    hint_flag2 = (
        hint_flags.unsqueeze(1).repeat(1, loss_reg.shape[1]).detach_().cuda().float()
    )
    loss_compare *= hint_flag2
    loss_compare = (loss_compare * comp_mask).sum() / (comp_mask.sum() + 0.0001)
    
    if opt.reg_loss_weight!=0:
        loss_reg *= hint_flag2
        loss_reg = loss_reg.mean() * opt.reg_loss_weight
    
    return loss_hint, loss_compare, loss_reg


def compute_hint_loss(
    opt, objs, gt_answers, logits, gt_hint_scores, hint_flags, model_impt
):
    """
    Implementation for the HINT paper (Selvaraju, Ramprasaath R., et al.)
    model_impt: B x num_objs
    """

    # Subtract hint of every object from other objects
    gt_hint_scores, gt_hintscore_ixs = torch.sort(gt_hint_scores, 1, descending=True)
    gt_hint_scores = gt_hint_scores.squeeze()
    gt_hint_score_diff = gt_hint_scores.unsqueeze(2) - gt_hint_scores.unsqueeze(1)

    # Sort the predicted hint scores in the same order as GT hint scores
    model_impt_sorted_as_gt = model_impt.gather(1, gt_hintscore_ixs.squeeze())
    model_impt_sorted_as_gt_diff = model_impt_sorted_as_gt.unsqueeze(
        2
    ) - model_impt_sorted_as_gt.unsqueeze(1)

    # Mask off the hint differences that are negative in GT, as we don't need to consider them for the loss
    # This should basically produce an upper triangular matrix
    gt_mask = torch.where(
        gt_hint_score_diff < 0,
        torch.zeros_like(gt_hint_score_diff),
        torch.ones_like(gt_hint_score_diff),
    )
    model_impt_sorted_as_gt_diff = model_impt_sorted_as_gt_diff * gt_mask

    # Mask off prediction hint differences which have negative signs
    # i.e., only keep the object pairs which do not match the order defined by GT
    pred_mask = torch.where(
        model_impt_sorted_as_gt_diff < 0,
        -1 * torch.ones_like(model_impt_sorted_as_gt_diff),
        torch.zeros_like(model_impt_sorted_as_gt_diff),
    )
    model_impt_sorted_as_gt_diff = model_impt_sorted_as_gt_diff * pred_mask
    model_impt_sorted_as_gt_diff = (
        model_impt_sorted_as_gt_diff
        * hint_flags.unsqueeze(1).unsqueeze(2).float().cuda()
    )
    hint_loss = model_impt_sorted_as_gt_diff.sum(dim=1).mean()
    return hint_loss


def compute_zero_loss(opt, hint_scores, hint_flags, model_impt):
    # align feats with zero vector
    if opt.normalize_FI:
        model_impt = F.normalize(model_impt)
    mask = (hint_scores <= opt.impt_threshold).float().squeeze()  # masking out impt objs
    all_zeros = torch.zeros(model_impt.size()).cuda()
    loss_zero = compute_alignment_loss(opt, 
                                       model_impt * mask * hint_flags.unsqueeze(-1).cuda().float(), 
                                       all_zeros)
    return loss_zero.mean()

def compute_alignment_loss(opt, output1, output2):
    # align two vectors
    assert(output1.shape == output2.shape)
    if opt.align_loss_type == "kl":
        # make sure that both are log_probabilities
        return F.kl_div(output1, output2, log_target=True)
    elif opt.align_loss_type == "l1":
        loss_func = nn.L1Loss()
    elif opt.align_loss_type == "l2":
        loss_func = nn.MSELoss()
    elif opt.align_loss_type == "cossim":
        loss_func = nn.CosineEmbeddingLoss()
        cur_batch_size = output1.size(0)
        return loss_func(output1, output2, torch.ones(cur_batch_size).float().cuda())
    else:
        raise ValueError(f"unknown alignment loss type: {opt.align_loss_type}")

    return loss_func(output1, output2)
    
def compute_loss(
    opt,
    train_loader,
    epoch,
    iter_num,
    objs,
    answers,
    logits,
    ans_idxs,
    hint_flags,
    hint_scores,
    ans_cossim,
    model_impt,
    batch_mask,
):
    # init
    cur_iter_total = len(train_loader)*epoch + iter_num
    if opt.random_suff or opt.random_unc or opt.random_inv_FI or opt.random_align:
        hint_scores_random = hint_scores[1]
        hint_scores = hint_scores[0]
    def add_new_loss(_new_loss, _loss_weight):
        nonlocal loss
        nonlocal msg
        loss = loss + _new_loss * _loss_weight
        msg += " , new loss = %.3f " % (_new_loss.item() * _loss_weight)
        
    if opt.use_input_mask:
        loss = instance_bce_with_logits(
            opt, logits * batch_mask.unsqueeze(-1), answers * batch_mask.unsqueeze(-1)
        )
    elif opt.aug_type in ["suff-human", "suff-random"]:
        full_batch_size = logits.size(0)
        assert full_batch_size % 2 == 0
        cur_batch_size = int(full_batch_size / 2)
        
        loss = instance_bce_with_logits(
            opt, logits[:cur_batch_size] * batch_mask.unsqueeze(-1), answers * batch_mask.unsqueeze(-1)
        )
        new_loss = instance_bce_with_logits(
            opt, logits[cur_batch_size:cur_batch_size*2] * batch_mask.unsqueeze(-1), answers * batch_mask.unsqueeze(-1)
        )
        loss_weight = opt.aug_loss_weight
        # add new loss
        msg = f"iter {iter_num}/{len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss.item())
        add_new_loss(new_loss, loss_weight)
    
    elif opt.aug_type in ["invariance", "saliency-guided"]: # align batch 0 and batch 1
        full_batch_size = logits.size(0)
        assert full_batch_size % 2 == 0
        cur_batch_size = int(full_batch_size / 2)
        
        loss = instance_bce_with_logits(
            opt, logits[:cur_batch_size] * batch_mask.unsqueeze(-1), answers * batch_mask.unsqueeze(-1)
        )
        # align batch 0 and batch 1
        assert(opt.align_loss_type in ['kl', 'l2'])
        if opt.align_loss_type == 'kl': # on log prob space
            if opt.vqa_loss_type=='softmax':
                input_0 = F.log_softmax(logits[:cur_batch_size], dim=1)
                input_1 = F.log_softmax(logits[cur_batch_size:cur_batch_size*2], dim=1)
            else:
                input_0 = F.logsigmoid(logits[:cur_batch_size])
                input_1 = F.logsigmoid(logits[cur_batch_size:cur_batch_size*2])
        elif opt.align_loss_type == 'l2': # on logit space
            input_0 = logits[:cur_batch_size]
            input_1 = logits[cur_batch_size:cur_batch_size*2]
            
        new_loss = compute_alignment_loss(
            opt,
            input_0 * batch_mask.unsqueeze(-1), 
            input_1 * batch_mask.unsqueeze(-1)
        )
        loss_weight = opt.alignment_loss_weight
        # add new loss
        msg = f"iter {iter_num}/{len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss.item())
        add_new_loss(new_loss, loss_weight)
        
    elif opt.aug_type in ["uncertainty-NonImptOnly"]: # align batch 1 and batch 2
        full_batch_size = logits.size(0)
        assert full_batch_size % 3 == 0
        cur_batch_size = int(full_batch_size / 3)
        
        # calc vqa loss
        loss = instance_bce_with_logits(
            opt, logits[:cur_batch_size], answers
        )
        # calc additional loss - align batch 1 and batch 2
        if opt.vqa_loss_type=='softmax':
            input_0 = F.log_softmax(logits[cur_batch_size : cur_batch_size * 2], dim=1)
            input_1 = F.log_softmax(logits[cur_batch_size * 2 : cur_batch_size * 3], dim=1)
        else:
            assert(opt.vqa_loss_type=='sigmoid')
            input_0 = F.logsigmoid(logits[cur_batch_size : cur_batch_size * 2])
            input_1 = F.logsigmoid(logits[cur_batch_size * 2 : cur_batch_size * 3])
        new_loss = compute_alignment_loss(
            opt,
            input_0 * batch_mask.unsqueeze(-1),
            input_1 * batch_mask.unsqueeze(-1),
        )
        loss_weight = opt.alignment_loss_weight
        # add new loss
        msg = f"iter {iter_num}/{len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss.item())
        add_new_loss(new_loss, loss_weight)
        
    elif opt.aug_type == "uncertainty-uniform": # align batch 1 with uniform dist
        # get batch size
        full_batch_size = logits.size(0)
        assert full_batch_size % 2 == 0
        cur_batch_size = int(full_batch_size / 2)
        # calc vqa loss
        loss = instance_bce_with_logits(
            opt, logits[:cur_batch_size], answers
        )
        # calc additional loss ## f(x*nonimpt_mask) = uniform
        # both on log prob space, since uniform cannot be on logit space
        assert(opt.align_loss_type in ['kl', 'l2']) 
        # get log prob for output
        if opt.vqa_loss_type=='softmax':
            output_prob = F.log_softmax(logits[cur_batch_size : cur_batch_size * 2], dim=1)
            # align with uniform dist
            uniform_output = torch.ones(answers.size()).cuda()
            uniform_output = F.normalize(uniform_output, p=1)
            uniform_output = torch.log(uniform_output)
        else:
            assert(opt.vqa_loss_type=='sigmoid')
            output_prob = F.logsigmoid(logits[cur_batch_size : cur_batch_size * 2])
            uniform_output = torch.zeros(answers.size()).cuda()
            # uniform_output = torch.ones(answers.size()).cuda() * 0.5 # align with 0.5 tensor
        new_loss = compute_alignment_loss(
            opt,
            output_prob * batch_mask.unsqueeze(-1),
            uniform_output * batch_mask.unsqueeze(-1),
        )
        loss_weight = opt.alignment_loss_weight
        # add new loss
        msg = f"iter {iter_num}/{len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss.item())
        add_new_loss(new_loss, loss_weight)
    
    elif opt.aug_type in ["suff-uncertainty"]: # batch 1 with y; align batch 2 with some dist
        # get batch size
        full_batch_size = logits.size(0)
        assert full_batch_size % 3 == 0
        cur_batch_size = int(full_batch_size / 3)
        batch_mask_1 = batch_mask.split(cur_batch_size)[0].unsqueeze(-1)
        batch_mask_2 = batch_mask.split(cur_batch_size)[1].unsqueeze(-1)
        
        # calc vqa loss
        loss = instance_bce_with_logits(
            opt, logits[:cur_batch_size], answers
        )
        if opt.OBJ11: # ugly hack for OBJ11
            opt.aug_loss_weight = 1
        
        # calc additional loss 1 # f(x*impt_mask) = gt
        new_loss = instance_bce_with_logits(
            opt, 
            logits[cur_batch_size:cur_batch_size*2] * batch_mask_1, 
            answers * batch_mask_1
        )
        loss_weight = opt.aug_loss_weight
        # add new loss
        msg = f"iter {iter_num}/{len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss.item())
        add_new_loss(new_loss, loss_weight)
        
        # calc additional loss 2 # f(x*nonimpt_mask) = uniform
        if opt.aug_type == "suff-uncertainty":
            # in log prob space
            if opt.vqa_loss_type=='softmax':
                output_prob = F.log_softmax(logits[cur_batch_size*2 : cur_batch_size*3], dim=1)
                # in log prob space
                uniform_output = torch.ones(answers.size()).cuda()
                uniform_output = F.normalize(uniform_output, p=1)
                uniform_output = torch.log(uniform_output)
            else:
                assert(opt.vqa_loss_type=='sigmoid')
                output_prob = F.logsigmoid(logits[cur_batch_size*2 : cur_batch_size*3])
                uniform_output = torch.zeros(answers.size()).cuda()
                # uniform_output = torch.ones(answers.size()).cuda() * 0.5
            if opt.OBJ11: # ugly hack for OBJ11
                opt.alignment_loss_weight = 1
                opt.align_loss_type = 'kl'
            new_loss = compute_alignment_loss(
                opt,
                output_prob * batch_mask_2,
                uniform_output * batch_mask_2,
            )
            loss_weight = opt.alignment_loss_weight
        # add new loss
        add_new_loss(new_loss, loss_weight)
    
    ## no augmentation
    elif opt.aug_type == "none":
        loss = instance_bce_with_logits(opt, logits, answers)  
        msg = f"iter {iter_num}/{len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss.item())
    else:
        raise ValueError(f"unsuported augmentation method {opt.aug_type}")
    
    ## other new loss terms
    if opt.use_zero_loss:
        if opt.OBJ11: # ugly hack for OBJ11
            if opt.model_type == 'lxmert':
                opt.zero_loss_weight = 1e-3
            else: # updn
                if 'hat' in opt.dataset:
                    opt.zero_loss_weight = 0.1
                else:
                    opt.zero_loss_weight = 1
            opt.align_loss_type = 'l1'
        # support controlled random
        if opt.random_inv_FI:
            new_loss = compute_zero_loss(opt, hint_scores_random, hint_flags, model_impt)
        else:
            new_loss = compute_zero_loss(opt, hint_scores, hint_flags, model_impt)
        loss_weight = opt.zero_loss_weight
        add_new_loss(new_loss, loss_weight)
    
    if opt.use_direct_alignment:
        if opt.OBJ11: # ugly hack for OBJ11
            if opt.model_type == 'lxmert':
                opt.alignment_loss_weight = 1e-5
            else:
                if 'hat' in opt.dataset:
                    opt.alignment_loss_weight = 0.1
                else:
                    opt.alignment_loss_weight = 1
            opt.align_loss_type = 'cossim'
        # support controlled random
        if opt.random_align:
            new_loss = compute_alignment_loss(opt, model_impt, hint_scores_random.squeeze())
        else:
            new_loss = compute_alignment_loss(opt, model_impt, hint_scores.squeeze())
        loss_weight = opt.alignment_loss_weight
        add_new_loss(new_loss, loss_weight)
    
    if opt.use_hint_loss:
        new_loss = compute_hint_loss(
            opt, objs, answers, logits, hint_scores, hint_flags, model_impt
        )
        loss_weight = opt.hint_loss_weight
        add_new_loss(new_loss, loss_weight)
        
    if opt.use_scr_loss:
        loss_scr_hint, loss_scr_compare, loss_scr_reg = compute_scr_loss(
            opt,
            objs,
            answers,
            ans_idxs,
            logits,
            hint_flags,
            hint_scores,
            ans_cossim,
        )
        add_new_loss(loss_scr_hint, 1)
        add_new_loss(loss_scr_compare, 1)

    if opt.print_every_batch or iter_num % 50 == 0:
        print(msg)
    return loss