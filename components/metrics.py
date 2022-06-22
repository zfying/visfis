from __future__ import print_function

import os
import time
import pickle 
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import softmax
from sklearn.metrics import average_precision_score

from components import feature_impt

def generate_mask(inputs, mask_type, percentage):
    ## generate mask to keep or remove top k% of feats
    inputs_sort, inputs_ind = inputs.sort(1, descending=True)
    cut_off_index = int(inputs.size(1)*percentage)
    cut_off_values = inputs_sort[:, cut_off_index : cut_off_index+1] # [Batch, 1]
    
    if mask_type == 'keep_top':
        mask_cutoff = (inputs>cut_off_values).float() # [Batch, num_objs]
        mask_positive = (inputs>0).float() # [Batch, num_objs]
        mask = mask_positive * mask_cutoff # only keep positive FI (AND)
    elif mask_type == 'remove_top':
        mask_cutoff = (inputs<=cut_off_values).float() # [Batch, num_objs]
        mask_negative = (inputs<=0).float() # [Batch, num_objs]
        mask = torch.logical_or(mask_negative, mask_cutoff).float() # only keep positive FI (OR)
    
    mask = mask.unsqueeze(-1) #[Batch, #objs, 1]
    return mask

def create_csv(opt, metrics):
    ## create .pkl file of a pandas dataframe for all metrics
    # get qtype
    if opt.dataset == 'xaicp':
        # get qtype for xai
        _path = './data/xaicp/questions/test-id_annotations.json'
        test_id_anns = json.load(open(_path))['annotations']
        _path = './data/xaicp/questions/test-ood_annotations.json'
        test_ood_anns = json.load(open(_path))['annotations']
        qid2qtype = {}
        for ann in test_id_anns:
            qid2qtype[ann['question_id']] = ann['question_type']
        for ann in test_ood_anns:
            qid2qtype[ann['question_id']] = ann['question_type']
    elif opt.dataset == 'hatcp':
        # get qtype for hat
        _path = './data/hatcp/questions/test-id_annotations.json'
        test_id_anns = json.load(open(_path))['annotations']
        _path = './data/hatcp/questions/test-ood_annotations.json'
        test_ood_anns = json.load(open(_path))['annotations']
        qid2qtype = {}
        for ann in test_id_anns:
            qid2qtype[ann['question_id']] = ann['answer_type']
        for ann in test_ood_anns:
            qid2qtype[ann['question_id']] = ann['answer_type']
            
    # csv columns
    column_names = ["dataset", "split", "model_type", "model_name", "seed",
               "qid", "qtype", "gt_answers", "output_pred", "output_gt", 
                "human_impt_max", "human_impt_min", "acc", 
                "RRR_suff", "RRR_inv", "RRR_unc_pred", "RRR_unc_gt",
               "FI_method", "suff_model", "comp_model", "plau_rank_corr", "plau_iou"]
    
    # init variables
    if opt.FI_predicted_class:
        FI_method = opt.model_importance+'_pred'
    else:
        FI_method = opt.model_importance+'_gt'
    dataset = opt.dataset
    model_type = opt.model_type
    split = opt.split_test
    model_name = opt.checkpoint_path
    seed = opt.seed
    
    # iter through metrics
    all_data = []
    for qid in metrics["gt_answers"]:
        # qtype
        if dataset in ['xaicp', 'hatcp']:
            _qtype = qid2qtype[qid]
        else:
            _qtype = ""
        # suff/comp/unc/plau
        _suff_model = metrics["suff_model"][qid][0.1] + metrics["suff_model"][qid][0.25] + metrics["suff_model"][qid][0.5]
        _suff_model /= 3
        _comp_model = metrics["comp_model"][qid][0.1] + metrics["comp_model"][qid][0.25] + metrics["comp_model"][qid][0.5]
        _comp_model /= 3
        _plau_iou = metrics["plau_iou"][qid][0.1] + metrics["plau_iou"][qid][0.25] + metrics["plau_iou"][qid][0.5]
        _plau_iou /= 3
        _unc_prob = softmax(metrics["RRR_unc"][qid])
        _unc_prob_pred = _unc_prob.max()
        _unc_prob_gt = (_unc_prob * metrics["gt_answers"][qid]).sum()
        # ans
        _gt_answer = metrics["gt_answers"][qid].argmax()
        # probability output
        _output_prob = softmax(metrics["model_outputs"][qid])
        _prob_pred = _output_prob.max()
        _prob_gt = (_output_prob * metrics["gt_answers"][qid]).sum()
        # human impt
        _human_impt_max = metrics["human_impt"][qid].max()
        _human_impt_min = metrics["human_impt"][qid].min()

        column_names = ["dataset", "split", "model_type", "model_name", "seed",
                   "qid", "qtype", "gt_answers", "output_pred", "output_gt", 
                    "human_impt_max", "human_impt_min", "acc", 
                    "RRR_suff", "RRR_inv", "RRR_unc_pred", "RRR_unc_gt",
                    "FI_method", "suff_model", "comp_model", "plau_rank_corr", "plau_iou"]

        new_row = [dataset, split, model_type, model_name, seed, 
                   qid, _qtype, int(_gt_answer), float(_prob_pred), float(_prob_gt),
                   float(_human_impt_max), float(_human_impt_min), float(metrics["accuracy"][qid]), 
                   float(metrics["RRR_suff"][qid]),  float(metrics["RRR_inv"][qid]), 
                   float(_unc_prob_pred), float(_unc_prob_gt), FI_method, 
                   float(_suff_model), float(_comp_model), float(metrics["plau_rank_corr"][qid]), 
                   float(_plau_iou)]
        all_data.append(new_row)
        
    # save
    df = pd.DataFrame(all_data, columns = column_names)
    _path = os.path.join(opt.checkpoint_path, 
                         opt.saved_model_prefix+opt.split_test+
                         '_'+opt.model_importance+"_gt_metrics.pkl") 
    df.to_pickle(_path)
        
def calc_dp_level_metrics(model, tokenizer, dataloader, opt, log_file):
    model.eval()
    if opt.vqa_loss_type == 'softmax':
        to_prob_func = nn.Softmax(dim=1)
    else:
        to_prob_func = nn.Sigmoid()
    
    ## results to record 
    # PART 1 
    acc_counter = 0
    gt_answers = {} # dict(qid, array)
    model_outputs = {} # dict(qid, array)
    human_impt = {} # dict(qid, array)
    accuracy = {} # dict(qid, binary acc) boolean
    # PART 2
    RRR_suff = {} # dict (qid, binary acc) boolean
    RRR_inv = {} # dict (qid, avg acc) float
    RRR_unc = {} # dict (qid, array) model output array(28,)
    # PART 3
    model_impt_results = {} # dict(qid, array)
    suff_model = {} # dict(qid, dict), dict(percentage, float)
    comp_model = {} # dict(qid, dict), dict(percentage, float)
    plau_rank_corr = {} # dict(qid, float)
    plau_iou = {} # dict(qid, float)
    
    # iter batches
    for objs, qns, answers, hint_scores, question_ids, image_ids, hint_flags, q_ori, a_ori in tqdm(iter(dataloader)):
        cur_batch_size = objs.size(0)
        # prep data
        objs = objs.cuda().float().requires_grad_()
        qns = qns.cuda().long()
        answers = answers.cuda()  # true labels
        hint_scores = hint_scores.cuda().float()  # B x num_objs x 1
        
        ### PART 1: original forward
        with torch.no_grad():
            _, logits, _, ans_idxs = feature_impt.forward(opt, model, tokenizer, objs, qns, q_ori)
        prob_original = to_prob_func(logits)
        predicted_ans = prob_original.ge(prob_original.max(-1, keepdim=True)[0]) # stritly greater
        
        # RECORD: gt_answer, model_outputs
        for index, qid in enumerate(question_ids): # iter through index
            qid = int(qid)
            # assert(qid not in model_outputs)
            gt_answers[qid] = answers[index].cpu().detach().numpy()
            if not opt.ACC_only:
                model_outputs[qid] = logits[index].cpu().detach().numpy()
            human_impt[qid] = hint_scores[index].squeeze().cpu().detach().numpy()
            accuracy[qid] = answers[index][logits[index].cpu().detach().numpy().argmax()].cpu().detach().numpy()
            acc_counter += accuracy[qid]
        
        
        ### PART 2: impt & nonimpt forward
        if not opt.ACC_only:
            qns_new = qns.repeat(2 + 3, 1) # 1 for suff, 1 for unc, 3 for inv
            objs_new = objs.repeat(2 + 3, 1, 1)
            q_ori_new = q_ori * (2+3)
            ## get masks
            mask_impt = (hint_scores >opt.impt_threshold).float()  # Impt mask
            mask_nonimpt = (hint_scores <=opt.impt_threshold).float()  # NonImpt mask
            batch_mask_1 = (mask_impt.sum(dim=(1, 2)) > 0).float()
            batch_mask_2 = (mask_nonimpt.sum(dim=(1, 2)) > 0).float()
            # inv mask * 3
            masks_inv = []
            batch_masks_inv = []
            for index_inv in range(3):
                # randomly select nonimpt objs - uniform over num of selected objs 
                mask_count = mask_nonimpt.sum(dim=(1,2))
                mask_uniform_nonimpt = torch.zeros(hint_scores.size()).cuda()
                for index, count in enumerate(mask_count):
                    prob = torch.randint(int(count)+1, (1,)).cuda() / count
                    mask_uniform_nonimpt[index] = (torch.rand((opt.num_objects,1)).cuda() <= prob).float().cuda()
                # combine with impt objs
                mask = torch.logical_or(mask_impt, mask_uniform_nonimpt)
                # if all objs are non-impt, ignore
                mask_ignore = (mask_nonimpt.sum(dim=(1, 2)) != mask_nonimpt.size(1)).float()
                mask = mask * mask_ignore.unsqueeze(-1).unsqueeze(-1)
                masks_inv.append(mask)
                batch_mask = (mask.sum(dim=(1, 2)) > 0).float()
                batch_masks_inv.append(batch_mask)
            # apply mask
            objs_new[ : cur_batch_size] = feature_impt.apply_mask(opt, objs, mask_impt)
            objs_new[cur_batch_size: cur_batch_size * 2] = feature_impt.apply_mask(opt, objs, mask_nonimpt)
            for index_inv in range(3):
                objs_new[cur_batch_size*(2+index_inv):cur_batch_size*(3+index_inv)] = feature_impt.apply_mask(opt, objs, masks_inv[index_inv])
            # forward
            with torch.no_grad():
                _, logits_new, _, ans_idxs_new = feature_impt.forward(opt, model, tokenizer, objs_new, qns_new, q_ori_new)
            # RECORD: RRR_suff, RRR_unc, RRR_inv
            logits_impt_only = logits_new.split(cur_batch_size)[0]
            logits_nonimpt_only = logits_new.split(cur_batch_size)[1]
            logits_inv_0 = logits_new.split(cur_batch_size)[2]
            logits_inv_1 = logits_new.split(cur_batch_size)[3]
            logits_inv_2 = logits_new.split(cur_batch_size)[4]
            for index, qid in enumerate(question_ids): # iter through index
                qid = int(qid)
                # assert(qid not in RRR_suff)
                # RRR_inv & RRR_suff & RRR_unc
                RRR_unc[qid] = logits_nonimpt_only[index].cpu().detach().numpy()
                RRR_suff[qid] = answers[index][logits_impt_only[index].argmax()].cpu().detach().numpy()
                acc_0 = (logits_impt_only[index].argmax() == logits_inv_0[index].argmax()).float()
                acc_1 = (logits_impt_only[index].argmax() == logits_inv_1[index].argmax()).float()
                acc_2 = (logits_impt_only[index].argmax() == logits_inv_2[index].argmax()).float()
                RRR_inv[qid] = ((acc_0 + acc_1 + acc_2) / 3.0).cpu().detach().numpy()   
            # del objs_new, qns_new, logits_new, ans_idxs_new, _
            # del logits_impt_only, logits_nonimpt_only, logits_inv_0, logits_inv_1, logits_inv_2
            # del masks_inv, batch_masks_inv
            
            
        ### PART 3: model impt with budget=1000
        # get model impt - select from ['gradcam', 'expected_gradient', 'LOO', 'KOI', 'SHAP', 'avg_effect']
        # gradcam, LOO, KOI does not care about budget (1, 15/36, 15/36 respectively)
        # SHAP, avg_effect can have no gradient
        
        # forward
        if not opt.ACC_only and not opt.RRR_only:
            _, _, model_impt, _ = feature_impt.FI_forward(opt, model, tokenizer, objs, qns, answers, hint_scores, hint_flags, q_ori)
            # RECORD: model_outputs
            for index, qid in enumerate(question_ids): # iter through index
                qid = int(qid)
                model_impt_results[qid] = model_impt[index].cpu().detach().numpy()

            # for each percentage cut-off
            for percentage in opt.percentages: 
                # forward with masks
                with torch.no_grad():
                    ## based on model exp - suff - keep top 
                    assert(model_impt.size() == (cur_batch_size, opt.num_objects))
                    mask_model = generate_mask(model_impt, 'keep_top', percentage)
                    objs_model = feature_impt.apply_mask(opt, objs, mask_model)
                    _, logits_model_exp_suff, _, _ = feature_impt.forward(opt, model, tokenizer, objs_model, qns, q_ori)
                    prob_model_exp_suff = to_prob_func(logits_model_exp_suff)

                    ## based on model exp - comp - remove top 
                    mask_model = generate_mask(model_impt, 'remove_top', percentage)
                    objs_model = feature_impt.apply_mask(opt, objs, mask_model)
                    _, logits_model_exp_comp, _, _ = feature_impt.forward(opt, model, tokenizer, objs_model, qns, q_ori)
                    prob_model_exp_comp = to_prob_func(logits_model_exp_comp)

                # RECORD: suff_model, comp_model
                mask_model = generate_mask(model_impt, 'keep_top', percentage).squeeze()
                mask_human = (hint_scores>opt.impt_threshold).squeeze()
                for index, qid in enumerate(question_ids): # iter through index
                    qid = int(qid)
                    # init suff_model/comp_model for this FI method
                    if qid not in suff_model:
                        suff_model[qid] = {}
                        comp_model[qid] = {}
                        plau_iou[qid] = {}
                    # record suff_model/comp_model
                    suff_model[qid][percentage] = ((prob_original[index] - prob_model_exp_suff[index])*(predicted_ans[index]>0).float()).sum().cpu().detach().numpy()
                    comp_model[qid][percentage] = ((prob_original[index] - prob_model_exp_comp[index])*(predicted_ans[index]>0).float()).sum().cpu().detach().numpy()
                    # record plau_rank_corr
                    plau_rank_corr[qid] = spearmanr(hint_scores[index].squeeze().cpu().numpy(),
                                                   model_impt[index].detach().cpu().numpy())[0]
                    # record plau_iou
                    intersection = torch.logical_and(mask_model[index], mask_human[index]).cpu().numpy()
                    union = torch.logical_or(mask_model[index], mask_human[index]).cpu().numpy()
                    plau_iou[qid][percentage] = intersection.sum()/union.sum()

                    
    results = {'gt_answers': gt_answers,
               'model_outputs': model_outputs,
               'human_impt': human_impt,
               'accuracy': accuracy,
               'RRR_suff': RRR_suff,
               'RRR_inv': RRR_inv,
               'RRR_unc': RRR_unc,
               'model_impt_results': model_impt_results,
              'suff_model': suff_model,
              'comp_model': comp_model,
               'plau_rank_corr': plau_rank_corr,
              'plau_iou': plau_iou}
    
    print(f"{opt.checkpoint_path} {opt.split_test} acc: %.4f" % (acc_counter / len(results['accuracy'])))
    
    ## save all metrics
    create_csv(opt, results)
    
    
    # if opt.ACC_only:
    #     _path = os.path.join(opt.checkpoint_path, "../",
    #                          opt.saved_model_prefix+opt.split_test+"_ACC_metrics.pth")
    # elif opt.RRR_only:
    #     _path = os.path.join(opt.checkpoint_path, 
    #                          opt.saved_model_prefix+opt.split_test+"_RRR_metrics.pth")
    # else: # full
    #     if opt.FI_predicted_class:
    #         _path = os.path.join(opt.checkpoint_path,
    #                              opt.saved_model_prefix+opt.split_test+
    #                              '_'+opt.model_importance+"_pred_metrics.pth")
    #     else:
    #         _path = os.path.join(opt.checkpoint_path, 
    #                              opt.saved_model_prefix+opt.split_test+
    #                              '_'+opt.model_importance+"_gt_metrics.pth")  
    # with open(_path, 'wb') as file:
    #     pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    # return