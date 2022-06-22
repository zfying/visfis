from __future__ import print_function

import json
import os
import pdb
import pickle
import pickle as cPickle
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing.preprocess_answer import preprocess_answer
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data.dataloader import DataLoader

import transformers

from components import losses, metrics
from components.optimizers import BertAdam
from components import feature_impt

def create_optim(opt, model):
    if opt.optimizer == "adadelta":
        optim = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=1e-6, weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "RMSprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=opt.weight_decay,
            momentum=0,
            centered=False,
        )
    elif opt.optimizer == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    return optim


def run(model, train_loader, eval_loader_all, eval_loader_all_2, opt):
    """Contains the main training loop and test logic.
    Also, handles saving/loading of checkpoints
    """

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    _model_checkpoints_path = os.path.join(opt.checkpoint_path, "model_checkpoints")
    if not os.path.exists(_model_checkpoints_path):
        os.makedirs(_model_checkpoints_path)
        
    # setup optimizer & tokenizer
    if opt.model_type == 'lxmert': 
        # init optm
        if not opt.calc_dp_level_metrics:
            from components.optimizers import BertAdam
            batch_per_epoch = len(train_loader)
            t_total = int(batch_per_epoch * opt.max_epochs)
            print("BertAdam Total Iters: %d" % t_total)
            optim = BertAdam(list(model.parameters()),
                             lr=opt.learning_rate,
                             warmup=0.1,
                             t_total=t_total)
        tokenizer = transformers.LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    else: # optim for UpDn
        tokenizer = None
        if opt.learning_rate == None:
            optim = getattr(torch.optim, opt.optimizer)(
                filter(lambda p: p.requires_grad, model.parameters())
            )
        else:
            optim = getattr(torch.optim, opt.optimizer)(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.learning_rate,
            )

    ## Preliminary setup
    best_eval_score = 0
    best_eval_score_ood = 0
    best_eval_epoch = None
    best_eval_score_2_id = 0
    best_eval_score_2 = 0
    best_eval_epoch_2 = None
    ans_cossim = pickle.load(open(f"{opt.data_dir}/ans_cossim.pkl", "rb"))
    # logger
    log_file = open(opt.checkpoint_path + "/log.txt", "a")
    if not opt.test and not opt.calc_dp_level_metrics:
        print(json.dumps(vars(opt), indent=4, sort_keys=True), file=log_file)
        log_file.flush()
    

    # If load_checkpoint_path flag is specified, then we need to load model from that state
    if opt.load_checkpoint_path is not None and len(opt.load_checkpoint_path) > 0:
        ckpt = torch.load(os.path.join(opt.load_checkpoint_path))
        if "epoch" in ckpt:
            states_ = ckpt["model_state_dict"]
        else:
            states_ = ckpt

        model.load_state_dict(states_)

    # handle calculate rank correlation
    if opt.calc_dp_level_metrics:
        print("Calculating data point level metrics ...")
        metrics.calc_dp_level_metrics(model, tokenizer, eval_loader_all, opt, log_file)
        return

    # The main training loop
    for epoch in range(opt.max_epochs):

        print(f"Training epoch {epoch}...")
        iter_num = 0
        train_score = 0
        train_start_time = time.time()

        ## ramen optim setup
        if opt.lr_type == "ramen":
            if epoch < len(gradual_warmup_steps):
                optim.param_groups[0]["lr"] = gradual_warmup_steps[epoch]
            elif epoch in lr_decay_epochs:
                optim.param_groups[0]["lr"] *= lr_decay_rate
            print("lr {}".format(optim.param_groups[0]["lr"]))
        
        # through batches
        for (
            objs,
            qns,
            answers,
            hint_scores,
            question_ids,
            image_ids,
            hint_flags,
            q_ori,
            a_ori,
        ) in iter(train_loader):
            if opt.change_scores_every_epoch:
                # Assign random scores every epoch, if the flag says to do so.
                hint_scores = torch.rand(hint_scores.shape).cuda()

            objs = objs.cuda().float().requires_grad_()  # B x num_objs x emb
            cur_batch_size = objs.size(0)
            qns = qns.cuda().long()  # B x len
            answers = answers.cuda()  # B x num classes
            if opt.random_suff or opt.random_unc or opt.random_inv_FI or opt.random_align:
                hint_scores[0] = hint_scores[0].cuda().float()
                hint_scores[1] = hint_scores[1].cuda().float()
            else:
                hint_scores = hint_scores.cuda().float()  # B x num_objs x 1
            ## input mask ##
            if opt.use_input_mask:
                if opt.mask_type == "impt":
                    mask = (hint_scores > opt.impt_threshold).float()
                    batch_mask = (mask.sum(dim=(1, 2)) > 0).float()
                    objs = objs * mask
                elif opt.mask_type == "non_impt":
                    mask = (hint_scores <= opt.impt_threshold).float()
                    batch_mask = (mask.sum(dim=(1, 2)) > 0).float()
                    objs = objs * mask
                else:
                    raise ValueError("unsupported mask type")
                        
            # forward (including augmentation & getting FI)
            logits, ans_idxs, model_impt, batch_mask = feature_impt.FI_forward(opt, model, tokenizer, objs, qns, answers, hint_scores, hint_flags, q_ori)

            ## add train score
            batch_score = float(losses.compute_score_with_logits(
                logits[:cur_batch_size], answers[:cur_batch_size].data
            ).sum())
            train_score += float(batch_score)

            # compute loss
            loss = losses.compute_loss(
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
                batch_mask
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optim.step()
            optim.zero_grad()
            log_file.flush()
            iter_num += 1
        # free memory
        del logits, ans_idxs, model_impt, batch_mask, loss
        del objs, qns, answers, hint_scores
        torch.cuda.empty_cache()
        
        final_train_score = train_score / len(train_loader.dataset)
        train_time = time.time() - train_start_time
        print(
            f"train score for epoch[{epoch}] is, score = %.3f, after %.3f"
            % (final_train_score, train_time)
        )
        print(
            f"train score for epoch[{epoch}] is, score = %.3f, after %.3f"
            % (final_train_score, train_time),
            file=log_file,
        )
        print("##\n")

        # for neg analysis optim setup
        if opt.lr_type == "neg_analysis":
            lr_scheduler.step()
            print(f"lr {lr_scheduler.get_lr()}")

        eval_score = evaluate_and_log(
            "Eval",
            model,
            tokenizer,
            eval_loader_all,
            opt,
            epoch,
            log_file
        )
        log_file.flush()
        if opt.use_two_testsets:
            eval_score_2 = evaluate_and_log(
                "Eval 2",
                model,
                tokenizer,
                eval_loader_all_2,
                opt,
                epoch,
                log_file
            )
        log_file.flush()
        # Save model
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
        }
        if eval_score > best_eval_score:
            torch.save(
                state,
                os.path.join(
                    opt.checkpoint_path, opt.saved_model_prefix + "model-best.pth"
                ),
            )
            best_eval_score = eval_score
            if opt.use_two_testsets:
                best_eval_score_ood = eval_score_2
            best_eval_epoch = epoch
        if opt.use_two_testsets:
            if eval_score_2 > best_eval_score_2:
                torch.save(
                    state,
                    os.path.join(
                        opt.checkpoint_path, opt.saved_model_prefix + "model-best_2.pth"
                    ),
                )
                best_eval_score_2_id = eval_score
                best_eval_score_2 = eval_score_2
                best_eval_epoch_2 = epoch
        # save each epoch
        if opt.save_every_epoch:
            torch.save(
                state,
                os.path.join(
                    _model_checkpoints_path, opt.saved_model_prefix + f"model-epoch{epoch}.pth"
                ),
            )
                
    print(f"best val score - epoch {best_eval_epoch} - id: %.4f, ood: %.4f" % (best_eval_score, best_eval_score_ood), file=log_file)
    if opt.use_two_testsets:
        print(f"best val 2 score - epoch {best_eval_epoch_2} - id: %.4f, ood: %.4f" % (best_eval_score_2_id, best_eval_score_2), file=log_file)

def predict(model, dataloader, opt):
    dataroot = "data"
    label2ans_path = os.path.join(dataroot, "processed", "trainval_label2ans.pkl")
    label2ans = cPickle.load(open(label2ans_path, "rb"))
    results = []
    for (
        objs,
        qns,
        gt_answers,
        hintscore,
        _,
        qid,
        image_id,
        hint_flag,
        q_ori,
        a_ori,
    ) in tqdm(iter(dataloader)):
        for _a, _qid in zip(gt_answers, qid):
            _a = int(torch.argmax(_a))
            _qid = int(_qid)
            results.append({"question_id": _qid, "answer": label2ans[_a]})
    json.dump(results, open(os.path.join(opt.predict_checkpoint, "scr.json"), "w"))


def compute_gt_ans_sensitivities(objs, gt_answers, logits):
    sensitivities = torch.autograd.grad(
        (logits * (gt_answers > 0).float()).sum(), objs, create_graph=True
    )[0]
    sensitivities = sensitivities.sum(2)
    return sensitivities


def compute_all_ans_sensitivities(objs, logits):
    sensitivities = torch.autograd.grad(logits.sum(), objs, create_graph=True)[0]
    sensitivities = sensitivities.sum(2)
    return sensitivities


def evaluate(
    model,
    tokenizer,
    dataloader,
    opt,
    epoch=None,
):
    model.eval()
    score = 0
    scorek = 0
    V_loss = 0

    upper_bound = 0
    num_data = 0
    qid_to_logits = {}
    qid_to_prediction_scores = (
        {}
    )  # 0 if prediction is incorrect and the GT softscore if it is correct
    qid_to_human_agreement = {}
    qid_to_gt_ans_sensitivities = {}
    qid_to_all_ans_sensitivities = {}

    for (
        objs,
        qns,
        answers,
        hint_scores,
        question_ids,
        image_ids,
        hint_flags,
        q_ori,
        a_ori,
    ) in tqdm(iter(dataloader)):
        objs = objs.cuda().float()
        qns = qns.cuda().long()
        answers = answers.cuda()  # true labels
        if opt.random_suff or opt.random_unc or opt.random_inv_FI or opt.random_align:
            hint_scores = hint_scores[0].cuda().float()
        else:
            hint_scores = hint_scores.cuda().float()  # B x num_objs x 1

        ## input mask ##
        batch_mask = torch.ones(hint_scores.size(0)).cuda()
        if opt.use_input_mask:
            if opt.mask_type == "impt":
                mask = (hint_scores > opt.impt_threshold).float()
                batch_mask = (mask.sum(dim=(1, 2)) > 0).float()
                objs = objs * mask
            elif opt.mask_type == "non_impt":
                mask = (hint_scores <= opt.impt_threshold).float()
                batch_mask = (mask.sum(dim=(1, 2)) > 0).float()
                objs = objs * mask
            else:
                raise ValueError("unsupported mask type")
        ## end input mask ##
        with torch.no_grad():
            _, logits, _, _ = feature_impt.forward(opt, model, tokenizer, objs, qns, q_ori)
        
        batch_score = losses.compute_score_with_logits(
            logits * batch_mask.unsqueeze(-1), (answers * batch_mask.unsqueeze(-1)).data
        ).sum()
        batch_scorek = losses.compute_score_with_k_logits(
            logits * batch_mask.unsqueeze(-1), (answers * batch_mask.unsqueeze(-1)).data
        ).sum()
        score += batch_score
        scorek += batch_scorek

        upper_bound += (answers.max(1)[0]).sum()
        num_data += logits.size(0)

    score = score / len(dataloader.dataset)
    scorek = scorek / len(dataloader.dataset)

    upper_bound = upper_bound / len(dataloader.dataset)
    model.train()
    return score, upper_bound, None, scorek


def evaluate_and_log(
    key,
    model,
    tokenizer,
    dataloader,
    opt,
    epoch,
    log_file
):
    print(f"Evaluating {key} ...")
    val_start_time = time.time()
    score, _, _, scorek = evaluate(
        model,
        tokenizer,
        dataloader,
        opt=opt,
        epoch=epoch
    )
    val_time = time.time() - val_start_time
    print(
        f"{key} (epoch {epoch}), score = %.3f, score_k = %.3f, after %.3f"
        % (score, scorek, val_time)
    )
    print(
        f"{key} (epoch {epoch}), score = %.3f, score_k = %.3f, after %.3f"
        % (score, scorek, val_time),
        file=log_file,
    )
    return score
