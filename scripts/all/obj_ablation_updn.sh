SAVE_DIR=./saved_models_xaicps
DATA_DIR=../data/neg_data_xaicps
dataset=clevrxai
full_dataset=clevrxaicsp

NUM=0
expt_num=exp0

## OBJ2 - Saliency-Guided-Training
expt=${full_dataset}_OBJ2_SaliencyGuided_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--aug_type saliency-guided \
--align_loss_type kl \
--alignment_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt


## OBJ3 - Align - EG-pred-1sample
expt=${full_dataset}_OBJ3_Cossim_EGPred1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--model_importance expected_gradient \
--FI_predicted_class True \
--num_sample_eg 1 \
--use_direct_alignment \
--align_loss_type cossim \
--alignment_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt


## OBJ4 - Suff-human
expt=${full_dataset}_OBJ4_SuffHuman_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--aug_type suff-human \
--aug_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

## OBJ5 - Align + Suff-human - EG-pred-1sample
expt=${full_dataset}_OBJ5_CossimSuffHuman_EG1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--model_importance expected_gradient \
--FI_predicted_class True \
--num_sample_eg 1 \
--use_direct_alignment \
--align_loss_type cossim \
--alignment_loss_weight 1 \
--aug_type suff-human \
--aug_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

## OBJ6 - Invariance
expt=${full_dataset}_OBJ6_AugInvariance_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--aug_type invariance \
--align_loss_type kl \
--alignment_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

## OBJ7 - Invariance-FI=0
# tuning -> KOI-gt
expt=${full_dataset}_OBJ7_zero_KOI1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--model_importance KOI \
--FI_predicted_class False \
--num_sample_omission 1 \
--use_zero_loss \
--align_loss_type l1 \
--zero_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt


## OBJ8 - Uncertainty
expt=${full_dataset}_OBJ8_uncertainty_kl_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--aug_type uncertainty-uniform \
--align_loss_type kl \
--alignment_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

## OBJ9 - Sufficiency + Uncertainty
expt=${full_dataset}_OBJ9_SuffUncertainty_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--aug_type suff-uncertainty \
--align_loss_type kl \
--aug_loss_weight 1 \
--alignment_loss_weight 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

## OBJ10 - Align-Cos + Invariance-FI
expt=${full_dataset}_OBJ10_AlignFI0_KOI1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--OBJ11 \
--use_zero_loss \
--use_direct_alignment \
--model_importance KOI \
--FI_predicted_class False \
--num_sample_omission 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

expt=${full_dataset}_OBJ10_AlignFI0_EGPred1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--OBJ11 \
--use_zero_loss \
--use_direct_alignment \
--model_importance expected_gradient \
--FI_predicted_class True \
--num_sample_eg 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

## OBJ11 - Align-Cos + Sufficiency + Invariance-FI + Uncertainty
expt=${full_dataset}_OBJ11_SuffUncertAlignFI0_KOI1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--OBJ11 \
--aug_type suff-uncertainty \
--use_zero_loss \
--use_direct_alignment \
--model_importance KOI \
--FI_predicted_class False \
--num_sample_omission 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

expt=${full_dataset}_OBJ11_SuffUncertAlignFI0_EGPred1sample_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--impt_threshold ? \
--OBJ11 \
--aug_type suff-uncertainty \
--use_zero_loss \
--use_direct_alignment \
--model_importance expected_gradient \
--FI_predicted_class True \
--num_sample_eg 1 \
--batch_size 64 \
--seed 7 \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--use_two_testsets \
--split train \
--split_test dev \
--split_test_2 test-ood \
--max_epochs 50 \
--checkpoint_path ${SAVE_DIR}/${expt} \
> ${SAVE_DIR}/${expt}/verbose_log.txt

