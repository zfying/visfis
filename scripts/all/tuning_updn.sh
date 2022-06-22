SAVE_DIR=./saved_models_xaicp
DATA_DIR=./data/xaicp
dataset=xaicp

NUM=0

### TUNING 1 - learning rate
# baseline
for learning_rate in 1e-4 5e-4 1e-3 5e-3 1e-2; do
    expt=${dataset}_TUNING1_baseline_${learning_rate}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt}
    
    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --learning_rate ${learning_rate} \
    --do_not_discard_items_without_hints \
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
done

### TUNING 2 - .train() or .eval()
# align - train
expt=${dataset}_TUNING2_CossimTrain_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--model_importance LOO \
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

# align - eval
expt=${dataset}_TUNING2_CossimEval_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--train_eval_FI \
--hint_type hints \
--model_importance LOO \
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

### TUNING 3 - align weight
# align
for alignment_loss_weight in 0.1 1 10 100 1000; do
    expt=${dataset}_TUNING3_cossim_${alignment_loss_weight}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance LOO \
    --use_direct_alignment \
    --align_loss_type cossim \
    --alignment_loss_weight ${alignment_loss_weight} \
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
done

### TUNING 4 - sufficiency weight
# suff-human
for aug_loss_weight in 0.1 1 10 100 1000; do
    expt=${dataset}_TUNING4_SuffHuman_${aug_loss_weight}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --aug_type suff-human \
    --aug_loss_weight ${aug_loss_weight} \
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
done


### TUNING 5 - invariance weight: invariance aug
# aug - invariance
for alignment_loss_weight in 0.1 1 10 100 1000; do
    expt=${dataset}_TUNING5_AugInvariance_${alignment_loss_weight}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --aug_type invariance \
    --align_loss_type kl \
    --alignment_loss_weight ${alignment_loss_weight} \
    --saved_model_prefix ${alignment_loss_weight}_ \
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
done

### TUNING 5 - invariance weight: FI=0
# # zero loss
for zero_loss_weight in 0.1 1 10 100 1000; do
    expt=${dataset}_TUNING5_zero_${zero_loss_weight}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance LOO \
    --use_zero_loss \
    --align_loss_type l1 \
    --zero_loss_weight ${zero_loss_weight} \
    --saved_model_prefix ${zero_loss_weight}_ \
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
done

### TUNING 6 - uncertainty weight
# uncertainty-uniform
for align_loss_type in l2 kl; do
    for alignment_loss_weight in 0.1 1 10 100 1000; do
        expt=${dataset}_TUNING6_UncertaintyUniformWeight_${align_loss_type}_${alignment_loss_weight}_${expt_num}
        mkdir -p ${SAVE_DIR}/${expt} 

        CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
        --hint_type hints \
        --aug_type uncertainty-uniform \
        --align_loss_type ${align_loss_type} \
        --alignment_loss_weight ${alignment_loss_weight} \
        --saved_model_prefix ${alignment_loss_weight}_ \
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
    done
done

### TUNING 7 - replace func: align
for replace_func in 0s negative_ones random_sample gaussian; do
    expt=${dataset}_TUNING7_RepFunc_cossim_${replace_func}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --replace_func ${replace_func} \
    --hint_type hints \
    --model_importance LOO \
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
done

### TUNING 7 - replace func: suff-human+align
for replace_func in 0s negative_ones random_sample gaussian; do
    expt=${dataset}_TUNING7_AlignSuffTuning_cossim_${replace_func}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --replace_func ${replace_func} \
    --hint_type hints \
    --model_importance LOO \
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
done


### TUNING 8 - align func: l1 vs. l2 vs. cossim
for align_loss_type in l1 l2 cossim; do
    expt=${dataset}_TUNING8_AlignFunc_${align_loss_type}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance LOO \
    --use_direct_alignment \
    --align_loss_type ${align_loss_type} \
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
done


### TUNING 9 - explained class: predicted vs. ground-truth
# align - gt class
expt=${dataset}_TUNING9_align_gt_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--model_importance LOO \
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

# align - predicted class
expt=${dataset}_TUNING9_align_predicted_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--FI_predicted_class \
--hint_type hints \
--model_importance LOO \
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

# FI=0 - gt class
expt=${dataset}_TUNING9_zero_gt_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--hint_type hints \
--model_importance LOO \
--use_zero_loss \
--zero_loss_weight 0.1 \
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

# FI=0 - predicted class
expt=${dataset}_TUNING9_zero_predicted_${expt_num}
mkdir -p ${SAVE_DIR}/${expt} 

CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
--FI_predicted_class \
--hint_type hints \
--model_importance LOO \
--use_zero_loss \
--zero_loss_weight 0.1 \
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


### TUNING 10 - FI for align
# align
for FI_method in gradcam LOO KOI SHAP avg_effect; do
    expt=${dataset}_Tuning10_cossim_${FI_method}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance ${FI_method} \
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
done

# align
for FI_method in expected_gradient; do
    expt=${dataset}_Tuning10_cossim_${FI_method}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance ${FI_method} \
    --num_sample_eg 7 \
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
done

### TUNING 11 - FI for FI=0
# FI=0
for FI_method in gradcam LOO KOI SHAP avg_effect; do
    expt=${dataset}_Tuning11_zero_${FI_method}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance ${FI_method} \
    --use_zero_loss \
    --zero_loss_weight 0.1 \
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
done

for FI_method in expected_gradient; do
    expt=${dataset}_Tuning11_zero_${FI_method}_${expt_num}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --model_importance ${FI_method} \
    --num_sample_eg 7 \
    --use_zero_loss \
    --zero_loss_weight 0.1 \
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
done

### TUNING 10 - explanation methods
