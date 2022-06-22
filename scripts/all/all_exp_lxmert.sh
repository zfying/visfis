SAVE_DIR=./saved_models_xaicp
DATA_DIR=./data/xaicp
dataset=xaicp

NUM=0

# MAIN1 baseline
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN1_baseline_${seed}
    mkdir -p ${SAVE_DIR}/${expt}
    
    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold 0.3 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done

## MAIN2 - baseline + CF
# suff-random
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN2_BaselineCF_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --aug_type suff-random \
    --aug_loss_weight 1 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done 

## MAIN3 - Simpson 2019 (Invariance-FI=0 w VGrad, L2 align)
# FI=0
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN3_Simpson2019_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --model_importance gradcam \
    --FI_predicted_class false \
    --use_zero_loss \
    --align_loss_type l2 \
    --zero_loss_weight 1e-3 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done

## MAIN4 - Chang 2021 (Sufficiency + FI=0 w VGrad w L2, shuffle replace f.)
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN4_Chang2021_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --aug_type suff-human \
    --replace_func shuffle \
    --aug_loss_weight 1 \
    --model_importance gradcam \
    --FI_predicted_class false \
    --use_zero_loss \
    --align_loss_type l2 \
    --zero_loss_weight 1e-3 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done

## MAIN5 - Singla 2022 (Sufficiency + FI=0 w VGrad w L2, Gaussian replace f.)
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN5_Singla2022_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --aug_type suff-human \
    --replace_func gaussian \
    --aug_loss_weight 1 \
    --model_importance gradcam \
    --FI_predicted_class false \
    --use_zero_loss \
    --align_loss_type l2 \
    --zero_loss_weight 1e-3 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done


## MAIN6 - VisFIS
# Align-Cos + Invariance-FI + Suff-Human + Uncertainty
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN6_gradcam_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --OBJ11 \
    --aug_type suff-uncertainty \
    --use_zero_loss \
    --use_direct_alignment \
    --model_importance gradcam \
    --FI_predicted_class false \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done

## MAIN7 - VisFIS + random supervision
# Align-Cos + Invariance-FI + Suff-Human + Uncertainty
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN7_gradcam_random_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints_random \
    --impt_threshold ? \
    --OBJ11 \
    --aug_type suff-uncertainty \
    --use_zero_loss \
    --use_direct_alignment \
    --model_importance gradcam \
    --FI_predicted_class false \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done

## MAIN8 - HINT weight = 1e-6
# baseline
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN8_HINT_${seed}
    mkdir -p ${SAVE_DIR}/${expt}

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --model_importance gradcam \
    --FI_predicted_class false \
    --use_hint_loss \
    --hint_loss_weight 1e-6 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done 

## MAIN9 - SCR
# baseline
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_MAIN9_SCR_${seed}
    mkdir -p ${SAVE_DIR}/${expt}

    CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ? \
    --model_importance gradcam \
    --FI_predicted_class false \
    --use_scr_loss \
    --scr_hint_loss_weight 1e-6 \
    --scr_compare_loss_weight 1e-4 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
done 