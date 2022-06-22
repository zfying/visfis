SAVE_DIR=./saved_models_xaicp
DATA_DIR=../data/neg_data_xaicp
dataset=clevrxai
full_dataset=clevrxaicp

NUM=0
expt_num=exp0

## BUDGET
# align - pred class
for seed in 7 77 777 7777 77777; do
    for num_samples in 1 2 15 30; do
        for FI_method in expected_gradient SHAP avg_effect; do
            expt=BUDGET_cossim_${FI_method}_pred_sample${num_samples}_${seed}
            mkdir -p ${SAVE_DIR}/${expt} 

            CUDA_VISIBLE_DEVICES=${NUM} python -u main.py \
            --FI_predicted_class \
            --hint_type hints \
            --model_importance ${FI_method} \
            --num_sample_omission ${num_samples} \
            --num_sample_eg ${num_samples} \
            --use_direct_alignment \
            --align_loss_type cossim \
            --alignment_loss_weight 1 \
            --batch_size 64 \
            --seed ${seed} \
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
done