set -e

# set arguments
dataset=$1
checkpoint=$2
GPU_NUM=$3
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

# set num_sample for calc metrics
case "$dataset" in
    #case 1
    "xaicp") num_sample=15 ;;
    #case 2
    "hatcp") num_sample=36 ;;
    #case 3
    "gqacp") num_sample=36 ;;
esac

# calc all metrics
for split_test in test-id${split_postfix} test-ood${split_postfix}; do
    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ${impt_threshold} \
    --calc_dp_level_metrics \
    --model_importance ${FI_metrics} \
    --num_sample_omission ${num_sample} \
    --FI_predicted_class true \
    --batch_size 64 \
    --seed 7 \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split_test ${split_test} \
    --checkpoint_path ${SAVE_DIR}/${checkpoint} \
    --load_checkpoint_path ${SAVE_DIR}/${checkpoint}/model-best.pth
done