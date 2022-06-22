set -e

# set arguments
dataset=$1
GPU_NUM=$2
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

# baseline
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_lxmert_baseline_${seed}
    mkdir -p ${SAVE_DIR}/${expt}
    
    # train model
    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ${impt_threshold} \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train${split_postfix} \
    --split_test dev${split_postfix} \
    --model_type lxmert \
    --spatial_type simple \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --max_epochs 35 \
    --grad_clip 5 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
    
    # calc all metrics
    source scripts/calc_metrics/val_lxmert.sh ${dataset} ${expt} ${GPU_NUM}
done 