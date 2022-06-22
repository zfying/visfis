set -e

# set arguments
dataset=$1
GPU_NUM=$2
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

# set threshold
case "$dataset" in
    #case 1
    "xaicp") impt_threshold=0.85 FI_metrics=KOI ;;
    #case 2
    "hatcp") impt_threshold=0.55 FI_metrics=KOI ;;
    #case 3
    "gqacp") impt_threshold=0.3 split_postfix=-100k FI_metrics=LOO ;;
esac

# baseline
for seed in 7 77 777 7777 77777; do
    expt=${dataset}_updn_visfis_${seed}
    mkdir -p ${SAVE_DIR}/${expt} 

    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
    --hint_type hints \
    --impt_threshold ${impt_threshold} \
    --OBJ11 \
    --aug_type suff-uncertainty \
    --use_zero_loss \
    --use_direct_alignment \
    --model_importance gradcam \
    --FI_predicted_class false \
    --batch_size 64 \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train${split_postfix} \
    --split_test dev${split_postfix} \
    --max_epochs 50 \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    > ${SAVE_DIR}/${expt}/verbose_log.txt
    
    # calc all metrics
    source scripts/calc_metrics/val_updn.sh ${dataset} ${expt} ${GPU_NUM}
done 