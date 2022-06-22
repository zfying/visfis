# set path
PROJ_DIR=/ssd-playpen/home/zfying/visfis-tmp/
cd ${PROJ_DIR}
export PYTHONPATH=${PROJ_DIR}

# set threshold
dataset=$1
case "$dataset" in
    #case 1
    "xaicp") impt_threshold=0.85 FI_metrics=KOI ;;
    #case 2
    "hatcp") impt_threshold=0.55 FI_metrics=KOI ;;
    #case 3
    "gqacp") impt_threshold=0.3 split_postfix=-100k FI_metrics=LOO ;;
esac