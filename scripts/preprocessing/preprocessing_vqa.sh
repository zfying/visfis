set -e
source scripts/common.sh

DATA_DIR=./data/hatcp
python preprocessing/create_dictionary.py --data_dir ${DATA_DIR}
python preprocessing/compute_softscore.py --data_dir ${DATA_DIR}