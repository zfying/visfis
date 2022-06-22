set -e
source scripts/common.sh

mkdir -p data/xaicp
cd data/xaicp
gdrive_path=$1

### download features
${gdrive_path} download 1H8wqNzFA8WkvntVWPnJcemTCG5ExVHeh # xai hdf5
${gdrive_path} download 1SrG-GrivTeEuGfA9zDKSpihF34T8jsMx # ids_map

mkdir processed

# download human explanations
mkdir hints
cd ./hints
${gdrive_path} download 13kcv9t3sDPb5bErvwLZZOaoUJdzrfZ3V # xai hints
${gdrive_path} download 1SCPPHtQNQoSW7DP-dDYMPwSJ486iojWM # xai hints - random
cd ..

# download questions
mkdir questions
cd ./questions
${gdrive_path} download 11IwUS-WUxOhWPU5guX8qe_uBEogEEHP7 # xaicp train
${gdrive_path} download 1nndFoieHOzVG2xySDFlu6wWof4iUvboR # xaicp dev
${gdrive_path} download 1Bt_rlKHRJ2VH1vPsH5LDvYieLiOhQbuu # xaicp test-id
${gdrive_path} download 1asm4zXVhFyz5-g6jupEz4pB96oO9ldDH # xaicp test-ood
cd ..

# download glove
mkdir glove
cd ./glove
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip
rm glove.6B.zip
cd ..