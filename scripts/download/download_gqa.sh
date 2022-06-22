set -e
source scripts/common.sh

mkdir -p data/gqacp
cd data/gqacp
gdrive_path=$1

# get features
${gdrive_path} download 1DRW_O885n413eGze4iFRVnl98j_kklih # hdf5
${gdrive_path} download 1sXPat0zmZfH_AxoUYZTChFkUFpH_5aUG # imgid2img

mkdir processed

# get questions and annotations
mkdir questions
cd ./questions
${gdrive_path} download 19ck_ySsMj5wzIFnR0-IRoLi6leYtRzBS # train-100k q
${gdrive_path} download 1kY2OTK40qhrCA6g5d2W1V-rmb1z0oUn4 # train-100k a
${gdrive_path} download 1DCuaY9KvZiHUSJTbC0GDcHuf4JQiGGW_ # dev-100k q
${gdrive_path} download 16vfGMgqJrQG5ku_S-Xw-bBnmc8mGW8HK # dev-100k a
${gdrive_path} download 1JBGB1pjBWk5lgdM2EOvRADveLIzbr-m_ # id-100k a
${gdrive_path} download 1QCPS-gKLoA1uHpEJyYgKz5vhmXr6NNKU # id-100k q
${gdrive_path} download 1VAhmez21KkJtrybh8Qc8Pwlvi6vw2Rlg # ood-100k q
${gdrive_path} download 1poz9dYQWVJJsjcaQbKsNUFx1RBFTMBvz # ood-100k a

${gdrive_path} download 1WEDRMbHzcpgvu-dPfaXBr3zkVcklGzSZ # train q
${gdrive_path} download 1oM5Y-7Xr6ewTns356frsA9gzB1uRoqMd # train a
${gdrive_path} download 119lN1NvMVGI6FrbHayJY5etmCKVxMA71 # dev q
${gdrive_path} download 159ARzf0xWj0OUgS9oraYVtILTVFyCxhR # dev a
${gdrive_path} download 1Z8owsg4pTeY8ZSkkBzOCMgv9-QhJMCdO # id q
${gdrive_path} download 1z3BDJ_pwBF5MegpO0AAj2_1XvGeGQ0AH # id a
${gdrive_path} download 1LcsYK14fHcsTwvgGOhiB203pJboiGuaz # ood q
${gdrive_path} download 1oh6qxfXa9sWZ7OqmYpn2uuPJ-tPOITnt # ood a
cd ..

# get hints
mkdir hints
cd ./hints
${gdrive_path} download 1HVg601fykLS8sdk7RWT470suntQg9ksb
${gdrive_path} download 1NAMmV2iXDOQwdsFYLTvicD9C2FqUbzdh
cd ..

# glove
mkdir glove
cd ./glove
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip
rm glove.6B.zip
cd ..