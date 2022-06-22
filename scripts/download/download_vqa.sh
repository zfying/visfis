set -e
source scripts/common.sh

mkdir -p data/hatcp
cd data/hatcp
gdrive_path=$1

## setup vqa-hat
# visual features
${gdrive_path} download 1mEyG1tS4KXI5h3lwup-W1eiDhzXf_xxY # val36
${gdrive_path} download 1jEbsbihkT-Ie1uQYdlHbPcAqgyy0JLy5 # val36 imgid2img
${gdrive_path} download 1GYa73eJ3YxWohssseKtS8ZRuEzxUh9Cd # train36
${gdrive_path} download 1vGfLS2JIhq0xyfMiMQeYmOSZJA7SDL1D # train36 imgid2img

# glove
mkdir glove
cd ./glove
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip
rm glove.6B.zip
cd ..

# ans_cossim
${gdrive_path} download 1Rjqd6fA8jSOZT9VtFgCtURy_qb486a-8

# hints
mkdir hints
cd ./hints
${gdrive_path} download 1vgB4ixBAmOXwMxnpsVzXvIzXwqloRgTq # full hat
${gdrive_path} download 1dSmQToYyGnPZBbz87u5GJVyrBnG7ldMm # random
cd ..

# download hat-cp questions and annotations
mkdir questions
cd ./questions
${gdrive_path} download 1H8gvvcZgZhPwIDNR9pvrdQgok5mb7Pt4
${gdrive_path} download 1I8i7nA_0IcO6wmpmH0mYuvCACjtU3Bqb 
${gdrive_path} download 1NMm9Qc0IT5qLVBjaHcMc85iZ04rqbvWX
${gdrive_path} download 1QqqCQw4S_EYGjlCxV2fSwdNKUO6dFZ-a
${gdrive_path} download 1_14RPpTybpmctp66zO8f-ZhYqV5WS5OF
${gdrive_path} download 1axr-qgFDxcuSZfaq_fexvFgsf-DAgY8P 
${gdrive_path} download 1gcwuR6v8fwFkpFC96ahEYGHEwZOA0Xel
${gdrive_path} download 1vAVlRZAL7Q7RsLh44H6Jsik_0d_JTxRW 

# vqav2 Questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
rm v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
rm v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_Test_mscoco.zip
rm v2_Questions_Test_mscoco.zip

# vqav2 annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
cd ..