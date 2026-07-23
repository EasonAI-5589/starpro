#!/bin/bash

#d2p
# export ENABLE_DEBUG=1
# export DIV_1=0
# export DIV_2=1.5
# bash scripts/v1_5/7b/mme.sh d2p 

# #prefixvlm
# export ENABLE_DEBUG=1
# export ALPHA=0.6 
# export LAMBDA=0.5
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128 

#prefixvlm_2
export ENABLE_DEBUG=1
export ALPHA_1=1 #1.5对应prefix中的ALPHA=0.6
export LAMBDA_1=0.1 #越大表示越考虑第一阶段的coverage
export ALPHA_2=1.0 #对应prefix中的LAMBDA=0.5
export LAMBDA_2=0.1 #越大表示越考虑第二阶段的coverage
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
export STAGE1_KEEP=256 #--128
# export STAGE1_KEEP=384 #--192  

# #svd
# export SVD_TAU=0.95
# export ALPHA=0.6 
# export LAMBDA=0.48 

# export NUM_CLUSTERS=64
# export ATTN_THRESHOLD_RATIO=0.5
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"

CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_${TOKEN}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vqav2/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

VQAV2_DIR="./playground/data/eval/vqav2"
output_file=./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2_DIR} \
    --src answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl \
    --dst answers_upload/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.json
