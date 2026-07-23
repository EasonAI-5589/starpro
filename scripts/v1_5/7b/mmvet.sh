#!/bin/bash

#d2p
# export ENABLE_DEBUG=1
# export DIV_1=0
# export DIV_2=1.5
# bash scripts/v1_5/7b/mmvet.sh prefixvlm_2 128 

# #prefixvlm
# export ENABLE_DEBUG=1
# export ALPHA=0.6 
# export LAMBDA=0.5
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128 

#prefixvlm_2
# export http_proxy=http://192.168.32.28:18000  
# export https_proxy=http://192.168.32.28:18000  
# export ENABLE_DEBUG=1
# export ALPHA_1=1 #1.5对应prefix中的ALPHA=0.6
# export LAMBDA_1=1 #越大表示越考虑第一阶段的coverage
# export ALPHA_2=1 #对应prefix中的LAMBDA=0.5
# export LAMBDA_2=1 #越大表示越考虑第二阶段的coverage
# # # export STAGE1_KEEP=64 #--32
# # export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128  

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

CKPT="llava-v1.5-7b"
SPLIT="llava-mm-vet"

METHOD=${1}
TOKEN=${2}

# VScan-specific parameters
if [ "${METHOD}" = "vscan" ]; then
    STAGE2_TOKEN=${3:-32}  # Default: 32
    PRUNE_LAYER=${4:-16}   # Default: 16 for 7B
    PARAM="s1_${TOKEN}_s2_${STAGE2_TOKEN}"
else
    PARAM="vtn_${TOKEN}"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    if [ "${METHOD}" = "vscan" ]; then
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/mm-vet/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/mm-vet/images \
            --answers-file ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${METHOD} \
            --visual_token_num ${TOKEN} \
            --vscan_stage2_tokens ${STAGE2_TOKEN} \
            --vscan_prune_layer ${PRUNE_LAYER} \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    else
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/mm-vet/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/mm-vet/images \
            --answers-file ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${METHOD} \
            --visual_token_num ${TOKEN} \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    fi
done

wait

output_file=./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/${METHOD}

python scripts/convert_mmvet_for_eval.py \
    --src $output_file \
    --dst ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.json
