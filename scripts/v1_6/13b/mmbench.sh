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
# export ENABLE_DEBUG=1
# export ALPHA_1=1 #1.5对应prefix中的ALPHA=0.6
# export LAMBDA_1=0.1 #越大表示越考虑第一阶段的coverage
# export ALPHA_2=1.0 #对应prefix中的LAMBDA=0.5
# export LAMBDA_2=0.1 #越大表示越考虑第二阶段的coverage
# export STAGE1_KEEP=320 #--160
# export STAGE1_KEEP=640 #--320
# export STAGE1_KEEP=1280 #--640  

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

CKPT="llava-v1.6-vicuna-13b"
SPLIT="mmbench_dev_20230712"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/${METHOD}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/${METHOD} \
    --experiment ${PARAM}/merge