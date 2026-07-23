#!/bin/bash
# bash llava/eval/benchmark_performance.sh prefixvlm_2 64
# bash llava/eval/benchmark_performance.sh prefixvlm_2 64
# bash llava/eval/benchmark_performance.sh fastv 64
# bash llava/eval/benchmark_performance.sh divprune 64

export PYTHONPATH=/mnt/eason/LLaVA-STAR-Pro2:$PYTHONPATH

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="llava_pope_test"

METHOD=${1:-prefixvlm_2}
TOKEN=${2:-64}
PARAM="vtn_${TOKEN}"

export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  
export ENABLE_DEBUG=0
export ALPHA_1=1 #1.5对应prefix中的ALPHA=0.6
export LAMBDA_1=1 #越大表示越考虑第一阶段的coverage
export ALPHA_2=1 #对应prefix中的LAMBDA=0.5
export LAMBDA_2=1 #越大表示越考虑第二阶段的coverage
export COVERAGE=SCOPE
export STAGE1_KEEP=640 #--64

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/benchmark_performance.py \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --warmup_samples 20 \
        --num_samples 200 \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --output-json ./results/benchmark_perf_${CKPT}_${METHOD}_${PARAM}_${IDX}.json &
done

wait

# POPE accuracy eval is skipped when --num_samples is set (incomplete answer file)
# output_file=./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl
# > "$output_file"
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done
#
# mkdir -p ./results
# python llava/eval/eval_pope.py \
#     --annotation-dir ${DATA_DIR}/pope/coco \
#     --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
#     --result-file $output_file | tee ./results/pope_v1.6_7b_${METHOD}_${PARAM}.log
