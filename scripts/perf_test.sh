#!/bin/bash
# Single GPU Performance Test
# Usage: bash scripts/perf_test.sh <method> <token_num> [gpu_id]
# Example: bash scripts/perf_test.sh star_v3 128 0

METHOD=${1:-"star_v3"}
TOKEN=${2:-128}
GPU_ID=${3:-0}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.6-vicuna-7b"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ${DATA_DIR}/pope/val2014 \
    --answers-file ./results/${METHOD}_vtn${TOKEN}.jsonl \
    --pruning_method ${METHOD} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1
