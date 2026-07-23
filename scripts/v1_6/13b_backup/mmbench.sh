#!/bin/bash

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.6-vicuna-13b"
SPLIT="mmbench_dev_20230712"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --pruning_method ${METHOD} \
    --visual_token_num ${TOKEN} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/${METHOD}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/${METHOD} \
    --experiment ${PARAM}
