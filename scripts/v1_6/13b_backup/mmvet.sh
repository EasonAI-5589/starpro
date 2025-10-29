#!/bin/bash

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.6-vicuna-13b"
SPLIT="llava-mm-vet"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

python -m llava.eval.model_vqa \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --pruning_method ${METHOD} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/${METHOD}

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --dst ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.json
