#!/bin/bash

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.6-vicuna-13b"
SPLIT="llava_textvqa_val_v051_ocr"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --pruning_method ${METHOD} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl
