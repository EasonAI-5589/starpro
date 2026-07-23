#!/bin/bash

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.6-vicuna-13b"
SPLIT="llava_test_CQM-I"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

python -m llava.eval.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/${SPLIT}.json \
    --image-folder ${DATA_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --pruning_method ${METHOD} \
    --visual_token_num ${TOKEN} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_science_qa \
    --base-dir ${DATA_DIR}/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}_result.json
