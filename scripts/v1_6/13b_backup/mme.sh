#!/bin/bash

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.6-vicuna-13b"
SPLIT="llava_mme"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --pruning_method ${METHOD} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py \
    --data_path ${DATA_DIR}/MME \
    --experiment ${SPLIT}/${CKPT}/${METHOD}/${PARAM}

cd eval_tool

python calculation.py --results_dir answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}
