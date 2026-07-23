#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/mnt/bn/bes-mllm-shared/checkpoint/LLaVA}"
DATA_DIR="${DATA_DIR:-/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval}"

CKPT="llava-v1.5-7b"
SPLIT="llava_test"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_${TOKEN}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vizwiz/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/vizwiz/test \
        --answers-file ./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/${SPLIT}.jsonl \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.json
