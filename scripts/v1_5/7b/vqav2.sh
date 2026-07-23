#!/bin/bash
export TOP_K_ANCHORS=144 
export ALPHA=0.5 
export LAMBDA=0.5 
export NUM_CLUSTERS=64
export ATTN_THRESHOLD_RATIO=0.5
export NEGATE_RELEVANCE=1 
export ENABLE_DEBUG=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/mnt/bn/bes-mllm-shared/checkpoint/LLaVA}"
DATA_DIR="${DATA_DIR:-/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval}"

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

METHOD=${1}
TOKEN=${2}
VSCAN_STAGE2=${3:-32}  # VScan Stage 2 tokens (default: 32)
PARAM="vtn_${TOKEN}"

# 🔬 VScan: stage1=${TOKEN}, stage2=${VSCAN_STAGE2}
# Average tokens = (stage1 + stage2) / 2
VSCAN_ARGS=""
if [ "$METHOD" == "vscan" ]; then
    AVG=$(( (TOKEN + VSCAN_STAGE2) / 2 ))
    PARAM="vtn_${AVG}"
    VSCAN_ARGS="--vscan_stage2_tokens ${VSCAN_STAGE2}"
    echo ">>> VScan: stage1=${TOKEN}, stage2=${VSCAN_STAGE2}, avg=${AVG}"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vqav2/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        ${VSCAN_ARGS} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

VQAV2_DIR="./playground/data/eval/vqav2"
output_file=./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2_DIR} \
    --src answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl \
    --dst answers_upload/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.json
