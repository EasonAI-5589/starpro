#!/bin/bash
# D-λR POPE — λ={0.5, 1.0, 2.0} × Token={128, 64, 32}
# 统一用 λ 表示：Score = C + λ·D
set -e
eval "$(conda shell.bash hook)"
conda activate fastv

WORKDIR="/mnt/eason/LLaVA-STAR-Pro"
CKPT_DIR="/mnt/eason_ckp/models"
DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="star_pro"
SPLIT="llava_pope_test"
CONV_MODE="vicuna_v1"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

cd "$WORKDIR"

LOGFILE="${WORKDIR}/results/dr_pope_v3_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "================================================================" | tee "$LOGFILE"
echo "D-λR POPE — λ={0.5,1.0,2.0} × Token={128,64,32}" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

run_one() {
    local TOKEN=$1
    local LAMBDA=$2
    local EXP_NAME="lambda_${LAMBDA}"

    echo "" | tee -a "$LOGFILE"
    echo "▶ λ=${LAMBDA} — ${TOKEN}T" | tee -a "$LOGFILE"

    ANS_DIR="./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}"
    mkdir -p "${ANS_DIR}"

    # Inference (4-chunk 并行)
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=1 \
        LAMBDA=${LAMBDA} \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/pope/val2014 \
            --answers-file ${ANS_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${METHOD} \
            --visual_token_num ${TOKEN} \
            --temperature 0 \
            --conv-mode ${CONV_MODE} &
    done
    wait

    # Merge
    MERGE_FILE="${ANS_DIR}/merge.jsonl"
    > "$MERGE_FILE"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE_FILE"
    done
    echo "  merged: $(wc -l < ${MERGE_FILE}) lines" | tee -a "$LOGFILE"

    # Evaluate
    python llava/eval/eval_pope.py \
        --annotation-dir ${DATA_DIR}/pope/coco \
        --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
        --result-file "$MERGE_FILE" 2>&1 | tee -a "$LOGFILE"
}

# ── 串行跑：128T → 64T → 32T ──────────────────────

echo "" | tee -a "$LOGFILE"
echo "════════════════ 128T ════════════════" | tee -a "$LOGFILE"
run_one 128 0.5
run_one 128 1.0
run_one 128 2.0

echo "" | tee -a "$LOGFILE"
echo "════════════════ 64T ════════════════" | tee -a "$LOGFILE"
run_one 64 0.5
run_one 64 1.0
run_one 64 2.0

echo "" | tee -a "$LOGFILE"
echo "════════════════ 32T ════════════════" | tee -a "$LOGFILE"
run_one 32 0.5
run_one 32 1.0
run_one 32 2.0

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "ALL DONE: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
