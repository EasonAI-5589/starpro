#!/bin/bash
# D-αR GQA — 精简版 α={0.5, 1.0, 2.0} × Token={128, 64, 32}
set -e
eval "$(conda shell.bash hook)"
conda activate fastv

WORKDIR="/mnt/eason/LLaVA-STAR-Pro"
CKPT_DIR="/mnt/eason_ckp/models"
DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="star_pro"
SPLIT="llava_gqa_testdev_balanced"
CONV_MODE="vicuna_v1"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

cd "$WORKDIR"

LOGFILE="${WORKDIR}/results/dr_gqa_v2_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee "$LOGFILE"
echo "D-αR GQA — α={0.5,1.0,2.0} × Token={128,64,32}" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

run_one() {
    local TOKEN=$1
    local ALPHA=$2
    local EXP_NAME="D-${ALPHA}R"

    echo "" | tee -a "$LOGFILE"
    echo "▶ ${EXP_NAME}  (NEGATE=1, α=${ALPHA}, β=1.0) — ${TOKEN}T" | tee -a "$LOGFILE"

    ANS_DIR="./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}"
    mkdir -p "${ANS_DIR}"

    # Inference (4-chunk 并行)
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=1 \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=1.0 \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/gqa/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/gqa/data/images \
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
    
    # eval using standard GQA script format
    python scripts/convert_gqa_for_eval.py --src "$MERGE_FILE" --dst ./playground/data/eval/gqa/data/testdev_balanced_predictions.json
    cd ./playground/data/eval/gqa/data
    python eval/eval.py --tier testdev_balanced | tee -a "$LOGFILE"
    rm testdev_balanced_predictions.json
    cd "$WORKDIR"
}

# ── 128T ──────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "════════════════ 128T ════════════════" | tee -a "$LOGFILE"
run_one 128 0.5
run_one 128 1.0
run_one 128 2.0

# ── 64T ───────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "════════════════ 64T ════════════════" | tee -a "$LOGFILE"
run_one 64 0.5
run_one 64 1.0
run_one 64 2.0

# ── 32T ───────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "════════════════ 32T ════════════════" | tee -a "$LOGFILE"
run_one 32 0.5
run_one 32 1.0
run_one 32 2.0

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "ALL DONE: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
