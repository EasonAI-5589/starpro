#!/bin/bash
# D-αR TextVQA — 精简版 α={0.5, 1.0, 2.0} × Token={128, 64, 32}
# 128T 的 0.5 和 1.0 已有结果，从 128T α=2.0 开始
set -e
eval "$(conda shell.bash hook)"
conda activate fastv

WORKDIR="/mnt/eason/LLaVA-STAR-Pro"
CKPT_DIR="/mnt/eason_ckp/models"
DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="star_pro"
SPLIT="llava_textvqa_val_v051_ocr"
CONV_MODE="vicuna_v1"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

cd "$WORKDIR"

LOGFILE="${WORKDIR}/results/dr_textvqa_v2_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee "$LOGFILE"
echo "D-αR TextVQA — α={0.5,1.0,2.0} × Token={128,64,32}" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

run_one() {
    local TOKEN=$1
    local ALPHA=$2
    local EXP_NAME="D-${ALPHA}R"

    echo "" | tee -a "$LOGFILE"
    echo "▶ ${EXP_NAME}  (NEGATE=1, α=${ALPHA}, β=1.0) — ${TOKEN}T" | tee -a "$LOGFILE"

    ANS_DIR="./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}"
    mkdir -p "${ANS_DIR}"

    # Inference (4-chunk 并行)
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=1 \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=1.0 \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/textvqa/train_images \
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
    echo "  merge" | tee -a "$LOGFILE"

    # Evaluate
    python -m llava.eval.eval_textvqa \
        --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
        --result-file "$MERGE_FILE" 2>&1 | tee -a "$LOGFILE"
}

# ── 128T ──────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "════════════════ 128T ════════════════" | tee -a "$LOGFILE"
# 0.5 和 1.0 已完成，只跑 2.0
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
