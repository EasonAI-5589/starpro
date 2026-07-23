#!/bin/bash
# Sign Ablation: NEGATE=0 (正相似性 cos) vs NEGATE=1 (互补性 1-cos)
# 补跑 GQA / POPE / TextVQA 的 NEGATE=0 组，对应 α=2.0，T={128,64,32}
set -e
eval "$(conda shell.bash hook)"
conda activate fastv

WORKDIR="/mnt/eason/LLaVA-STAR-Pro"
CKPT_DIR="/mnt/eason_ckp/models"
DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="star_pro"
CONV_MODE="vicuna_v1"
ALPHA=2.0   # 与现有 NEGATE=1 数据对齐

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

cd "$WORKDIR"
LOGFILE="${WORKDIR}/results/sign_ablation_pos_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "================================================================" | tee "$LOGFILE"
echo "Sign Ablation: NEGATE=0 (similarity) | α=${ALPHA} | T=128/64/32" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# GQA
# ──────────────────────────────────────────────
run_gqa() {
    local TOKEN=$1
    SPLIT="llava_gqa_testdev_balanced"
    ANS_DIR="./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${METHOD}/sign_pos_${TOKEN}T"
    mkdir -p "${ANS_DIR}"

    echo "" | tee -a "$LOGFILE"
    echo "▶ [GQA] D+${ALPHA}R  (NEGATE=0) — ${TOKEN}T" | tee -a "$LOGFILE"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=0 \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=1.0 \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/gqa/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/gqa/data/images \
            --answers-file ${ANS_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} --chunk-idx ${IDX} \
            --pruning_method ${METHOD} --visual_token_num ${TOKEN} \
            --temperature 0 --conv-mode ${CONV_MODE} &
    done
    wait

    MERGE="${ANS_DIR}/merge.jsonl"
    > "$MERGE"
    for IDX in $(seq 0 $((CHUNKS-1))); do cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE"; done
    echo "  merged: $(wc -l < ${MERGE}) lines" | tee -a "$LOGFILE"

    python scripts/convert_gqa_for_eval.py --src "$MERGE" \
        --dst ./playground/data/eval/gqa/data/testdev_balanced_predictions.json
    cd ./playground/data/eval/gqa/data
    python eval/eval.py --path questions --tier testdev_balanced 2>&1 | tee -a "$LOGFILE"
    rm -f testdev_balanced_predictions.json
    cd "$WORKDIR"
}

# ──────────────────────────────────────────────
# POPE
# ──────────────────────────────────────────────
run_pope() {
    local TOKEN=$1
    ANS_DIR="./playground/data/eval/pope/answers/${CKPT}/${METHOD}/sign_pos_${TOKEN}T"
    mkdir -p "${ANS_DIR}"

    echo "" | tee -a "$LOGFILE"
    echo "▶ [POPE] D+${ALPHA}R  (NEGATE=0) — ${TOKEN}T" | tee -a "$LOGFILE"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=0 \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=1.0 \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --image-folder ${DATA_DIR}/coco/images/val2014 \
            --answers-file ${ANS_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} --chunk-idx ${IDX} \
            --pruning_method ${METHOD} --visual_token_num ${TOKEN} \
            --temperature 0 --conv-mode ${CONV_MODE} &
    done
    wait

    MERGE="${ANS_DIR}/merge.jsonl"
    > "$MERGE"
    for IDX in $(seq 0 $((CHUNKS-1))); do cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE"; done

    python llava/eval/eval_pope.py \
        --annotation-dir ./playground/data/eval/pope \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --result-file "$MERGE" 2>&1 | tee -a "$LOGFILE"
}

# ──────────────────────────────────────────────
# TextVQA
# ──────────────────────────────────────────────
run_textvqa() {
    local TOKEN=$1
    ANS_DIR="./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/sign_pos_${TOKEN}T"
    mkdir -p "${ANS_DIR}"

    echo "" | tee -a "$LOGFILE"
    echo "▶ [TextVQA] D+${ALPHA}R  (NEGATE=0) — ${TOKEN}T" | tee -a "$LOGFILE"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=0 \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=1.0 \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder ${DATA_DIR}/textvqa/train_images \
            --answers-file ${ANS_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} --chunk-idx ${IDX} \
            --pruning_method ${METHOD} --visual_token_num ${TOKEN} \
            --temperature 0 --conv-mode ${CONV_MODE} &
    done
    wait

    MERGE="${ANS_DIR}/merge.jsonl"
    > "$MERGE"
    for IDX in $(seq 0 $((CHUNKS-1))); do cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE"; done

    python -m llava.eval.eval_textvqa \
        --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
        --result-file "$MERGE" 2>&1 | tee -a "$LOGFILE"
}

# ══════════════════════════════════════════════
# 串行跑：GQA → POPE → TextVQA，128T → 64T → 32T
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "════════ GQA ════════" | tee -a "$LOGFILE"
run_gqa 128
run_gqa 64
run_gqa 32

echo "" | tee -a "$LOGFILE"
echo "════════ POPE ════════" | tee -a "$LOGFILE"
run_pope 128
run_pope 64
run_pope 32

echo "" | tee -a "$LOGFILE"
echo "════════ TextVQA ════════" | tee -a "$LOGFILE"
run_textvqa 128
run_textvqa 64
run_textvqa 32

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "ALL DONE: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
