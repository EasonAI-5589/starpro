#!/bin/bash
# Resume TextVQA ablation: α=2.0 128T, then all α × {64T, 32T}
set -e
eval "$(conda shell.bash hook)"
conda activate fastv
export HF_HUB_OFFLINE=1

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

LOGFILE="/tmp/dr_textvqa_resume.log"

echo "================================================================" | tee "$LOGFILE"
echo "TextVQA Ablation RESUME — $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

run_one() {
    local ALPHA=$1 TOKENS=$2
    local TAG="D-${ALPHA}R"
    echo "" | tee -a "$LOGFILE"
    echo "▶ ${TAG}  (NEGATE=1, α=${ALPHA}, β=1.0) — ${TOKENS}T" | tee -a "$LOGFILE"

    OUTDIR="${WORKDIR}/playground/data/eval/textvqa/answers/${CKPT}/${METHOD}_${TAG}_${TOKENS}T"
    mkdir -p "$OUTDIR"

    # Parallel inference
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path "${CKPT_DIR}/${CKPT}" \
            --question-file "${DATA_DIR}/textvqa/${SPLIT}.jsonl" \
            --image-folder "${DATA_DIR}/textvqa/train_images" \
            --answers-file "${OUTDIR}/${CHUNKS}_${IDX}.jsonl" \
            --num-chunks $CHUNKS --chunk-idx $IDX \
            --temperature 0 --conv-mode $CONV_MODE \
            --pruning_method $METHOD --visual_token_num $TOKENS \
            2>&1 | grep -v "Token indices" &
    done
    wait

    # Merge
    MERGED="${OUTDIR}/merge.jsonl"
    > "$MERGED"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${OUTDIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGED"
    done

    # Evaluate
    python -m llava.eval.eval_textvqa \
        --annotation-file "${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json" \
        --result-file "$MERGED" 2>&1 | tee -a "$LOGFILE"
}

# α=2.0, 128T (was interrupted)
run_one 2.0 128

# All α × 64T
for ALPHA in 0.5 0.75 1.0 1.25 1.5 1.75 2.0; do
    run_one $ALPHA 64
done

# All α × 32T
for ALPHA in 0.5 0.75 1.0 1.25 1.5 1.75 2.0; do
    run_one $ALPHA 32
done

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "ALL COMPLETE — $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
