#!/bin/bash
# D-αR ablation on TextVQA — all α values × 3 token budgets
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

ALPHAS=(0.5 0.75 1.0 1.25 1.5 1.75 2.0)
TOKEN_BUDGETS=(128 64 32)

LOGFILE="${WORKDIR}/results/dr_textvqa_ablation_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee "$LOGFILE"
echo "D-αR TextVQA Ablation — LLaVA-1.5-7B — star_pro" | tee -a "$LOGFILE"
echo "α values: ${ALPHAS[*]}" | tee -a "$LOGFILE"
echo "Token budgets: ${TOKEN_BUDGETS[*]}" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

for TOKEN in "${TOKEN_BUDGETS[@]}"; do
    echo "" | tee -a "$LOGFILE"
    echo "════════════════ ${TOKEN}T ════════════════" | tee -a "$LOGFILE"
    
    for ALPHA in "${ALPHAS[@]}"; do
        EXP_NAME="D-${ALPHA}R"
        
        echo "" | tee -a "$LOGFILE"
        echo "▶ ${EXP_NAME}  (NEGATE=1, α=${ALPHA}, β=1.0) — ${TOKEN}T" | tee -a "$LOGFILE"
        
        ANS_DIR="./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}"
        mkdir -p "${ANS_DIR}"
        
        # Inference
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
        
        # Evaluate
        EVAL_OUTPUT=$(python -m llava.eval.eval_textvqa \
            --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
            --result-file "$MERGE_FILE" 2>&1)
        echo "  $EVAL_OUTPUT" | tee -a "$LOGFILE"
    done
done

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "TEXTVQA ABLATION COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
