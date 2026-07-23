#!/bin/bash
# ====================================================================
# D-R Coupling Ablation Study on MME
# Explores how Diversity (D) and Relevance (R) should be coupled
# in Stage 1 THCP token selection
# ====================================================================

set -e

# ========== Conda Setup ==========
eval "$(conda shell.bash hook)"
conda activate fastv

# ========== Configuration ==========
WORKDIR="/mnt/eason/LLaVA-STAR-Pro"
CKPT_DIR="/mnt/eason_ckp/models"
DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
RESULT_DIR="/mnt/eason/LLaVA-STAR-Pro/results/dr_ablation"
CKPT="llava-v1.5-7b"
METHOD="star_pro"
TOKEN=128
SPLIT="llava_mme"
CONV_MODE="vicuna_v1"

# GPU setup
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

cd "$WORKDIR"
mkdir -p "$RESULT_DIR"

# ========== Experiment Definitions ==========
# Format: "NAME|NEGATE|RELEVANCE_WEIGHT|DIVERSITY_WEIGHT"
EXPERIMENTS=(
    "pure_D|1|0.0|1.0"        # Pure Diversity (baseline)
    "pure_R_pos|0|1.0|0.0"    # Pure Relevance, positive (baseline)
    "pure_R_neg|1|1.0|0.0"    # Pure Relevance, negated (baseline)
    "D_minus_R|1|1.0|1.0"     # D - R (legacy default)
    "D_plus_R|0|1.0|1.0"      # D + R (paper-aligned)
    "D_minus_0.5R|1|0.5|1.0"  # D - 0.5R
    "D_minus_2R|1|2.0|1.0"    # D - 2R
    "D_plus_0.5R|0|0.5|1.0"   # D + 0.5R
    "D_plus_2R|0|2.0|1.0"     # D + 2R
)

TOTAL=${#EXPERIMENTS[@]}
SUMMARY_FILE="${RESULT_DIR}/summary.txt"

echo "=====================================================================" | tee "$SUMMARY_FILE"
echo "D-R Coupling Ablation Study on MME" | tee -a "$SUMMARY_FILE"
echo "Model: ${CKPT} | Method: ${METHOD} | Tokens: ${TOKEN}" | tee -a "$SUMMARY_FILE"
echo "GPUs: ${gpu_list} (${CHUNKS} chunks)" | tee -a "$SUMMARY_FILE"
echo "Total experiments: ${TOTAL}" | tee -a "$SUMMARY_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_FILE"
echo "Python: $(which python)" | tee -a "$SUMMARY_FILE"
echo "=====================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_NAME NEGATE REL_W DIV_W <<< "${EXPERIMENTS[$i]}"
    EXP_NUM=$((i+1))
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "---------------------------------------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "[${EXP_NUM}/${TOTAL}] Experiment: ${EXP_NAME}" | tee -a "$SUMMARY_FILE"
    echo "  NEGATE_RELEVANCE=${NEGATE}, RELEVANCE_WEIGHT=${REL_W}, DIVERSITY_WEIGHT=${DIV_W}" | tee -a "$SUMMARY_FILE"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_FILE"
    echo "---------------------------------------------------------------------" | tee -a "$SUMMARY_FILE"
    
    # Unique output directory per experiment
    ANS_DIR="./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/dr_ablation_${EXP_NAME}"
    mkdir -p "${ANS_DIR}"
    
    # ===== Step 1: Inference =====
    echo "  [Step 1/3] Running inference..." | tee -a "$SUMMARY_FILE"
    
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
        NEGATE_RELEVANCE=${NEGATE} \
        RELEVANCE_WEIGHT=${REL_W} \
        DIVERSITY_WEIGHT=${DIV_W} \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
            --answers-file ${ANS_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${METHOD} \
            --visual_token_num ${TOKEN} \
            --temperature 0 \
            --conv-mode ${CONV_MODE} &
    done
    wait
    echo "  Inference done at $(date '+%H:%M:%S')." | tee -a "$SUMMARY_FILE"
    
    # ===== Step 2: Merge answers =====
    echo "  [Step 2/3] Merging answers..." | tee -a "$SUMMARY_FILE"
    MERGE_FILE="${ANS_DIR}/merge.jsonl"
    > "$MERGE_FILE"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE_FILE"
    done
    LINES=$(wc -l < "$MERGE_FILE")
    echo "  Merged ${LINES} answers." | tee -a "$SUMMARY_FILE"
    
    # ===== Step 3: Convert & Evaluate =====
    echo "  [Step 3/3] Evaluating..." | tee -a "$SUMMARY_FILE"
    
    cd ./playground/data/eval/MME
    python convert_answer_to_mme.py \
        --data_path ${DATA_DIR}/MME \
        --experiment ${SPLIT}/${CKPT}/${METHOD}/dr_ablation_${EXP_NAME}/merge
    
    cd eval_tool
    LOG_FILE="${RESULT_DIR}/mme_${EXP_NAME}.log"
    echo ">>> Experiment: ${EXP_NAME}" > "${LOG_FILE}"
    echo ">>> NEGATE_RELEVANCE=${NEGATE}, RELEVANCE_WEIGHT=${REL_W}, DIVERSITY_WEIGHT=${DIV_W}" >> "${LOG_FILE}"
    echo ">>> Model: ${CKPT}, Method: ${METHOD}, Tokens: ${TOKEN}" >> "${LOG_FILE}"
    echo ">>> Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    
    EVAL_OUTPUT=$(python calculation.py \
        --results_dir answers/${SPLIT}/${CKPT}/${METHOD}/dr_ablation_${EXP_NAME}/merge 2>&1)
    echo "$EVAL_OUTPUT" | tee -a "${LOG_FILE}"
    
    # Extract scores for summary
    PERCEPTION=$(echo "$EVAL_OUTPUT" | grep -i "perception" | grep -oP '[\d.]+' | tail -1)
    COGNITION=$(echo "$EVAL_OUTPUT" | grep -i "cognition" | grep -oP '[\d.]+' | tail -1)
    TOTAL_SCORE=$(echo "$EVAL_OUTPUT" | grep -i "total" | grep -oP '[\d.]+' | tail -1)
    
    echo "" | tee -a "$SUMMARY_FILE"
    printf "  ✅ %s: Perception=%-8s Cognition=%-8s Total=%-8s\n" \
        "${EXP_NAME}" "${PERCEPTION:-?}" "${COGNITION:-?}" "${TOTAL_SCORE:-?}" | tee -a "$SUMMARY_FILE"
    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_FILE"
    
    cd "$WORKDIR"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "=====================================================================" | tee -a "$SUMMARY_FILE"
echo "ALL ${TOTAL} EXPERIMENTS COMPLETE" | tee -a "$SUMMARY_FILE"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_FILE"
echo "Results saved to: ${RESULT_DIR}/" | tee -a "$SUMMARY_FILE"
echo "=====================================================================" | tee -a "$SUMMARY_FILE"

# ===== Final Summary Table =====
echo "" | tee -a "$SUMMARY_FILE"
echo "===== FINAL SUMMARY TABLE =====" | tee -a "$SUMMARY_FILE"
printf "%-20s %-6s %-6s %-6s %-10s %-10s %-10s\n" \
    "Experiment" "NEG" "α(R)" "β(D)" "Percept" "Cognit" "Total" | tee -a "$SUMMARY_FILE"
echo "----------------------------------------------------------------------" | tee -a "$SUMMARY_FILE"

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_NAME NEGATE REL_W DIV_W <<< "${EXPERIMENTS[$i]}"
    LOG_FILE="${RESULT_DIR}/mme_${EXP_NAME}.log"
    if [ -f "$LOG_FILE" ]; then
        P=$(grep -i "perception" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
        C=$(grep -i "cognition" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
        T=$(grep -i "total" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
        printf "%-20s %-6s %-6s %-6s %-10s %-10s %-10s\n" \
            "${EXP_NAME}" "${NEGATE}" "${REL_W}" "${DIV_W}" "${P:-?}" "${C:-?}" "${T:-?}" | tee -a "$SUMMARY_FILE"
    fi
done
