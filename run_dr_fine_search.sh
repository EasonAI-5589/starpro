#!/bin/bash
# Fine-grained search for optimal α in D-αR (NEGATE=1)
# Fill in: α = 0.75, 1.25, 1.5, 1.75 (we already have 0.5, 1.0, 2.0)
set -e
eval "$(conda shell.bash hook)"
conda activate fastv

WORKDIR="/mnt/eason/LLaVA-STAR-Pro"
CKPT_DIR="/mnt/eason_ckp/models"
DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="star_pro"
SPLIT="llava_mme"
CONV_MODE="vicuna_v1"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

cd "$WORKDIR"

# α values to test (NEGATE=1 for all)
ALPHAS=(0.75 1.25 1.5 1.75)
TOKEN_BUDGETS=(128 64 32)

LOGFILE="${WORKDIR}/results/dr_fine_search_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee "$LOGFILE"
echo "Fine-grained α search: D-αR on MME Perception" | tee -a "$LOGFILE"
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
        echo "  Started: $(date)" | tee -a "$LOGFILE"
        
        ANS_DIR="./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/fine_${TOKEN}T_${EXP_NAME}"
        mkdir -p "${ANS_DIR}"
        
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
            NEGATE_RELEVANCE=1 \
            RELEVANCE_WEIGHT=${ALPHA} \
            DIVERSITY_WEIGHT=1.0 \
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
        
        MERGE_FILE="${ANS_DIR}/merge.jsonl"
        > "$MERGE_FILE"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE_FILE"
        done
        
        cd ./playground/data/eval/MME
        python convert_answer_to_mme.py \
            --data_path ${DATA_DIR}/MME \
            --experiment ${SPLIT}/${CKPT}/${METHOD}/fine_${TOKEN}T_${EXP_NAME}/merge
        
        cd eval_tool
        EVAL_OUTPUT=$(python calculation.py \
            --results_dir answers/${SPLIT}/${CKPT}/${METHOD}/fine_${TOKEN}T_${EXP_NAME}/merge 2>&1)
        echo "$EVAL_OUTPUT" | tee -a "$LOGFILE"
        echo "  Finished: $(date)" | tee -a "$LOGFILE"
        
        cd "$WORKDIR"
    done
done

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "FINE SEARCH COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
