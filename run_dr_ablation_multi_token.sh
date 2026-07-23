#!/bin/bash
# ====================================================================
# D-R Coupling Ablation — MME — 64T & 32T
# Same experiments as 128T, now for 64 and 32 token budgets
# ====================================================================
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

# Experiments: "NAME|NEGATE|RELEVANCE_WEIGHT|DIVERSITY_WEIGHT"
EXPERIMENTS=(
    "D+R|0|1.0|1.0"
    "D-R|1|1.0|1.0"
    "D+2R|0|2.0|1.0"
    "D-2R|1|2.0|1.0"
    "D+0.5R|0|0.5|1.0"
    "D-0.5R|1|0.5|1.0"
    "D_only|0|0.0|1.0"
    "R_only|0|1.0|0.0"
    "negR_only|1|1.0|0.0"
)

TOKEN_BUDGETS=(64 32)

for TOKEN in "${TOKEN_BUDGETS[@]}"; do
    LOGFILE="${WORKDIR}/results/dr_coupling_ablation_${TOKEN}T_$(date +%Y%m%d_%H%M%S).log"
    
    echo "================================================================" | tee "$LOGFILE"
    echo "D-R Coupling Ablation — MME — LLaVA-1.5-7B — ${TOKEN}T — star_pro" | tee -a "$LOGFILE"
    echo "Started: $(date)" | tee -a "$LOGFILE"
    echo "================================================================" | tee -a "$LOGFILE"
    
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r EXP_NAME NEGATE REL_W DIV_W <<< "$exp"
        
        echo "" | tee -a "$LOGFILE"
        echo "──────────────────────────────────────────────────" | tee -a "$LOGFILE"
        echo "▶ ${EXP_NAME}  (NEGATE=${NEGATE}, α=${REL_W}, β=${DIV_W})" | tee -a "$LOGFILE"
        echo "  Started: $(date)" | tee -a "$LOGFILE"
        echo "──────────────────────────────────────────────────" | tee -a "$LOGFILE"
        
        ANS_DIR="./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}"
        mkdir -p "${ANS_DIR}"
        
        # Inference (4 GPUs parallel)
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
        
        # Merge
        MERGE_FILE="${ANS_DIR}/merge.jsonl"
        > "$MERGE_FILE"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat "${ANS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$MERGE_FILE"
        done
        LINES=$(wc -l < "$MERGE_FILE")
        echo "  Inference done: ${LINES} answers" | tee -a "$LOGFILE"
        
        # Evaluate
        cd ./playground/data/eval/MME
        python convert_answer_to_mme.py \
            --data_path ${DATA_DIR}/MME \
            --experiment ${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}/merge
        
        cd eval_tool
        echo "  ── Results ──" | tee -a "$LOGFILE"
        python calculation.py \
            --results_dir answers/${SPLIT}/${CKPT}/${METHOD}/dr_${TOKEN}T_${EXP_NAME}/merge 2>&1 | tee -a "$LOGFILE"
        
        echo "" | tee -a "$LOGFILE"
        echo "  Finished: $(date)" | tee -a "$LOGFILE"
        
        cd "$WORKDIR"
    done
    
    echo "" | tee -a "$LOGFILE"
    echo "================================================================" | tee -a "$LOGFILE"
    echo "${TOKEN}T experiments completed: $(date)" | tee -a "$LOGFILE"
    echo "================================================================" | tee -a "$LOGFILE"
done

echo ""
echo "ALL TOKEN BUDGETS DONE: $(date)"
