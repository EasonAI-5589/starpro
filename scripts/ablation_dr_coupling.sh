#!/bin/bash
# =============================================================================
# D-R Coupling Ablation on MME (LLaVA-1.5-7B, star_pro, 128 tokens)
# =============================================================================
# Formula: scores = α × R̃_normalized + β × D_normalized
# Where R̃ = -cos_sim if NEGATE=1, cos_sim if NEGATE=0
# Both R̃_normalized and D_normalized are min-max normalized to [0,1]
#
# Configs tested:
#   1. D+R      (NEG=0, α=1,   β=1)   — relevance + diversity
#   2. D-R      (NEG=1, α=1,   β=1)   — anti-relevance + diversity (current default)
#   3. D+2R     (NEG=0, α=2,   β=1)   — strong relevance + diversity
#   4. D-2R     (NEG=1, α=2,   β=1)   — strong anti-relevance + diversity
#   5. D+0.5R   (NEG=0, α=0.5, β=1)   — weak relevance + diversity
#   6. D-0.5R   (NEG=1, α=0.5, β=1)   — weak anti-relevance + diversity
#   7. D_only   (α=0,   β=1)           — pure diversity (no relevance)
#   8. R_only   (NEG=0, α=1,   β=0)   — pure relevance (no diversity)
#   9. negR_only(NEG=1, α=1,   β=0)   — pure anti-relevance (no diversity)
# =============================================================================

set -e

export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export RESULT_DIR="/mnt/eason/LLaVA-STAR-Pro/results"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd /mnt/eason/LLaVA-STAR-Pro

METHOD="star_pro"
TOKEN=128
CKPT="llava-v1.5-7b"
SPLIT="llava_mme"
PARAM="vtn_${TOKEN}"
CHUNKS=4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${RESULT_DIR}/dr_coupling_ablation_${TIMESTAMP}.log"
mkdir -p ${RESULT_DIR}

echo "================================================================" | tee ${MASTER_LOG}
echo "D-R Coupling Ablation — MME — LLaVA-1.5-7B — 128T — star_pro" | tee -a ${MASTER_LOG}
echo "Started: $(date)" | tee -a ${MASTER_LOG}
echo "================================================================" | tee -a ${MASTER_LOG}

# Define experiments: NAME|NEGATE|ALPHA|BETA
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

run_mme_experiment() {
    local NAME=$1
    local NEGATE=$2
    local ALPHA=$3
    local BETA=$4

    local EXP_DIR="./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}_${NAME}/${PARAM}"
    mkdir -p "${EXP_DIR}"

    echo "" | tee -a ${MASTER_LOG}
    echo "──────────────────────────────────────────────────" | tee -a ${MASTER_LOG}
    echo "▶ Running: ${NAME}  (NEGATE=${NEGATE}, α=${ALPHA}, β=${BETA})" | tee -a ${MASTER_LOG}
    echo "  Started: $(date)" | tee -a ${MASTER_LOG}
    echo "──────────────────────────────────────────────────" | tee -a ${MASTER_LOG}

    # Step 1: Inference with 4 GPUs
    for IDX in $(seq 0 3); do
        CUDA_VISIBLE_DEVICES=${IDX} \
        NEGATE_RELEVANCE=${NEGATE} \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=${BETA} \
        python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
            --answers-file ${EXP_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${METHOD} \
            --visual_token_num ${TOKEN} \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait

    # Step 2: Merge
    local MERGE_FILE="${EXP_DIR}/merge.jsonl"
    > "${MERGE_FILE}"
    for IDX in $(seq 0 3); do
        cat "${EXP_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGE_FILE}"
    done

    # Step 3: Convert answers
    cd ./playground/data/eval/MME
    python convert_answer_to_mme.py \
        --data_path ${DATA_DIR}/MME \
        --experiment ${SPLIT}/${CKPT}/${METHOD}_${NAME}/${PARAM}/merge
    cd /mnt/eason/LLaVA-STAR-Pro

    # Step 4: Evaluate
    echo "  Results for ${NAME}:" | tee -a ${MASTER_LOG}
    cd ./playground/data/eval/MME/eval_tool
    python calculation.py --results_dir answers/${SPLIT}/${CKPT}/${METHOD}_${NAME}/${PARAM}/merge 2>&1 | tee -a ${MASTER_LOG}
    cd /mnt/eason/LLaVA-STAR-Pro

    echo "  Finished: $(date)" | tee -a ${MASTER_LOG}
}

# Run all experiments sequentially
for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME NEGATE ALPHA BETA <<< "${EXP}"
    run_mme_experiment "${NAME}" "${NEGATE}" "${ALPHA}" "${BETA}"
done

echo "" | tee -a ${MASTER_LOG}
echo "================================================================" | tee -a ${MASTER_LOG}
echo "All experiments completed: $(date)" | tee -a ${MASTER_LOG}
echo "Results saved to: ${MASTER_LOG}" | tee -a ${MASTER_LOG}
echo "================================================================" | tee -a ${MASTER_LOG}
