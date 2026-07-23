#!/bin/bash
# =============================================================================
# D-R Coupling Ablation on MME (LLaVA-1.5-7B, star_pro, 128 tokens)
# Uses absolute python path — no conda activate needed
# =============================================================================

set -eo pipefail

PYTHON="/mnt/eason/miniconda3/envs/fastv/bin/python"
export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export RESULT_DIR="/mnt/eason/LLaVA-STAR-Pro/results"

cd /mnt/eason/LLaVA-STAR-Pro

METHOD="star_pro"
TOKEN=128
CKPT="llava-v1.5-7b"
SPLIT="llava_mme"
PARAM="vtn_${TOKEN}"
CHUNKS=4
GPU_LIST="0 1 2 3"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${RESULT_DIR}/dr_coupling_ablation_${TIMESTAMP}.log"
mkdir -p ${RESULT_DIR}

log() { echo "$1" | tee -a ${MASTER_LOG}; }

log "================================================================"
log "D-R Coupling Ablation — MME — LLaVA-1.5-7B — 128T — star_pro"
log "Started: $(date)"
log "================================================================"

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

    local EXP_TAG="${METHOD}_${NAME}"
    local EXP_DIR="./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${EXP_TAG}/${PARAM}"
    mkdir -p "${EXP_DIR}"

    log ""
    log "──────────────────────────────────────────────────"
    log "▶ ${NAME}  (NEGATE=${NEGATE}, α=${ALPHA}, β=${BETA})"
    log "  Started: $(date)"
    log "──────────────────────────────────────────────────"

    # Step 1: Inference on 4 GPUs in parallel
    local PIDS=""
    for IDX in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${IDX} \
        NEGATE_RELEVANCE=${NEGATE} \
        RELEVANCE_WEIGHT=${ALPHA} \
        DIVERSITY_WEIGHT=${BETA} \
        ${PYTHON} -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
            --answers-file ${EXP_DIR}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${METHOD} \
            --visual_token_num ${TOKEN} \
            --temperature 0 \
            --conv-mode vicuna_v1 2>&1 | while IFS= read -r line; do echo "  [GPU${IDX}] ${line}"; done &
        PIDS="$PIDS $!"
    done

    # Wait for all GPUs
    local FAILED=0
    for PID in ${PIDS}; do
        wait ${PID} || FAILED=1
    done
    if [ ${FAILED} -eq 1 ]; then
        log "  ❌ INFERENCE FAILED for ${NAME}"
        return 1
    fi

    # Verify all chunks produced output
    local TOTAL_LINES=0
    for IDX in 0 1 2 3; do
        local CHUNK_LINES=$(wc -l < "${EXP_DIR}/${CHUNKS}_${IDX}.jsonl" 2>/dev/null || echo 0)
        TOTAL_LINES=$((TOTAL_LINES + CHUNK_LINES))
    done
    log "  Inference done: ${TOTAL_LINES} answers"

    # Step 2: Merge
    local MERGE_FILE="${EXP_DIR}/merge.jsonl"
    > "${MERGE_FILE}"
    for IDX in 0 1 2 3; do
        cat "${EXP_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGE_FILE}"
    done

    # Step 3: Convert answers to MME format
    cd ./playground/data/eval/MME
    ${PYTHON} convert_answer_to_mme.py \
        --data_path ${DATA_DIR}/MME \
        --experiment ${SPLIT}/${CKPT}/${EXP_TAG}/${PARAM}/merge 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
    cd /mnt/eason/LLaVA-STAR-Pro

    # Step 4: Evaluate
    log "  ── Results ──"
    cd ./playground/data/eval/MME/eval_tool
    ${PYTHON} calculation.py --results_dir answers/${SPLIT}/${CKPT}/${EXP_TAG}/${PARAM}/merge 2>&1 | tee -a ${MASTER_LOG}
    cd /mnt/eason/LLaVA-STAR-Pro

    log "  Finished: $(date)"
}

# Run all experiments sequentially
for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME NEGATE ALPHA BETA <<< "${EXP}"
    run_mme_experiment "${NAME}" "${NEGATE}" "${ALPHA}" "${BETA}" || log "  ⚠ Skipping due to error"
done

log ""
log "================================================================"
log "All experiments completed: $(date)"
log "Master log: ${MASTER_LOG}"
log "================================================================"
