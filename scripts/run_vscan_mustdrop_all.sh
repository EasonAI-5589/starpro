#!/bin/bash
# ============================================================
# Run VScan + MustDrop on ALL benchmarks (128, 64 token budgets)
# Serial execution per Eason's rules
# ============================================================
set -e

cd /mnt/eason/LLaVA-STAR-Pro
source /mnt/eason/miniconda3/bin/activate llava

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro/results

echo "============================================"
echo " VScan + MustDrop Full Benchmark Suite"
echo " Started: $(date)"
echo "============================================"

# ─────────────────────────────────────────────
# Helper: run a benchmark if result doesn't exist
# ─────────────────────────────────────────────
run_if_missing() {
    local METHOD=$1
    local BENCH=$2
    local VTN=$3
    local LOG_NAME=$4
    local EXTRA_ARGS=$5  # e.g. "224 32" for vscan stage1 stage2

    local LOG_FILE="results/${LOG_NAME}.log"

    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        echo "[SKIP] ${LOG_NAME} already exists ($(wc -l < "$LOG_FILE") lines)"
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "[$(date)] Running: ${METHOD} ${BENCH} vtn=${VTN} ${EXTRA_ARGS}"
    echo "=========================================="

    if [ -n "$EXTRA_ARGS" ]; then
        bash scripts/v1_5/7b/${BENCH}.sh ${METHOD} ${EXTRA_ARGS} 2>&1 | tee "${LOG_FILE}"
    else
        bash scripts/v1_5/7b/${BENCH}.sh ${METHOD} ${VTN} 2>&1 | tee "${LOG_FILE}"
    fi

    echo "[$(date)] Done: ${LOG_NAME}"
}

# ─────────────────────────────────────────────
# Part 1: MustDrop (only VQAv2 missing)
# ─────────────────────────────────────────────
echo ""
echo ">>> PART 1: MustDrop VQAv2"
echo ""

run_if_missing mustdrop vqav2 128 "vqav2_scripts_v1_5_mustdrop_vtn128"
run_if_missing mustdrop vqav2 64  "vqav2_scripts_v1_5_mustdrop_vtn64"

# ─────────────────────────────────────────────
# Part 2: VScan avg=128 (stage1=224, stage2=32)
# Need: GQA, POPE, MME, TextVQA, SQA, VQAv2
# ─────────────────────────────────────────────
echo ""
echo ">>> PART 2: VScan avg=128 (stage1=224, stage2=32)"
echo ""

run_if_missing vscan gqa     128 "gqa_scripts_v1_5_vscan_vtn128"     "224 32"
run_if_missing vscan pope    128 "pope_scripts_v1_5_vscan_vtn128"    "224 32"
run_if_missing vscan mme     128 "mme_scripts_v1_5_vscan_vtn128"     "224 32"
run_if_missing vscan textvqa 128 "textvqa_scripts_v1_5_vscan_vtn128" "224 32"
run_if_missing vscan sqa     128 "sqa_scripts_v1_5_vscan_vtn128"     "224 32"
run_if_missing vscan vqav2   128 "vqav2_scripts_v1_5_vscan_vtn128"   "224 32"

# ─────────────────────────────────────────────
# Part 3: VScan avg=64 (stage1=96, stage2=32)
# GQA/POPE/MME/TextVQA already done (named vtn_96 in old scripts)
# Need: SQA, VQAv2
# ─────────────────────────────────────────────
echo ""
echo ">>> PART 3: VScan avg=64 (stage1=96, stage2=32)"
echo ""

# Re-run with proper avg naming for consistency (old ones used stage1 naming)
run_if_missing vscan gqa     64  "gqa_scripts_v1_5_vscan_vtn64"     "96 32"
run_if_missing vscan pope    64  "pope_scripts_v1_5_vscan_vtn64"    "96 32"
run_if_missing vscan mme     64  "mme_scripts_v1_5_vscan_vtn64"     "96 32"
run_if_missing vscan textvqa 64  "textvqa_scripts_v1_5_vscan_vtn64" "96 32"
run_if_missing vscan sqa     64  "sqa_scripts_v1_5_vscan_vtn64"     "96 32"
run_if_missing vscan vqav2   64  "vqav2_scripts_v1_5_vscan_vtn64"   "96 32"

echo ""
echo "============================================"
echo " ALL DONE!"
echo " Finished: $(date)"
echo "============================================"
echo ""
echo "Results summary:"
echo "  MustDrop: VQAv2 × 2 (new) + GQA/POPE/MME/TextVQA/SQA × 2 (existing)"
echo "  VScan:    All 6 benchmarks × 2 token budgets"
