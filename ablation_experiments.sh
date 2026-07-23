#!/bin/bash
# ============================================================
# Ablation Experiments for ECCV 2026 §5 Figure
# Run on LLaVA-1.5-7B, tasks: pope, mme, textvqa
# Serial execution (one config at a time)
# ============================================================

# Don't use set -e: some eval steps may fail non-fatally
# set -e

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate llava

# Correct paths for this machine
export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro/results

GPUS="0,1,2,3"
TASKS=(pope mme gqa textvqa)
RESULT_LOG="${RESULT_DIR}/ablation_progress.log"
mkdir -p ${RESULT_DIR}

echo "===== Ablation Experiments Started: $(date) =====" | tee $RESULT_LOG

# ============================================================
# GROUP 1: Text Rater Strategy Ablation (Sub-figure b)
# TEXT_AGG_MODE: last_token / random / multi_token (default=ours)
# Budgets: 64, 32 (128 already exists)
# ============================================================

echo ">>> GROUP 1: Text Rater Strategy" | tee -a $RESULT_LOG

for MODE in last_token random; do
    for BUDGET in 64 32; do
        for TASK in "${TASKS[@]}"; do
            echo "[$(date)] Running: TEXT_AGG_MODE=${MODE} budget=${BUDGET} task=${TASK}" | tee -a $RESULT_LOG
            TEXT_AGG_MODE=${MODE} CUDA_VISIBLE_DEVICES=$GPUS bash scripts/v1_5/7b/${TASK}.sh star_pro ${BUDGET} 2>&1 | tail -10 | tee -a $RESULT_LOG
            
            # Copy result log with ablation label
            SRC="${RESULT_DIR}/${TASK}_scripts_v1_5_star_pro_vtn${BUDGET}.log"
            DST="${RESULT_DIR}/ablation_rater_${MODE}_${TASK}_vtn${BUDGET}.log"
            if [ -f "$SRC" ]; then
                cp "$SRC" "$DST"
                echo "  → Saved: $DST" | tee -a $RESULT_LOG
            fi
            echo "[$(date)] Done: ${MODE} ${BUDGET} ${TASK}" | tee -a $RESULT_LOG
            echo "" | tee -a $RESULT_LOG
        done
    done
done

echo ">>> GROUP 1 COMPLETE" | tee -a $RESULT_LOG

# ============================================================
# GROUP 2: Progressive Schedule Ablation (Sub-figure c)
# One-shot (layer 2 only) and 3-stage, at budgets 64 and 32
# ============================================================

echo ">>> GROUP 2: Schedule Ablation" | tee -a $RESULT_LOG

# One-shot: single prune at layer 2
for BUDGET in 64 32; do
    for TASK in "${TASKS[@]}"; do
        echo "[$(date)] Running: ONE-SHOT L2 budget=${BUDGET} task=${TASK}" | tee -a $RESULT_LOG
        CUSTOM_PRUNING_SCHEDULE="[(2, ${BUDGET})]" CUDA_VISIBLE_DEVICES=$GPUS bash scripts/v1_5/7b/${TASK}.sh star_pro ${BUDGET} 2>&1 | tail -10 | tee -a $RESULT_LOG
        
        SRC="${RESULT_DIR}/${TASK}_scripts_v1_5_star_pro_vtn${BUDGET}.log"
        DST="${RESULT_DIR}/ablation_schedule_oneshot_${TASK}_vtn${BUDGET}.log"
        if [ -f "$SRC" ]; then
            cp "$SRC" "$DST"
            echo "  → Saved: $DST" | tee -a $RESULT_LOG
        fi
        echo "[$(date)] Done: ONE-SHOT ${BUDGET} ${TASK}" | tee -a $RESULT_LOG
        echo "" | tee -a $RESULT_LOG
    done
done

# 3-stage: L2, L10, L18
for BUDGET in 64 32; do
    if [ "$BUDGET" == "64" ]; then
        SCHEDULE="[(2, 96), (10, 48), (18, 32)]"
    else
        SCHEDULE="[(2, 48), (10, 24), (18, 16)]"
    fi
    for TASK in "${TASKS[@]}"; do
        echo "[$(date)] Running: 3-STAGE budget=${BUDGET} task=${TASK} schedule=${SCHEDULE}" | tee -a $RESULT_LOG
        CUSTOM_PRUNING_SCHEDULE="${SCHEDULE}" CUDA_VISIBLE_DEVICES=$GPUS bash scripts/v1_5/7b/${TASK}.sh star_pro ${BUDGET} 2>&1 | tail -10 | tee -a $RESULT_LOG
        
        SRC="${RESULT_DIR}/${TASK}_scripts_v1_5_star_pro_vtn${BUDGET}.log"
        DST="${RESULT_DIR}/ablation_schedule_3stage_${TASK}_vtn${BUDGET}.log"
        if [ -f "$SRC" ]; then
            cp "$SRC" "$DST"
            echo "  → Saved: $DST" | tee -a $RESULT_LOG
        fi
        echo "[$(date)] Done: 3-STAGE ${BUDGET} ${TASK}" | tee -a $RESULT_LOG
        echo "" | tee -a $RESULT_LOG
    done
done

echo ">>> GROUP 2 COMPLETE" | tee -a $RESULT_LOG
echo "===== All Ablation Experiments Completed: $(date) =====" | tee -a $RESULT_LOG

# Summary
echo "" | tee -a $RESULT_LOG
echo "===== RESULTS SUMMARY =====" | tee -a $RESULT_LOG
for f in ${RESULT_DIR}/ablation_*.log; do
    if [ "$f" != "$RESULT_LOG" ]; then
        echo "--- $(basename $f) ---" | tee -a $RESULT_LOG
        tail -3 "$f" | tee -a $RESULT_LOG
        echo "" | tee -a $RESULT_LOG
    fi
done
