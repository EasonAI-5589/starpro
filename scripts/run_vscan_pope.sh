#!/bin/bash
# ============================================================
# VScan POPE: avg=128, avg=64, avg=32 (serial, 8 GPUs)
# ============================================================
set -e

cd /mnt/eason/LLaVA-STAR-Pro
source /mnt/eason/miniconda3/bin/activate llava

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro/results

echo "============================================"
echo " VScan POPE (8 GPUs) - Started: $(date)"
echo "============================================"

# --- avg=128: stage1=224, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan POPE avg=128 (stage1=224, stage2=32)"
bash scripts/v1_5/7b/pope.sh vscan 224 32 2>&1 | tee results/pope_vscan_avg128.log
echo "[$(date)] Done: VScan POPE avg=128"

# --- avg=64: stage1=96, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan POPE avg=64 (stage1=96, stage2=32)"
bash scripts/v1_5/7b/pope.sh vscan 96 32 2>&1 | tee results/pope_vscan_avg64.log
echo "[$(date)] Done: VScan POPE avg=64"

# --- avg=32: stage1=32, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan POPE avg=32 (stage1=32, stage2=32)"
bash scripts/v1_5/7b/pope.sh vscan 32 32 2>&1 | tee results/pope_vscan_avg32.log
echo "[$(date)] Done: VScan POPE avg=32"

echo ""
echo "============================================"
echo " VScan POPE ALL DONE - $(date)"
echo "============================================"
