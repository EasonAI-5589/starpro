#!/bin/bash
# ============================================================
# VScan MME: avg=128, avg=64, avg=32 (serial)
# ============================================================
set -e

cd /mnt/eason/LLaVA-STAR-Pro
source /mnt/eason/miniconda3/bin/activate llava

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro/results

echo "============================================"
echo " VScan MME - Started: $(date)"
echo "============================================"

# --- avg=128: stage1=224, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan MME avg=128 (stage1=224, stage2=32)"
bash scripts/v1_5/7b/mme.sh vscan 224 32 2>&1 | tee results/mme_vscan_avg128.log
echo "[$(date)] Done: VScan MME avg=128"

# --- avg=64: stage1=96, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan MME avg=64 (stage1=96, stage2=32)"
bash scripts/v1_5/7b/mme.sh vscan 96 32 2>&1 | tee results/mme_vscan_avg64.log
echo "[$(date)] Done: VScan MME avg=64"

# --- avg=32: stage1=32, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan MME avg=32 (stage1=32, stage2=32)"
bash scripts/v1_5/7b/mme.sh vscan 32 32 2>&1 | tee results/mme_vscan_avg32.log
echo "[$(date)] Done: VScan MME avg=32"

echo ""
echo "============================================"
echo " VScan MME ALL DONE - $(date)"
echo "============================================"

# Restart occupy.py to hold GPUs
cd /mnt/eason/LLaVA-STAR-Pro
nohup python occupy.py --reserve-mem-gb 20 > results/occupy_after_mme.log 2>&1 &
echo "occupy.py restarted"
