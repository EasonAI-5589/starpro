#!/bin/bash
# ============================================================
# VScan VQAv2: avg=128, avg=64, avg=32 (serial)
# ============================================================
set -e

cd /mnt/eason/LLaVA-STAR-Pro
source /mnt/eason/miniconda3/bin/activate llava

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1

echo "============================================"
echo " VScan VQAv2 - Started: $(date)"
echo "============================================"

# --- avg=128: stage1=224, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan VQAv2 avg=128 (stage1=224, stage2=32)"
bash scripts/v1_5/7b/vqav2.sh vscan 224 32 2>&1 | tee results/vqav2_vscan_avg128.log
echo "[$(date)] Done: VScan VQAv2 avg=128"

# --- avg=64: stage1=96, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan VQAv2 avg=64 (stage1=96, stage2=32)"
bash scripts/v1_5/7b/vqav2.sh vscan 96 32 2>&1 | tee results/vqav2_vscan_avg64.log
echo "[$(date)] Done: VScan VQAv2 avg=64"

# --- avg=32: stage1=32, stage2=32 ---
echo ""
echo "[$(date)] Running: VScan VQAv2 avg=32 (stage1=32, stage2=32)"
bash scripts/v1_5/7b/vqav2.sh vscan 32 32 2>&1 | tee results/vqav2_vscan_avg32.log
echo "[$(date)] Done: VScan VQAv2 avg=32"

echo ""
echo "============================================"
echo " VScan VQAv2 ALL DONE - $(date)"
echo "============================================"

# Git push results
echo ""
echo "Pushing results to git..."
cd /mnt/eason/LLaVA-STAR-Pro
git add -f playground/data/eval/vqav2/answers_upload/
git commit -m "VScan VQAv2 results (avg128/64/32) submission files" || echo "Nothing to commit"
git push origin main || echo "Git push failed"

echo "============================================"
echo " Git push complete!"
echo "============================================"

# Restart occupy.py
nohup python occupy.py --reserve-mem-gb 20 > results/occupy_after_vqav2.log 2>&1 &
echo "occupy.py restarted"
