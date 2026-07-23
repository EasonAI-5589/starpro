#!/bin/bash
set -e
cd /mnt/eason/LLaVA-STAR-Pro
source /mnt/eason/miniconda3/bin/activate llava

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1

echo "============================================"
echo " VScan MMVet - Started: $(date)"
echo "============================================"

echo "[$(date)] Running: VScan MMVet avg=128 (s1=224, s2=32)"
bash scripts/v1_5/7b/mmvet.sh vscan 224 32 2>&1 | tee results/mmvet_vscan_avg128.log
echo "[$(date)] Done: VScan MMVet avg=128"

echo "[$(date)] Running: VScan MMVet avg=64 (s1=96, s2=32)"
bash scripts/v1_5/7b/mmvet.sh vscan 96 32 2>&1 | tee results/mmvet_vscan_avg64.log
echo "[$(date)] Done: VScan MMVet avg=64"

echo "[$(date)] Running: VScan MMVet avg=32 (s1=32, s2=32)"
bash scripts/v1_5/7b/mmvet.sh vscan 32 32 2>&1 | tee results/mmvet_vscan_avg32.log
echo "[$(date)] Done: VScan MMVet avg=32"

echo ""
echo "============================================"
echo " VScan MMVet ALL DONE - $(date)"
echo "============================================"
