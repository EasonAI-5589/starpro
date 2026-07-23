#!/bin/bash
# ============================================================
# VScan: MMBench-EN, MMBench-CN, MMVet
# avg=128, avg=64, avg=32 (serial per benchmark)
# ============================================================
set -e

cd /mnt/eason/LLaVA-STAR-Pro
source /mnt/eason/miniconda3/bin/activate llava

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1

echo "============================================"
echo " VScan MMBench-EN + MMBench-CN + MMVet"
echo " Started: $(date)"
echo "============================================"

# ==================== MMBench EN ====================
echo ""
echo ">>>>>>>>>> MMBench EN <<<<<<<<<<<"

echo "[$(date)] Running: VScan MMBench-EN avg=128 (s1=224, s2=32)"
bash scripts/v1_5/7b/mmbench.sh vscan 224 32 2>&1 | tee results/mmbench_vscan_avg128.log
echo "[$(date)] Done: VScan MMBench-EN avg=128"

echo "[$(date)] Running: VScan MMBench-EN avg=64 (s1=96, s2=32)"
bash scripts/v1_5/7b/mmbench.sh vscan 96 32 2>&1 | tee results/mmbench_vscan_avg64.log
echo "[$(date)] Done: VScan MMBench-EN avg=64"

echo "[$(date)] Running: VScan MMBench-EN avg=32 (s1=32, s2=32)"
bash scripts/v1_5/7b/mmbench.sh vscan 32 32 2>&1 | tee results/mmbench_vscan_avg32.log
echo "[$(date)] Done: VScan MMBench-EN avg=32"

# ==================== MMBench CN ====================
echo ""
echo ">>>>>>>>>> MMBench CN <<<<<<<<<<<"

echo "[$(date)] Running: VScan MMBench-CN avg=128 (s1=224, s2=32)"
bash scripts/v1_5/7b/mmbench_cn.sh vscan 224 32 2>&1 | tee results/mmbench_cn_vscan_avg128.log
echo "[$(date)] Done: VScan MMBench-CN avg=128"

echo "[$(date)] Running: VScan MMBench-CN avg=64 (s1=96, s2=32)"
bash scripts/v1_5/7b/mmbench_cn.sh vscan 96 32 2>&1 | tee results/mmbench_cn_vscan_avg64.log
echo "[$(date)] Done: VScan MMBench-CN avg=64"

echo "[$(date)] Running: VScan MMBench-CN avg=32 (s1=32, s2=32)"
bash scripts/v1_5/7b/mmbench_cn.sh vscan 32 32 2>&1 | tee results/mmbench_cn_vscan_avg32.log
echo "[$(date)] Done: VScan MMBench-CN avg=32"

# ==================== MMVet ====================
echo ""
echo ">>>>>>>>>> MMVet <<<<<<<<<<<"

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
echo " ALL DONE - $(date)"
echo "============================================"

# Git push results
echo ""
echo "Pushing results to git..."
cd /mnt/eason/LLaVA-STAR-Pro
git add playground/data/eval/mmbench/answers_upload/ playground/data/eval/mmbench_cn/answers_upload/ playground/data/eval/mm-vet/answers_upload/ results/mmbench_vscan*.log results/mmbench_cn_vscan*.log results/mmvet_vscan*.log
git commit -m "VScan results: MMBench-EN, MMBench-CN, MMVet (avg128/64/32)" || echo "Nothing to commit"
git push origin || echo "Git push failed - will retry manually"

echo "============================================"
echo " Git push complete!"
echo "============================================"

# Restart occupy.py
nohup python occupy.py --reserve-mem-gb 20 > results/occupy_after_mmbench_mmvet.log 2>&1 &
echo "occupy.py restarted"
