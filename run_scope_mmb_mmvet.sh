#!/bin/bash
# SCOPE eval: MMBench-EN, MMBench-CN, MMVet × 3 token budgets (128, 64, 32)
# 串行跑，不并行不同配置

export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"

cd /mnt/eason/LLaVA-STAR-Pro

METHOD="scope"

for TOKEN in 128 64 32; do
    echo ""
    echo "=========================================="
    echo ">>> Running SCOPE MMBench-EN T=${TOKEN}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/7b/mmbench.sh ${METHOD} ${TOKEN} 2>&1 | tail -5

    echo ""
    echo "=========================================="
    echo ">>> Running SCOPE MMBench-CN T=${TOKEN}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/7b/mmbench_cn.sh ${METHOD} ${TOKEN} 2>&1 | tail -5

    echo ""
    echo "=========================================="
    echo ">>> Running SCOPE MMVet T=${TOKEN}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/7b/mmvet.sh ${METHOD} ${TOKEN} 2>&1 | tail -5
done

echo ""
echo "=========================================="
echo ">>> ALL SCOPE MMB/MMB-CN/MMVet DONE"
echo "=========================================="
