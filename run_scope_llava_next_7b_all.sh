#!/bin/bash
# SCOPE eval on LLaVA-NeXT (v1.6) 7B - 8 GPUs
# Use fastv conda env via PATH

set -e

export PATH="/mnt/eason/miniconda3/envs/fastv/bin:$PATH"
export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"

cd /mnt/eason/LLaVA-STAR-Pro

echo "Python: $(which python) ($(python --version))"

METHOD="scope"
BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)

for TOKEN in 128 64 32; do
    for BENCH in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=========================================="
        echo ">>> SCOPE LLaVA-NeXT-7B | ${BENCH} | T=${TOKEN} (visual=${TOKEN}*5=$((TOKEN*5)))"
        echo ">>> $(date)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_6/7b/${BENCH}.sh ${METHOD} ${TOKEN} 2>&1
        echo ">>> ${BENCH} T=${TOKEN} DONE ✅"
    done
    echo ""
    echo ">>> === All benchmarks for T=${TOKEN} DONE === $(date)"
done

echo ""
echo "=========================================="
echo ">>> ALL SCOPE LLaVA-NeXT-7B experiments DONE! $(date)"
echo "=========================================="
