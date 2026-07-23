#!/bin/bash
# SCOPE eval on LLaVA-NeXT (v1.6) 7B - 8 GPUs
# Use fastv conda env via PATH

set -e

# export PATH="/mnt/eason/miniconda3/envs/fastv/bin:$PATH"
# cd LLaVA-STAR-Pro2
# conda activate llava
# bash run_fastv_llava_13b.sh
export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128
# export STAGE1_KEEP=384 #--192   

echo "Python: $(which python) ($(python --version))"

METHOD="fastv"
BENCHMARKS=(pope)
# BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)

# for TOKEN in 128 64 32; do
for TOKEN in 32 64 128 ; do
    for BENCH in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=========================================="
        echo ">>> SCOPE LLaVA-13B | ${BENCH} | T=${TOKEN} (visual=${TOKEN}=$((TOKEN)))"
        echo ">>> $(date)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/13b/${BENCH}.sh ${METHOD} ${TOKEN} 2>&1
        echo ">>> ${BENCH} T=${TOKEN} DONE ✅"
    done
    echo ""
    echo ">>> === All benchmarks for T=${TOKEN} DONE === $(date)"
done

echo ""
echo "=========================================="
echo ">>> ALL FASTV LLaVA-13B experiments DONE! $(date)"
echo "=========================================="
