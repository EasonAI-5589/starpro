#!/bin/bash
# export PATH="/mnt/eason/miniconda3/envs/fastv/bin:$PATH"
# cd LLaVA-STAR-Pro2
# conda activate llava
# bash run_prefixvlm2_llava_7b.sh

set -e

export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export http_proxy=http://192.168.32.28:18000.  
export https_proxy=http://192.168.32.28:18000  
export ENABLE_DEBUG=1
export ALPHA_1=2 #1.5对应prefix中的ALPHA=0.6
export LAMBDA_1=2 #越大表示越考虑第一阶段的coverage
export ALPHA_2=2 #对应prefix中的LAMBDA=0.5
export LAMBDA_2=2 #越大表示越考虑第二阶段的coverage
export COVERAGE=SCOPE
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128 

echo "Python: $(which python) ($(python --version))"

METHOD="fastv"
BENCHMARKS=(pope sqa textvqa mmbench mmvet)  # textvqa gqa mmbench mmbench_cn
# BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)

# for TOKEN in 128 64 32; do
for TOKEN in 256 192; do
    export STAGE1_KEEP=$((2*TOKEN))
    for BENCH in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=========================================="
        echo ">>> PrefixVLM_2 LLaVA-7B | ${BENCH} | T=${TOKEN} (visual=${TOKEN}=$((TOKEN)))"
        echo ">>> $(date)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/7b/${BENCH}.sh ${METHOD} ${TOKEN} 2>&1
        echo ">>> ${BENCH} T=${TOKEN} DONE ✅"
    done
    echo ""
    echo ">>> === All benchmarks for T=${TOKEN} DONE === $(date)"
done

echo ""
echo "=========================================="
echo ">>> ALL PrefixVLM_2 LLaVA-7B experiments DONE! $(date)"
echo "=========================================="
