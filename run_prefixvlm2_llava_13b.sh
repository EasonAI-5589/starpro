#!/bin/bash

set -e

# cd LLaVA-STAR-Pro2
# conda activate llava
# bash run_prefixvlm2_llava_13b.sh
export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  
export ENABLE_DEBUG=1
export ALPHA_1=0.4 #1.5对应prefix中的ALPHA=0.6
export LAMBDA_1=0.4 #越大表示越考虑第一阶段的coverage
export ALPHA_2=0.4 #对应prefix中的LAMBDA=0.5
export LAMBDA_2=0.4 #越大表示越考虑第二阶段的coverage
export COVERAGE=SCOPE
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128
# export STAGE1_KEEP=384 #--192   

echo "Python: $(which python) ($(python --version))"

METHOD="prefixvlm_2"
BENCHMARKS=(mmvet mme textvqa pope sqa gqa mmbench mmbench_cn)
# BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)

# for TOKEN in 128 64 32; do
for TOKEN in 32; do
    export STAGE1_KEEP=$((2*TOKEN))
    for BENCH in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=========================================="
        echo ">>> SCOPE LLaVA-13B | ${BENCH} | T=${TOKEN} (visual=${TOKEN}=$((TOKEN)))"
        echo ">>> $(date)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/13b/${BENCH}.sh ${METHOD} ${TOKEN} 2>&1
        echo ">>> ${BENCH} T=${TOKEN} DONE ✅"
    done
    echo ""
    echo ">>> === All benchmarks for T=${TOKEN} DONE === $(date)"
done

echo ""
echo "=========================================="
echo ">>> ALL SCOPE LLaVA-13B experiments DONE! $(date)"
echo "=========================================="