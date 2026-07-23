#!/bin/bash

set -e

# export PATH="/mnt/eason/miniconda3/envs/fastv/bin:$PATH"
# cd LLaVA-STAR-Pro2
# conda activate llava
# bash run_prefixvlm2_llava_next_13b.sh
export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  
export ENABLE_DEBUG=1
export ALPHA_1=1 #1.5对应prefix中的ALPHA=0.6
export LAMBDA_1=1 #越大表示越考虑第一阶段的coverage
export ALPHA_2=1 #对应prefix中的LAMBDA=0.5
export LAMBDA_2=1 #越大表示越考虑第二阶段的coverage
export COVERAGE=SCOPE
# export STAGE1_KEEP=320 #--160=5*32
# export STAGE1_KEEP=640 #--320=5*64
# export STAGE1_KEEP=1280 #--640=5*128

echo "Python: $(which python) ($(python --version))"

METHOD="prefixvlm_2"
BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)  # textvqa gqa mmbench mmbench_cn
# BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)

# for TOKEN in 128 64 32; do
for TOKEN in 32 64 128 ; do
    export STAGE1_KEEP=$((2*5*TOKEN))
    for BENCH in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=========================================="
        echo ">>> PrefixVLM_2 LLaVA-NeXT-13B | ${BENCH} | T=${TOKEN} (visual=${TOKEN}*5=$((TOKEN*5)))"
        echo ">>> $(date)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_6/13b/${BENCH}.sh ${METHOD} ${TOKEN} 2>&1
        echo ">>> ${BENCH} T=${TOKEN} DONE ✅"
    done
    echo ""
    echo ">>> === All benchmarks for T=${TOKEN} DONE === $(date)"
done

echo ""
echo "=========================================="
echo ">>> ALL PrefixVLM_2 LLaVA-NeXT-13B experiments DONE! $(date)"
echo "=========================================="
