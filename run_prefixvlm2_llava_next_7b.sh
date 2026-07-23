#!/bin/bash
# SCOPE eval on LLaVA-NeXT (v1.6) 7B - 8 GPUs
# Use fastv conda env via PATH
# cd LLaVA-STAR-Pro2
# bash run_prefixvlm2_llava_next_7b.sh

set -e

# export PATH="/mnt/eason/miniconda3/envs/fastv/bin:$PATH"
# cd LLaVA-STAR-Pro2
# conda activate llava
# bash run_prefixvlm2_llava_next_7b.sh
export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"
export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  
export ENABLE_DEBUG=1
export ALPHA_1=0.5 #1.5对应prefix中的ALPHA=0.6
export LAMBDA_1=0.4 #越大表示越考虑第一阶段的coverage
export ALPHA_2=0.5 #对应prefix中的LAMBDA=0.5
export LAMBDA_2=0.4 #越大表示越考虑第二阶段的coverage
export COVERAGE=SCOPE
# export STAGE1_KEEP=320 #--160=5*32
# export STAGE1_KEEP=640 #--320=5*64
# export STAGE1_KEEP=1280 #--640=5*128

echo "Python: $(which python) ($(python --version))"

METHOD="prefixvlm_2"
BENCHMARKS=(mme pope sqa textvqa mmvet gqa mmbench mmbench_cn)  # textvqa gqa mmbench mmbench_cn
# BENCHMARKS=(gqa mme pope sqa textvqa mmbench mmbench_cn mmvet)

# for TOKEN in 128 64 32; do
for TOKEN in 64 32 ; do
    export STAGE1_KEEP=$((2*5*TOKEN))
    for BENCH in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=========================================="
        echo ">>> PrefixVLM_2 LLaVA-NeXT-7B | ${BENCH} | T=${TOKEN} (visual=${TOKEN}*5=$((TOKEN*5)))"
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
echo ">>> ALL PrefixVLM_2 LLaVA-NeXT-7B experiments DONE! $(date)"
echo "=========================================="
