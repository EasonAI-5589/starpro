#!/bin/bash
# bash run_scope_eval.sh

# ==================== SCOPE Evaluation Script ====================
# SCOPE: Saliency-Coverage Oriented Token Pruning
# Extracted from eval.sh - runs only the SCOPE evaluation section

set -e

# ==================== Conda Environment ====================
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
eval "$(conda shell.bash hook)"
conda activate llava

# ==================== Configuration ====================
MODEL_VERSION="v1_6"
MODEL_SCALE="13b"
METHOD="scope"

# GPU devices
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Paths
export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro2/results

# ==================== Download Model if Missing ====================
CKPT="llava-v1.6-vicuna-13b"
MODEL_PATH="${CKPT_DIR}/${CKPT}"

if [ ! -f "${MODEL_PATH}/model-00001-of-00006.safetensors" ]; then
    echo ">>> Model ${CKPT} weights not found at ${MODEL_PATH}, downloading from HuggingFace..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='liuhaotian/llava-v1.6-vicuna-13b',
    local_dir='${MODEL_PATH}',
    local_dir_use_symlinks=False
)
print('>>> Download complete!')
"
fi

# ==================== Working Directory ====================
cd /mnt/eason/LLaVA-STAR-Pro2
mkdir -p ${RESULT_DIR}

echo "============================================"
echo ">>> SCOPE Evaluation"
echo ">>> Model: LLaVA ${MODEL_VERSION} ${MODEL_SCALE}"
echo ">>> Method: ${METHOD}"
echo ">>> GPUs: ${GPUS}"
echo ">>> CKPT_DIR: ${CKPT_DIR}"
echo ">>> DATA_DIR: ${DATA_DIR}"
echo ">>> RESULT_DIR: ${RESULT_DIR}"
echo "============================================"

# ==================== SCOPE with 32 tokens ====================
echo ""
echo ">>> [1/8] MME - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mme.sh scope 32

echo ""
echo ">>> [2/8] POPE - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/pope.sh scope 32

echo ""
echo ">>> [3/8] SQA - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/sqa.sh scope 32

echo ""
echo ">>> [4/8] GQA - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/gqa.sh scope 32

echo ""
echo ">>> [5/8] TextQA - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/textvqa.sh scope 32

echo ""
echo ">>> [6/8] MMbench - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmbench.sh scope 32

echo ""
echo ">>> [7/8] MMbench_cn - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmbench_cn.sh scope 32

echo ""
echo ">>> [8/8] MMVet - scope 32 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmvet.sh scope 32

# ==================== SCOPE with 64 tokens ====================
echo ""
echo ">>> [1/8] MME - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mme.sh scope 64

echo ""
echo ">>> [2/8] POPE - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/pope.sh scope 64

echo ""
echo ">>> [3/8] SQA - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/sqa.sh scope 64

echo ""
echo ">>> [4/8] GQA - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/gqa.sh scope 64

echo ""
echo ">>> [5/8] TextQA - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/textvqa.sh scope 64

echo ""
echo ">>> [6/8] MMbench - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmbench.sh scope 64

echo ""
echo ">>> [7/8] MMbench_cn - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmbench_cn.sh scope 64

echo ""
echo ">>> [8/8] MMVet - scope 64 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmvet.sh scope 64

# ==================== SCOPE with 128 tokens ====================
echo ""
echo ">>> [1/8] MME - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mme.sh scope 128

echo ""
echo ">>> [2/8] POPE - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/pope.sh scope 128

echo ""
echo ">>> [3/8] SQA - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/sqa.sh scope 128

echo ""
echo ">>> [4/8] GQA - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/gqa.sh scope 128

echo ""
echo ">>> [5/8] TextQA - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/textvqa.sh scope 128

echo ""
echo ">>> [6/8] MMbench - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmbench.sh scope 128

echo ""
echo ">>> [7/8] MMbench_cn - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmbench_cn.sh scope 128

echo ""
echo ">>> [8/8] MMVet - scope 128 tokens"
CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mmvet.sh scope 128

# ==================== SCOPE Hyperparameter Tuning (Optional) ====================
# Uncomment below to run hyperparameter tuning experiments

# echo ""
# echo ">>> [Optional] MME - scope 64 tokens, ALPHA=0.5"
# ALPHA=0.5 CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mme.sh scope 64

# echo ""
# echo ">>> [Optional] MME - scope 64 tokens, COMBINED=add"
# COMBINED=add CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/${MODEL_VERSION}/${MODEL_SCALE}/mme.sh scope 64

echo ""
echo "============================================"
echo ">>> All SCOPE evaluations completed!"
echo "============================================"
