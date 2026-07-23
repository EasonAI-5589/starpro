#!/bin/bash
# 可视化 STAR-Pro (prefixvlm_2) 的分阶段剪枝过程（v1.6-7b）
# 每个样本输出三张图：原图、Stage1(arch后)、Stage2+(LLM内)
#
# 用法:
#   cd LLaVA-STAR-Pro2
#   conda activate llava
#   bash visualize_prefixvlm2.sh

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

# ==================== Configuration ====================
MODEL_VERSION="v1_5"
MODEL_SCALE="7b"
TOKEN_BUDGET=64

# Dataset: mme, mmvet, pope, textvqa, gqa, sqa
#   or leave empty and set SAMPLES_FILE for a custom json
DATASET="textvqa"
SAMPLES_FILE="/mnt/eason/LLaVA-STAR-Pro2/vis_mmvet_better.json"

MAX_SAMPLES=200
GPUS="0"

# STAR-Pro hyperparameters
export STAGE1_KEEP=$(( TOKEN_BUDGET * 2 ))
export ALPHA_1=1
export LAMBDA_1=1
export ALPHA_2=1
export LAMBDA_2=1
export COVERAGE=SCOPE
export ENABLE_DEBUG=0

# ==================== Derived paths ====================
CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"

CKPT="llava-${MODEL_VERSION//_/.}-${MODEL_SCALE}"
MODEL_PATH="${CKPT_DIR}/${CKPT}"
OUTPUT_DIR="/mnt/eason/LLaVA-STAR-Pro2/vis_mmvet_better_output"

echo "=========================================="
echo ">>> Method:      STAR-Pro (prefixvlm_2)"
echo ">>> Model:       ${MODEL_PATH}"
echo ">>> Dataset:     ${DATASET}"
echo ">>> Budget:      ${TOKEN_BUDGET}  STAGE1_KEEP: ${STAGE1_KEEP}"
echo ">>> Output:      ${OUTPUT_DIR}"
echo "=========================================="

if [ -n "${SAMPLES_FILE}" ]; then
    DATASET_ARG="--samples_file ${SAMPLES_FILE}"
else
    DATASET_ARG="--dataset ${DATASET}"
fi

CUDA_VISIBLE_DEVICES=${GPUS} python visualize_prefixvlm2.py \
    --model_path   ${MODEL_PATH} \
    --token_budget ${TOKEN_BUDGET} \
    --output_dir   ${OUTPUT_DIR} \
    --max_samples  ${MAX_SAMPLES} \
    ${DATASET_ARG}
