#!/bin/bash
# cd /mnt/eason/LLaVA-STAR-Pro2
# conda activate llava
# bash visualize_fastv.sh

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

# ==================== Configuration ====================
MODEL_VERSION="v1_5"
MODEL_SCALE="7b"
TOKEN_BUDGET=64
METHOD="fastv"

# Dataset: mme, mmvet, pope, textvqa, gqa, sqa
DATASET="textvqa"
SAMPLES_FILE=""

MAX_SAMPLES=200
GPUS="0"

# ==================== Derived paths ====================
CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"

CKPT="llava-${MODEL_VERSION//_/.}-${MODEL_SCALE}"
MODEL_PATH="${CKPT_DIR}/${CKPT}"
OUTPUT_DIR="/mnt/eason/LLaVA-STAR-Pro2/${METHOD}_vis_output/${MODEL_VERSION}_${MODEL_SCALE}/${DATASET}_vtn${TOKEN_BUDGET}"

echo "=========================================="
echo ">>> Method:  ${METHOD}"
echo ">>> Model:   ${MODEL_PATH}"
echo ">>> Dataset: ${DATASET}"
echo ">>> Budget:  ${TOKEN_BUDGET}"
echo ">>> Output:  ${OUTPUT_DIR}"
echo "=========================================="

if [ -n "${SAMPLES_FILE}" ]; then
    DATASET_ARG="--samples_file ${SAMPLES_FILE}"
else
    DATASET_ARG="--dataset ${DATASET}"
fi

CUDA_VISIBLE_DEVICES=${GPUS} python visualize_pruning.py \
    --method       ${METHOD} \
    --model_path   ${MODEL_PATH} \
    --token_budget ${TOKEN_BUDGET} \
    --output_dir   ${OUTPUT_DIR} \
    --max_samples  ${MAX_SAMPLES} \
    ${DATASET_ARG}
