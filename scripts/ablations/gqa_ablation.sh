#!/bin/bash
################################################################################
# GQA Benchmark - Stage 1 Lambda Ablation
################################################################################

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_${TOKEN}"

# Lambda ablation 专用配置
LAMBDA_VALUE=${LAMBDA:-0.5}  # 从环境变量读取，默认0.5
MODEL_VERSION="v1_5"
MODEL_SCALE="7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

GQA_DIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst ${GQA_DIR}/testdev_balanced_predictions.json

# ==================== Lambda Ablation 结果保存 ====================
RESULT_DIR="./results/ablation_stage1_lambda"
mkdir -p ${RESULT_DIR}
LOG_FILE="${RESULT_DIR}/lambda_${LAMBDA_VALUE}_gqa_${MODEL_VERSION}_${MODEL_SCALE}_${METHOD}_vtn${TOKEN}.log"

# 计算 relevance weight
RELEVANCE_WEIGHT=$(awk "BEGIN {print 1-${LAMBDA_VALUE}}")

# 写入配置和结果
echo "================================================================================" > ${LOG_FILE}
echo "Stage 1 (THCP) Lambda Ablation - GQA Benchmark" >> ${LOG_FILE}
echo "================================================================================" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}
echo ">>> Lambda Configuration:" >> ${LOG_FILE}
echo ">>>   λ (lambda): ${LAMBDA_VALUE}" >> ${LOG_FILE}
echo ">>>   Formula: L_i(S) = (1-λ)R_i + λD_i(S)" >> ${LOG_FILE}
echo ">>>   Relevance weight (1-λ): ${RELEVANCE_WEIGHT}" >> ${LOG_FILE}
echo ">>>   Diversity weight (λ): ${LAMBDA_VALUE}" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}
echo ">>> Model Configuration:" >> ${LOG_FILE}
echo ">>>   模型版本: ${MODEL_VERSION}" >> ${LOG_FILE}
echo ">>>   模型规模: ${MODEL_SCALE}" >> ${LOG_FILE}
echo ">>>   压缩方法: ${METHOD}" >> ${LOG_FILE}
echo ">>>   Token预算: ${TOKEN}" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}
echo "================================================================================" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}

cd ${GQA_DIR}
python eval/eval.py \
    --path ${DATA_DIR}/gqa/data/questions \
    --tier testdev_balanced | tee -a ../../../../${LOG_FILE}

rm testdev_balanced_predictions.json
