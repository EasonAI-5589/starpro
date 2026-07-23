#!/bin/bash

#d2p
# export ENABLE_DEBUG=1
# export DIV_1=0
# export DIV_2=1.5
# bash scripts/v1_5/7b/textvqa.sh prefixvlm_2 64 

# #prefixvlm
# export ENABLE_DEBUG=1
# export ALPHA=0.6 
# export LAMBDA=0.5
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128 

#prefixvlm_2
# export ENABLE_DEBUG=1
# export ALPHA_1=1 #1.5对应prefix中的ALPHA=0.6
# export LAMBDA_1=0.7 #越大表示越考虑第一阶段的coverage
# export ALPHA_2=1.0 #对应prefix中的LAMBDA=0.5
# export LAMBDA_2=0.7 #越大表示越考虑第二阶段的coverage
# export EMA_ALPHA=0 #越小表示越考虑当前的attention，越大表示越考虑历史的attention
# # export STAGE1_KEEP=64 #--32
# # export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128  

# #svd
# export SVD_TAU=0.95
# export ALPHA=0.6 
# export LAMBDA=0.48 

# export NUM_CLUSTERS=64
# export ATTN_THRESHOLD_RATIO=0.5
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"

CKPT="llava-v1.5-7b"
SPLIT="llava_textvqa_val_v051_ocr"

METHOD=${1}
TOKEN=${2}
VSCAN_STAGE2=${3:-32}  # VScan Stage 2 tokens (default: 32)
PARAM="vtn_${TOKEN}${RUN_TAG:+_${RUN_TAG}}"

# 🔬 VScan: stage1=${TOKEN}, stage2=${VSCAN_STAGE2}
# Average tokens = (stage1 + stage2) / 2
VSCAN_ARGS=""
if [ "$METHOD" == "vscan" ]; then
    AVG=$(( (TOKEN + VSCAN_STAGE2) / 2 ))
    PARAM="vtn_${AVG}${RUN_TAG:+_${RUN_TAG}}"
    VSCAN_ARGS="--vscan_stage2_tokens ${VSCAN_STAGE2}"
    echo ">>> VScan: stage1=${TOKEN}, stage2=${VSCAN_STAGE2}, avg=${AVG}"
fi

# 🔧 从脚本路径自动识别模型版本和规模
# 脚本路径示例: scripts/v1_5/7b/mme.sh
SCRIPT_PATH=$(dirname "${BASH_SOURCE[0]}")
MODEL_VERSION=$(basename $(dirname $(dirname "$SCRIPT_PATH")))
MODEL_SCALE=$(basename $(dirname "$SCRIPT_PATH"))

ANSWER_DIR=./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}
mkdir -p "${ANSWER_DIR}"
rm -f "${ANSWER_DIR}"/*_*.jsonl

pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/textvqa/train_images \
        --answers-file "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        ${VSCAN_ARGS} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    pids+=("$!")
done

worker_failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        worker_failed=1
    fi
done
if [ "${worker_failed}" -ne 0 ]; then
    echo "[ERROR] At least one distributed TextVQA worker failed" >&2
    exit 1
fi

output_file="${ANSWER_DIR}/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

n_q=$(wc -l < ./playground/data/eval/textvqa/${SPLIT}.jsonl)
n_a=$(wc -l < "$output_file")
if [ "$n_q" != "$n_a" ]; then
    echo "[ERROR] TextVQA answers incomplete: questions=${n_q}, answers=${n_a}" >&2
    exit 1
fi
echo ">>> TextVQA answer completeness check passed: ${n_a}/${n_q}"

# 保存结果
RESULT_DIR="${RESULT_DIR:-/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/code/EasonAI/STAR-LLaVA/results}"
mkdir -p ${RESULT_DIR}
LOG_FILE="${RESULT_DIR}/textvqa_${MODEL_VERSION}_${MODEL_SCALE}_${METHOD}_${PARAM}.log"

# 写入配置和结果
echo ">>> 实验配置：" > ${LOG_FILE}
echo ">>>   模型版本: ${MODEL_VERSION}" >> ${LOG_FILE}
echo ">>>   模型规模: ${MODEL_SCALE}" >> ${LOG_FILE}
echo ">>>   压缩方法: ${METHOD}" >> ${LOG_FILE}
echo ">>>   Token预算: ${TOKEN}" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}


python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file | tee -a ${LOG_FILE}
