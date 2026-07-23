#!/bin/bash

#d2p
# export ENABLE_DEBUG=1
# export DIV_1=0
# export DIV_2=1.5
# bash scripts/v1_5/7b/pope.sh prefixvlm_2 32 

# #prefixvlm
# export ENABLE_DEBUG=1
# export ALPHA=0.6 
# export LAMBDA=0.5
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
# export STAGE1_KEEP=256 #--128 

#prefixvlm_2
# export http_proxy=http://192.168.32.28:18000  
# export https_proxy=http://192.168.32.28:18000  
# export ENABLE_DEBUG=1
# export ALPHA_1=4 #1.5对应prefix中的ALPHA=0.6
# export LAMBDA_1=1 #越大表示越考虑第一阶段的coverage
# export ALPHA_2=4 #对应prefix中的LAMBDA=0.5
# export LAMBDA_2=1 #越大表示越考虑第二阶段的coverage
# export COVERAGE=SCOPE
# export STAGE1_KEEP=64 #--32
# export STAGE1_KEEP=128 #--64
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
SPLIT="llava_pope_test"

METHOD=${1}
TOKEN=${2}
VSCAN_STAGE2=${3:-32}  # VScan Stage 2 tokens (default: 32)
PARAM="vtn_${TOKEN}${RUN_TAG:+_${RUN_TAG}}"

# 🔬 VScan: stage1=${TOKEN}, stage2=${VSCAN_STAGE2}
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


ANSWER_DIR=./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}
mkdir -p ${ANSWER_DIR}

# 清掉上一次运行残留的分片，避免不同 CHUNKS 的结果混在一起污染 merge
rm -f ${ANSWER_DIR}/*_*.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/pope/val2014 \
        --answers-file ${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        ${VSCAN_ARGS} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${ANSWER_DIR}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 校验：答案数必须与问题数一致，否则算出来的分数无意义
n_q=$(wc -l < ./playground/data/eval/pope/${SPLIT}.jsonl)
n_a=$(wc -l < "$output_file")
if [ "$n_q" != "$n_a" ]; then
    echo "[ERROR] 答案不完整: 问题 ${n_q} 条，答案 ${n_a} 条 —— 拒绝算分" >&2
    exit 1
fi
echo ">>> 答案完整性校验通过: ${n_a}/${n_q}"

# 保存结果
RESULT_DIR="${RESULT_DIR:-$(pwd)/results}"
mkdir -p ${RESULT_DIR}
LOG_FILE="${RESULT_DIR}/pope_${MODEL_VERSION}_${MODEL_SCALE}_${METHOD}_${PARAM}.log"

# 写入配置和结果
echo ">>> 实验配置：" > ${LOG_FILE}
echo ">>>   模型版本: ${MODEL_VERSION}" >> ${LOG_FILE}
echo ">>>   模型规模: ${MODEL_SCALE}" >> ${LOG_FILE}
echo ">>>   压缩方法: ${METHOD}" >> ${LOG_FILE}
echo ">>>   Token预算: ${TOKEN}" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}

python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_DIR}/pope/coco \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --result-file $output_file | tee -a ${LOG_FILE}
