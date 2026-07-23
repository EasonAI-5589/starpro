#!/bin/bash

#d2p
# export ENABLE_DEBUG=1
# export DIV_1=0
# export DIV_2=1.5
# bash scripts/v1_6/13b/pope.sh prefixvlm_2 32 

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
# export LAMBDA_1=0.1 #越大表示越考虑第一阶段的coverage
# export ALPHA_2=1.0 #对应prefix中的LAMBDA=0.5
# export LAMBDA_2=0.1 #越大表示越考虑第二阶段的coverage
# export STAGE1_KEEP=320 #--160
# export STAGE1_KEEP=640 #--320
# export STAGE1_KEEP=1280 #--640  

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

CKPT="llava-v1.6-vicuna-13b"
SPLIT="llava_pope_test"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_$((TOKEN * 5))"

# 🔧 从脚本路径自动识别模型版本和规模
# 脚本路径示例: scripts/v1_5/7b/mme.sh
SCRIPT_PATH=$(dirname "${BASH_SOURCE[0]}")
MODEL_VERSION=$(basename $(dirname $(dirname "$SCRIPT_PATH")))
MODEL_SCALE=$(basename $(dirname "$SCRIPT_PATH"))


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


# 保存结果
RESULT_DIR="${RESULT_DIR:-/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/code/EasonAI/STAR-LLaVA/results}"
mkdir -p ${RESULT_DIR}
LOG_FILE="${RESULT_DIR}/pope_${CKPT}_${METHOD}_vtn${TOKEN}.log"

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