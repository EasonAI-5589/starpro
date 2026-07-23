#!/bin/bash
# ------------------------------------------------------------------------
# DUET-VLM (AMD-AGI, CVPR2026) baseline eval on POPE, LLaVA-1.5-7B.
# DUET = VisionZip (stage-1) + salient-guided PyramidDrop (stage-2).
# Runs in the dedicated `duet` conda env (has spacy/en_core_web_sm for T2V).
# Usage:   bash scripts/v1_5/7b/pope_duet.sh [BUDGET]
#   BUDGET is the AVERAGE-over-layers visual-token label (default 192).
# Override the exact VisionZip/PyramidDrop knobs via env if needed:
#   DUET_DOMINANT DUET_CONTEXTUAL DUET_CW DUET_LAYERS DUET_RATIOS DUET_SALIENT
#
# NOTE on budgets: the upstream LLaVA scripts ship only the ~192-avg config
# (dominant=300 contextual=7 layer_list=[16,24] ratios=[0.5,0.0]). Other paper
# points (e.g. 128/64 avg) need their own (dominant,contextual,schedule); set
# them via the env vars above once confirmed. This script defaults to the
# verified 192 config.
# ------------------------------------------------------------------------
set -uo pipefail

# 8-card by default (STAR-Pro 8-card rule); wrapper may override CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ENABLE_DEBUG="${ENABLE_DEBUG:-1}"
export NLTK_DATA="${NLTK_DATA:-/mnt/eason/nltk_data}"
# GitHub proxy (harmless if unused; some datasets/tokenizers fetch)
export http_proxy="${http_proxy:-http://192.168.32.28:18000}"
export https_proxy="${https_proxy:-http://192.168.32.28:18000}"

gpu_list="$CUDA_VISIBLE_DEVICES"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"
CKPT="llava-v1.5-7b"
SPLIT="llava_pope_test"

BUDGET="${1:-192}"
# ---- DUET config (192-avg default; override via env) ----
DUET_DOMINANT="${DUET_DOMINANT:-300}"
DUET_CONTEXTUAL="${DUET_CONTEXTUAL:-7}"
DUET_CW="${DUET_CW:-4}"
DUET_LAYERS="${DUET_LAYERS:-[16,24]}"
DUET_RATIOS="${DUET_RATIOS:-[0.5,0.0]}"
DUET_SALIENT="${DUET_SALIENT:-1}"          # 1 = enable T2V salient guidance
SALIENT_FLAG=""
[ "$DUET_SALIENT" = "1" ] && SALIENT_FLAG="--compute_salient_tokens"

METHOD="duet"
PARAM="vtn_${BUDGET}${RUN_TAG:+_${RUN_TAG}}"

ANSWER_DIR=./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}
mkdir -p ${ANSWER_DIR}
rm -f ${ANSWER_DIR}/*_*.jsonl

echo ">>> DUET: budget(avg)=${BUDGET} dominant=${DUET_DOMINANT} contextual=${DUET_CONTEXTUAL} cw=${DUET_CW} layers=${DUET_LAYERS} ratios=${DUET_RATIOS} salient=${DUET_SALIENT}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} timeout -k 60 5400 python -m llava.eval.model_vqa_loader_duet \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/pope/val2014 \
        --answers-file ${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method duet \
        --visual_token_num ${BUDGET} \
        --dominant ${DUET_DOMINANT} \
        --contextual ${DUET_CONTEXTUAL} \
        --cluster_width ${DUET_CW} \
        --layer_list "${DUET_LAYERS}" \
        --image_token_ratio_list "${DUET_RATIOS}" \
        ${SALIENT_FLAG} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait
# NOTE: no broad `pkill -f model_vqa_loader_duet` here — it would kill sibling
# budget runs when several pope_duet.sh run in parallel. `wait` above already
# joins this run's own chunk PIDs.

output_file=${ANSWER_DIR}/merge.jsonl
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

n_q=$(wc -l < ./playground/data/eval/pope/${SPLIT}.jsonl)
n_a=$(wc -l < "$output_file")
if [ "$n_q" != "$n_a" ]; then
    echo "[ERROR] answers incomplete: questions ${n_q}, answers ${n_a} -- refusing to score" >&2
    exit 1
fi
echo ">>> answer completeness OK: ${n_a}/${n_q}"

RESULT_DIR="${RESULT_DIR:-$(pwd)/results}"
mkdir -p ${RESULT_DIR}
LOG_FILE="${RESULT_DIR}/pope_v1_5_7b_${METHOD}_${PARAM}.log"
{
  echo ">>> DUET-VLM POPE eval"
  echo ">>>   budget(avg): ${BUDGET}"
  echo ">>>   dominant=${DUET_DOMINANT} contextual=${DUET_CONTEXTUAL} cw=${DUET_CW}"
  echo ">>>   layer_list=${DUET_LAYERS} ratios=${DUET_RATIOS} salient=${DUET_SALIENT}"
  echo ""
} > ${LOG_FILE}

python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_DIR}/pope/coco \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --result-file $output_file | tee -a ${LOG_FILE}
