#!/bin/bash
cd /mnt/eason/LLaVA-STAR-Pro2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

GPU=$1
MULT=$2
SCHED=$3
TAG=$4

LOG=/mnt/eason/LLaVA-STAR-Pro2/results/ablation_${TAG}.log
echo "[$(date)] START ${TAG} GPU${GPU} MULT=${MULT} SCHED=${SCHED}" > $LOG

# POPE
echo "[$(date)] POPE" >> $LOG
LAMBDA=1.0 ENABLE_DEBUG=0 STAGE1_MULT=${MULT} CUSTOM_PRUNING_SCHEDULE="${SCHED}" CUDA_VISIBLE_DEVICES=${GPU} \
python -m llava.eval.model_vqa_loader --model-path /mnt/eason_ckp/models/llava-v1.5-7b \
--question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
--image-folder /mnt/eason_ckp/LLaVA-Eval/pope/val2014 \
--answers-file /tmp/pope_${TAG}.jsonl --num-chunks 1 --chunk-idx 0 --temperature 0 \
--conv-mode vicuna_v1 --pruning_method star_pro --visual_token_num 128 >> $LOG 2>&1
python llava/eval/eval_pope.py --annotation-dir /mnt/eason_ckp/LLaVA-Eval/pope/coco \
--question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
--result-file /tmp/pope_${TAG}.jsonl >> $LOG 2>&1

# TextVQA
echo "[$(date)] TextVQA" >> $LOG
LAMBDA=1.0 ENABLE_DEBUG=0 STAGE1_MULT=${MULT} CUSTOM_PRUNING_SCHEDULE="${SCHED}" CUDA_VISIBLE_DEVICES=${GPU} \
python -m llava.eval.model_vqa_loader --model-path /mnt/eason_ckp/models/llava-v1.5-7b \
--question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
--image-folder /mnt/eason_ckp/LLaVA-Eval/textvqa/train_images \
--answers-file /tmp/textvqa_${TAG}.jsonl --num-chunks 1 --chunk-idx 0 --temperature 0 \
--conv-mode vicuna_v1 --pruning_method star_pro --visual_token_num 128 >> $LOG 2>&1
python -m llava.eval.eval_textvqa --annotation-file /mnt/eason_ckp/LLaVA-Eval/textvqa/TextVQA_0.5.1_val.json \
--result-file /tmp/textvqa_${TAG}.jsonl >> $LOG 2>&1

# SQA
echo "[$(date)] SQA" >> $LOG
LAMBDA=1.0 ENABLE_DEBUG=0 STAGE1_MULT=${MULT} CUSTOM_PRUNING_SCHEDULE="${SCHED}" CUDA_VISIBLE_DEVICES=${GPU} \
python -m llava.eval.model_vqa_loader --model-path /mnt/eason_ckp/models/llava-v1.5-7b \
--question-file ./playground/data/eval/scienceqa/llava_test_CQM-I.json \
--image-folder /mnt/eason_ckp/LLaVA-Eval/scienceqa/images/test \
--answers-file /tmp/sqa_${TAG}.jsonl --num-chunks 1 --chunk-idx 0 --temperature 0 \
--conv-mode vicuna_v1 --pruning_method star_pro --visual_token_num 128 --single-pred-prompt >> $LOG 2>&1
python llava/eval/eval_science_qa.py \
--base-dir /mnt/eason_ckp/LLaVA-Eval/scienceqa \
--result-file /tmp/sqa_${TAG}.jsonl \
--output-file /tmp/sqa_${TAG}_output.jsonl \
--output-result /tmp/sqa_${TAG}_result.json >> $LOG 2>&1

echo "[$(date)] ALL DONE ${TAG}" >> $LOG
