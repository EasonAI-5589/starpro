#!/bin/bash
# Rerun SCOPE mmbench_cn with fastv env, 3 configs (vtn 128/64/32)
# 串行跑，跑完生成 xlsx 提交文件

set -e

export CKPT_DIR="/mnt/eason_ckp/models"
export DATA_DIR="/mnt/eason_ckp/LLaVA-Eval"

cd /mnt/eason/LLaVA-STAR-Pro

METHOD="scope"

for TOKEN in 128 64 32; do
    echo ""
    echo "=========================================="
    echo ">>> Running SCOPE MMBench-CN T=${TOKEN}"
    echo ">>> $(date)"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/7b/mmbench_cn.sh ${METHOD} ${TOKEN}
    
    # 验证行数
    MERGE="playground/data/eval/mmbench_cn/answers/mmbench_dev_cn_20231003/llava-v1.5-7b/scope/vtn_${TOKEN}/merge.jsonl"
    LINES=$(wc -l < "$MERGE")
    echo ">>> vtn_${TOKEN} merge.jsonl: ${LINES} lines"
    if [ "$LINES" -lt 4000 ]; then
        echo "ERROR: Expected ~4329 lines, got ${LINES}. Something went wrong!"
        exit 1
    fi
    echo ">>> vtn_${TOKEN} DONE ✅"
done

# 复制 xlsx 到指定目录
TARGET_DIR="playground/data/eval/mm-vet/answers_upload/mmbench_cn/llava-v1.5-7b/scope"
mkdir -p "$TARGET_DIR"

for TOKEN in 128 64 32; do
    SRC="playground/data/eval/mmbench_cn/answers_upload/mmbench_dev_cn_20231003/llava-v1.5-7b/scope/vtn_${TOKEN}.xlsx"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$TARGET_DIR/vtn_${TOKEN}.xlsx"
        echo "Copied vtn_${TOKEN}.xlsx to ${TARGET_DIR}/"
    else
        echo "WARNING: $SRC not found!"
    fi
done

echo ""
echo "=========================================="
echo ">>> All 3 configs done! $(date)"
echo "=========================================="
