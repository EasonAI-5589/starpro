#!/bin/bash
# ScoP MME evaluation: 128T, 64T, 32T
set -e
eval "$(conda shell.bash hook)"
conda activate fastv

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro/results

cd /mnt/eason/LLaVA-STAR-Pro

echo "================================================================"
echo "ScoP MME — Token={128, 64, 32}"
echo "Started: $(date)"
echo "================================================================"

for TOKEN in 128 64 32; do
    echo ""
    echo "════════════════ ${TOKEN}T ════════════════"
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/7b/mme.sh scope ${TOKEN}
    echo "Done: scope ${TOKEN}T at $(date)"
done

echo ""
echo "================================================================"
echo "ALL DONE: $(date)"
echo "================================================================"
