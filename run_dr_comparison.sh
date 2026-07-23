#!/bin/bash
source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate fastv
export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export RESULT_DIR=/mnt/eason/LLaVA-STAR-Pro/results
unset ENABLE_DEBUG

GPUS="0,1,2,3"
TOKENS=(128 64 32)
TASKS=(gqa pope textvqa)
MODES=("D-R:1" "D+R:0")

LOGFILE="/mnt/eason/LLaVA-STAR-Pro/results/dr_comparison_$(date +%Y%m%d_%H%M%S).log"
echo "D-R vs D+R Comparison" > "$LOGFILE"
echo "=====================" >> "$LOGFILE"

for task in "${TASKS[@]}"; do
    for token in "${TOKENS[@]}"; do
        for mode in "${MODES[@]}"; do
            name="${mode%%:*}"
            negate="${mode##*:}"
            
            export NEGATE_RELEVANCE=$negate
            export RELEVANCE_WEIGHT=1.0
            export DIVERSITY_WEIGHT=1.0
            
            echo ""
            echo "=========================================="
            echo ">>> $name | $task | ${token} tokens"
            echo "=========================================="
            
            result=$(CUDA_VISIBLE_DEVICES=$GPUS bash scripts/v1_5/7b/$task.sh star_pro $token 2>&1 | tail -3)
            echo "$name | $task | ${token}T: $result"
            echo "$name | $task | ${token}T: $result" >> "$LOGFILE"
        done
    done
done

echo ""
echo "=========================================="
echo "All done! Results saved to: $LOGFILE"
echo "=========================================="
