#!/bin/bash

# ==================== Configuration ====================
# Model version: v1_5, v1_6
MODEL_VERSION="v1_6"

# Model scale: 7b, 13b
MODEL_SCALE="7b"

# Compression method: fastv, sparsevlm, pdrop, visionzip, dart, divprune, dp3, cdp3, star, star_v2, thcp, vanilla
METHOD="star_v2"

# Token budgets: 32, 64, 128, 192, 256
TOKEN_BUDGETS=(128 64 32)

# GPU devices
GPUS="0,1,2,3,4,5,6,7"

# Evaluation tasks: mme, pope, sqa, textvqa, gqa, vqav2, vizwiz, mmbench, mmbench_cn, mmvet
TASKS=(vqav2 vizwiz mmbench mmbench_cn mmvet)

# ==================== Run Evaluations ====================
for task in "${TASKS[@]}"; do
    for token_budget in "${TOKEN_BUDGETS[@]}"; do
        echo "=========================================="
        echo ">>> Running: $task with $METHOD (budget=$token_budget)"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $token_budget
    done
done
