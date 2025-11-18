#!/bin/bash
# ==================== Configuration ====================
# Stage 1 (THCP) Lambda Ablation Study
# Formula: L_i(S) = (1-λ)R_i + λD_i(S)

# Model configuration
MODEL_VERSION="v1_6"
MODEL_SCALE="7b"
METHOD="star_v3"
TOKEN_BUDGET=128

# Lambda values to test
LAMBDA_VALUES=(0.5)

# Evaluation tasks
TASKS=(mme)

# GPU devices
GPUS="0,1,2,3,4,5,6,7"

# Debug configuration
# Set to "1" to enable debug output and progress bars
# Set to "0" for clean logs (recommended for ablation experiments)
ENABLE_DEBUG=0

# ==================== Run Evaluations ====================
for lambda in "${LAMBDA_VALUES[@]}"; do
    relevance=$(awk "BEGIN {print 1-$lambda}")
    echo "=========================================="
    echo ">>> Lambda: $lambda (Relevance=$relevance, Diversity=$lambda)"
    echo "=========================================="

    # Export lambda for this run
    export LAMBDA=$lambda

    # Export debug flag (controls both star_v3 debug output and tqdm progress bar)
    export ENABLE_DEBUG=$ENABLE_DEBUG

    for task in "${TASKS[@]}"; do
        echo ">>> Running: $task with $METHOD (lambda=$lambda, budget=$TOKEN_BUDGET)"
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET
    done

    echo ""
    echo ">>> ✓ Finished Lambda=$lambda (Relevance=$relevance, Diversity=$lambda)"
    echo ""
done

echo "=========================================="
echo "Stage 1 Lambda Ablation Completed!"
echo "=========================================="
