#!/bin/bash
# ==================== Configuration ====================
# Stage 1 (THCP) Lambda Ablation Study
# Formula: L_i(S) = (1-λ)R_i + λD_i(S)

# Model configuration
MODEL_VERSION="v1_5"
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

# Results directory and log file
RESULT_DIR="./results/ablation_stage1_lambda"
mkdir -p ${RESULT_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/ablation_lambda_${TIMESTAMP}.log"

echo "Results will be saved to: ${LOG_FILE}"
echo ""

# Write experiment configuration to log
{
    echo "================================================================================"
    echo "Stage 1 (THCP) Lambda Ablation Study"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  Model: llava-${MODEL_VERSION}-${MODEL_SCALE}"
    echo "  Method: ${METHOD}"
    echo "  Token Budget: ${TOKEN_BUDGET}"
    echo "  Formula: L_i(S) = (1-λ)R_i + λD_i(S)"
    echo ""
    echo "Lambda values: ${LAMBDA_VALUES[@]}"
    echo "Benchmarks: ${TASKS[@]}"
    echo "Timestamp: $(date)"
    echo ""
    echo "================================================================================"
    echo ""
} | tee "${LOG_FILE}"

# ==================== Run Evaluations ====================
for lambda in "${LAMBDA_VALUES[@]}"; do
    relevance=$(awk "BEGIN {print 1-$lambda}")

    {
        echo "=========================================="
        echo ">>> Lambda: $lambda (Relevance=$relevance, Diversity=$lambda)"
        echo "=========================================="
    } | tee -a "${LOG_FILE}"

    # Export lambda for this run
    export LAMBDA=$lambda

    # Export debug flag (controls both star_v3 debug output and tqdm progress bar)
    export ENABLE_DEBUG=$ENABLE_DEBUG

    for task in "${TASKS[@]}"; do
        echo ">>> Running: $task with $METHOD (lambda=$lambda, budget=$TOKEN_BUDGET)" | tee -a "${LOG_FILE}"
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET 2>&1 | tee -a "${LOG_FILE}"
    done

    {
        echo ""
        echo ">>> ✓ Finished Lambda=$lambda (Relevance=$relevance, Diversity=$lambda)"
        echo ""
    } | tee -a "${LOG_FILE}"
done

{
    echo "=========================================="
    echo "Stage 1 Lambda Ablation Completed!"
    echo "=========================================="
    echo ""
    echo "Full log saved to: ${LOG_FILE}"
} | tee -a "${LOG_FILE}"
