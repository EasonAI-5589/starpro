#!/bin/bash
# ==================== Configuration ====================
# Stage 1 (THCP) Lambda Ablation Study
# Formula: L_i(S) = 0.5 × R_i + (λ/2) × D_i(S)
# Fixed relevance weight = 0.5, diversity weight = λ/2

# Model configuration
MODEL_VERSION="v1_5"
MODEL_SCALE="7b"
METHOD="star_v3"
TOKEN_BUDGET=128

# Lambda values to test
# λ=0.5 → Diversity=0.25 (0.5R+0.25D)
# λ=1.0 → Diversity=0.5  (0.5R+0.5D, balanced)
# λ=2.0 → Diversity=1.0  (0.5R+1.0D)
LAMBDA_VALUES=(1.0)

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
    echo "Stage 1 Lambda Ablation Study"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  Model: llava-${MODEL_VERSION}-${MODEL_SCALE}"
    echo "  Method: ${METHOD}"
    echo "  Token Budget: ${TOKEN_BUDGET}"
    echo "  Formula: L_i(S) = 0.5 × R_i + (λ/2) × D_i(S)"
    echo "  Fixed Relevance Weight: 0.5"
    echo "  Diversity Weight: λ/2"
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
    diversity_weight=$(awk "BEGIN {print $lambda/2}")

    {
        echo "=========================================="
        echo ">>> Lambda: $lambda"
        echo ">>> Diversity Weight: $diversity_weight (λ/2)"
        echo ">>> Formula: L_i(S) = 0.5 × R_i + $diversity_weight × D_i(S)"
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
        echo ">>> ✓ Finished Lambda = $lambda (Diversity Weight = $diversity_weight)"
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
