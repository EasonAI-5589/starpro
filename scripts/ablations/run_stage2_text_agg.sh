#!/bin/bash
# ==================== Configuration ====================
# Stage 2 Text Aggregation Ablation Study
# Test different text token aggregation strategies for Stage 2 progressive pruning

# Model configuration
MODEL_VERSION="v1_5"
MODEL_SCALE="7b"
METHOD="star_v3"
TOKEN_BUDGET=128

# Fixed lambda from Stage 1 ablation
LAMBDA=0.5

# Text aggregation modes to test
TEXT_AGG_MODES=(last_token multi_token average_all)

# Evaluation tasks
TASKS=(mme)

# GPU devices
GPUS="0,1,2,3,4,5,6,7"

# Debug configuration
ENABLE_DEBUG=0

# Results directory and log file
RESULT_DIR="./results/ablation_stage2_text_agg"
mkdir -p ${RESULT_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/ablation_text_agg_${TIMESTAMP}.log"

echo "Results will be saved to: ${LOG_FILE}"
echo ""

# Write experiment configuration to log
{
    echo "================================================================================"
    echo "Stage 2 Text Aggregation Ablation Study"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  Model: llava-${MODEL_VERSION}-${MODEL_SCALE}"
    echo "  Method: ${METHOD}"
    echo "  Token Budget: ${TOKEN_BUDGET}"
    echo "  Lambda (fixed): ${LAMBDA}"
    echo ""
    echo "Text Aggregation Modes:"
    echo "  - last_token: Use only last text token (PDrop baseline)"
    echo "  - multi_token: Importance-weighted multi-token [Ours]"
    echo "  - average_all: Simple average of all text tokens"
    echo ""
    echo "Modes to test: ${TEXT_AGG_MODES[@]}"
    echo "Benchmarks: ${TASKS[@]}"
    echo "Timestamp: $(date)"
    echo ""
    echo "================================================================================"
    echo ""
} | tee "${LOG_FILE}"

# ==================== Run Evaluations ====================
for mode in "${TEXT_AGG_MODES[@]}"; do
    {
        echo "=========================================="
        echo ">>> Text Aggregation Mode: $mode"
        echo "=========================================="
    } | tee -a "${LOG_FILE}"

    # Export configuration for this run
    export TEXT_AGG_MODE=$mode
    export LAMBDA=$LAMBDA
    export ENABLE_DEBUG=$ENABLE_DEBUG

    for task in "${TASKS[@]}"; do
        echo ">>> Running: $task with $METHOD (text_agg=$mode, lambda=$LAMBDA, budget=$TOKEN_BUDGET)" | tee -a "${LOG_FILE}"
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET 2>&1 | tee -a "${LOG_FILE}"
    done

    {
        echo ""
        echo ">>> ✓ Finished Text Aggregation Mode: $mode"
        echo ""
    } | tee -a "${LOG_FILE}"
done

{
    echo "=========================================="
    echo "Stage 2 Text Aggregation Ablation Completed!"
    echo "=========================================="
    echo ""
    echo "Full log saved to: ${LOG_FILE}"
} | tee -a "${LOG_FILE}"
