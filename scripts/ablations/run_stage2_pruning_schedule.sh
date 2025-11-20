#!/bin/bash
# ==================== Configuration ====================
# Stage 2 Pruning Schedule Ablation Study
# Test different pruning schedules for Stage 2 progressive pruning

# Model configuration
MODEL_VERSION="v1_5"
MODEL_SCALE="7b"
METHOD="star_v3"
TOKEN_BUDGET=128

# Fixed lambda from Stage 1 ablation
LAMBDA=0.5

# Pruning schedule configurations to test
# Format: "name|mode|custom_schedule"
# - name: Display name for the schedule
# - mode: predefined mode (progressive/single_stage/uniform) or "custom"
# - custom_schedule: JSON array (only used if mode="custom")

# ===== Pruning Schedule Configurations =====
# Format: "Display Name|custom|[[layer1, tokens1], [layer2, tokens2], ...]"
#
# Schedule syntax:
#   [(layer_idx, tokens)] - Single-stage pruning
#   [(layer1, tokens1), (layer2, tokens2)] - Two-stage pruning
#   [(layer1, tokens1), (layer2, tokens2), (layer3, tokens3)] - Three-stage pruning
#
# Example calculations for average = 128:
#   Two-stage [(12, 64), (24, 32)]:
#     Layers 0-11: 256 tokens (12 layers)
#     Layers 12-23: 64 tokens (12 layers)
#     Layers 24-31: 32 tokens (8 layers)
#     Average: (12×256 + 12×64 + 8×32) / 32 = 128.0 ✓

SCHEDULE_CONFIGS=(
    # TODO: 在这里填入你的自定义配置
    # 示例:
    # "Single-stage|custom|[[16, 32]]"
    # "Two-stage [Ours]|custom|[[12, 64], [24, 32]]"
    # "Three-stage|custom|[[8, 128], [18, 72], [26, 32]]"

    "Two-stage [STAR, Ours]|custom|[[12, 64], [24, 32]]"
)

# Evaluation tasks
TASKS=(mme)

# GPU devices
GPUS="0,1,2,3,4,5,6,7"

# Debug configuration
ENABLE_DEBUG=0

# Results directory and log file
RESULT_DIR="./results/ablation_stage2_pruning_schedule"
mkdir -p ${RESULT_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/ablation_pruning_schedule_${TIMESTAMP}.log"

echo "Results will be saved to: ${LOG_FILE}"
echo ""

# Write experiment configuration to log
{
    echo "================================================================================"
    echo "Stage 2 Pruning Schedule Ablation Study"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  Model: llava-${MODEL_VERSION}-${MODEL_SCALE}"
    echo "  Method: ${METHOD}"
    echo "  Token Budget: ${TOKEN_BUDGET}"
    echo "  Lambda (fixed): ${LAMBDA}"
    echo ""
    echo "Pruning Schedule Configurations:"
    for config in "${SCHEDULE_CONFIGS[@]}"; do
        IFS='|' read -r name mode schedule <<< "$config"
        if [ "$mode" = "custom" ]; then
            echo "  - $name: Custom schedule $schedule"
        else
            echo "  - $name: mode=$mode"
        fi
    done
    echo ""
    echo "All schedules maintain average = 128 tokens for fair comparison"
    echo ""
    echo "Benchmarks: ${TASKS[@]}"
    echo "Timestamp: $(date)"
    echo ""
    echo "================================================================================"
    echo ""
} | tee "${LOG_FILE}"

# ==================== Run Evaluations ====================
for config in "${SCHEDULE_CONFIGS[@]}"; do
    # Parse configuration
    IFS='|' read -r name mode custom_schedule <<< "$config"

    {
        echo "=========================================="
        echo ">>> Pruning Schedule: $name"
        echo "=========================================="
    } | tee -a "${LOG_FILE}"

    # Export configuration for this run
    export LAMBDA=$LAMBDA
    export ENABLE_DEBUG=$ENABLE_DEBUG

    if [ "$mode" = "custom" ]; then
        # Use custom schedule
        export CUSTOM_PRUNING_SCHEDULE="$custom_schedule"
        unset PRUNING_SCHEDULE_MODE
        echo ">>> Using custom schedule: $custom_schedule" | tee -a "${LOG_FILE}"
    else
        # Use predefined mode
        export PRUNING_SCHEDULE_MODE="$mode"
        unset CUSTOM_PRUNING_SCHEDULE
        echo ">>> Using predefined mode: $mode" | tee -a "${LOG_FILE}"
    fi

    for task in "${TASKS[@]}"; do
        echo ">>> Running: $task with $METHOD (schedule=$name, lambda=$LAMBDA, budget=$TOKEN_BUDGET)" | tee -a "${LOG_FILE}"
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/$MODEL_VERSION/$MODEL_SCALE/$task.sh $METHOD $TOKEN_BUDGET 2>&1 | tee -a "${LOG_FILE}"
    done

    {
        echo ""
        echo ">>> ✓ Finished Pruning Schedule: $name"
        echo ""
    } | tee -a "${LOG_FILE}"
done

{
    echo "=========================================="
    echo "Stage 2 Pruning Schedule Ablation Completed!"
    echo "=========================================="
    echo ""
    echo "Full log saved to: ${LOG_FILE}"
} | tee -a "${LOG_FILE}"
