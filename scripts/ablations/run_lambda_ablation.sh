#!/bin/bash
################################################################################
# Stage 1 (THCP) Lambda Ablation - Full Study
################################################################################
# Purpose: Test all λ values across multiple benchmarks
# Formula: L_i(S) = (1-λ)R_i + λD_i(S)
#
# Lambda values: 0.0, 0.25, 0.5, 0.75, 1.0
# Benchmarks: MME, TextVQA, GQA, POPE
# Model: llava-v1.5-7b
# Method: star_v3 (S1+S2 Full STAR-Pro)
# Token budget: T=128
################################################################################

set -e  # Exit on error

# ==================== Configuration ====================
METHOD="star_v3"
TOKEN_BUDGET=128

# Lambda values to test
LAMBDA_VALUES=(0.0 0.25 0.5 0.75 1.0)

# Evaluation tasks
TASKS=(mme textvqa gqa pope)

# GPU devices
GPUS="0,1,2,3,4,5,6,7"

# Results directory
RESULT_DIR="./results/ablation_stage1_lambda"
mkdir -p ${RESULT_DIR}

# Master log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${RESULT_DIR}/ablation_summary_${TIMESTAMP}.log"

echo "================================================================================"
echo "Stage 1 (THCP) Lambda Ablation - Full Study"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Model: llava-v1.5-7b"
echo "  Method: ${METHOD}"
echo "  Token budget: ${TOKEN_BUDGET}"
echo "  Formula: L_i(S) = (1-λ)R_i + λD_i(S)"
echo ""
echo "Lambda values: ${LAMBDA_VALUES[@]}"
echo "Benchmarks: ${TASKS[@]}"
echo ""
echo "Results directory: ${RESULT_DIR}"
echo "Master log: ${MASTER_LOG}"
echo ""
echo "================================================================================"
echo ""

# Initialize master log
echo "================================================================================" > ${MASTER_LOG}
echo "Stage 1 (THCP) Lambda Ablation - Full Study" >> ${MASTER_LOG}
echo "================================================================================" >> ${MASTER_LOG}
echo "" >> ${MASTER_LOG}
echo "Timestamp: $(date)" >> ${MASTER_LOG}
echo "Model: llava-v1.5-7b" >> ${MASTER_LOG}
echo "Method: ${METHOD}" >> ${MASTER_LOG}
echo "Token budget: ${TOKEN_BUDGET}" >> ${MASTER_LOG}
echo "Formula: L_i(S) = (1-λ)R_i + λD_i(S)" >> ${MASTER_LOG}
echo "" >> ${MASTER_LOG}
echo "================================================================================" >> ${MASTER_LOG}
echo "" >> ${MASTER_LOG}

# ==================== Run Experiments ====================
for lambda in "${LAMBDA_VALUES[@]}"; do
    relevance=$(awk "BEGIN {print 1-$lambda}")

    echo "================================================================================"
    echo "Testing LAMBDA=${lambda}"
    echo "================================================================================"
    echo ""
    echo "  Formula: L_i(S) = (1-λ)R_i + λD_i(S)"
    echo "  Relevance weight (1-λ): ${relevance}"
    echo "  Diversity weight (λ): ${lambda}"
    echo ""

    # Export lambda for this run
    export LAMBDA=${lambda}

    # Log lambda test start
    echo "LAMBDA=${lambda} (Relevance=${relevance}, Diversity=${lambda})" >> ${MASTER_LOG}
    echo "---------------------------------------------------" >> ${MASTER_LOG}

    # Run all benchmarks for this lambda
    for task in "${TASKS[@]}"; do
        echo "  >>> Running: ${task} with ${METHOD} (lambda=${lambda}, budget=${TOKEN_BUDGET})"

        # Run the ablation-specific benchmark script
        CUDA_VISIBLE_DEVICES=${GPUS} bash scripts/ablations/${task}_ablation.sh ${METHOD} ${TOKEN_BUDGET}

        # Check result log (already saved by ablation script)
        LAMBDA_TASK_LOG="${RESULT_DIR}/lambda_${lambda}_${task}_v1_5_7b_${METHOD}_vtn${TOKEN_BUDGET}.log"
        if [ -f "${LAMBDA_TASK_LOG}" ]; then
            echo "    ✓ Result saved to: ${LAMBDA_TASK_LOG}"

            # Extract key metrics and append to master log
            case ${task} in
                mme)
                    SCORE=$(grep "Total Score:" ${LAMBDA_TASK_LOG} | tail -1 || echo "N/A")
                    echo "  ${task}: ${SCORE}" >> ${MASTER_LOG}
                    echo "    ${SCORE}"
                    ;;
                pope)
                    ACC=$(grep "Accuracy:" ${LAMBDA_TASK_LOG} | tail -1 || echo "N/A")
                    echo "  ${task}: ${ACC}" >> ${MASTER_LOG}
                    echo "    ${ACC}"
                    ;;
                textvqa)
                    ACC=$(grep "Accuracy:" ${LAMBDA_TASK_LOG} | tail -1 || echo "N/A")
                    echo "  ${task}: ${ACC}" >> ${MASTER_LOG}
                    echo "    ${ACC}"
                    ;;
                gqa)
                    ACC=$(grep "Accuracy:" ${LAMBDA_TASK_LOG} | tail -1 || echo "N/A")
                    echo "  ${task}: ${ACC}" >> ${MASTER_LOG}
                    echo "    ${ACC}"
                    ;;
            esac
        else
            echo "    ✗ Warning: Result log not found at ${LAMBDA_TASK_LOG}"
            echo "  ${task}: NOT FOUND" >> ${MASTER_LOG}
        fi

        echo ""
    done

    echo "" >> ${MASTER_LOG}
    echo "  Completed LAMBDA=${lambda}"
    echo ""
    sleep 2  # Brief pause between lambda values
done

echo "================================================================================"
echo "Stage 1 Lambda Ablation Completed!"
echo "================================================================================"
echo ""
echo "Summary saved to: ${MASTER_LOG}"
echo ""
echo "Individual results in: ${RESULT_DIR}/"
ls -lh ${RESULT_DIR}/lambda_*.log | tail -20
echo ""
echo "================================================================================"
echo ""
echo "Results Summary:"
echo "================================================================================"
cat ${MASTER_LOG}
echo ""
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Analyze results in ${RESULT_DIR}/"
echo "  2. Update STAGE1_LAMBDA_ABLATION.md with findings"
echo "  3. Create visualization comparing lambda values"
echo "  4. Update paper ablation table with results"
echo ""
