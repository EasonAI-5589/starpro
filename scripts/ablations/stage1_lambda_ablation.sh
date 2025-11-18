#!/bin/bash
################################################################################
# Stage 1 (THCP) Lambda Ablation Study
################################################################################
#
# Purpose: Test different λ values in the THCP formula to isolate Stage 1 contribution
# Formula: L_i(S) = (1-λ)R_i + λD_i(S)
#
# Lambda values:
#   λ=0.0  → Pure relevance (1.0R + 0.0D)
#   λ=0.25 → Relevance-heavy (0.75R + 0.25D)
#   λ=0.5  → Balanced (0.5R + 0.5D)
#   λ=0.75 → Diversity-heavy (0.25R + 0.75D)
#   λ=1.0  → Pure diversity (0.0R + 1.0D)
#
# Configuration:
#   Method: thcp (Stage 1 only)
#   Token budget: T=128
#   Benchmark: POPE (fast validation)
#   Model: llava-v1.6-vicuna-7b
#
################################################################################

set -e  # Exit on error

METHOD="thcp"
TOKEN=128

echo "================================================================================"
echo "Stage 1 (THCP) Lambda Ablation Study"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Method: ${METHOD}"
echo "  Token budget: ${TOKEN}"
echo "  Benchmark: POPE"
echo "  Formula: L_i(S) = (1-λ)R_i + λD_i(S)"
echo ""
echo "Lambda values to test: 0.0, 0.25, 0.5, 0.75, 1.0"
echo ""
echo "================================================================================"
echo ""

# Results directory
RESULT_DIR="./results/ablation_stage1_lambda"
mkdir -p ${RESULT_DIR}

# Log file for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${RESULT_DIR}/ablation_summary_${TIMESTAMP}.log"

echo "Master log: ${MASTER_LOG}"
echo ""

# Lambda values to test
LAMBDA_VALUES=(0.0 0.25 0.5 0.75 1.0)

for lambda in "${LAMBDA_VALUES[@]}"; do
    echo "================================================================================"
    echo "Testing LAMBDA=${lambda}"
    echo "================================================================================"
    echo ""
    echo "  Relevance weight (1-λ): $(echo "1 - ${lambda}" | bc -l)"
    echo "  Diversity weight (λ):   ${lambda}"
    echo ""

    # Export lambda for this run
    export LAMBDA=${lambda}

    # Run POPE benchmark
    echo "  Running POPE benchmark..."
    bash scripts/v1_6/7b/pope.sh ${METHOD} ${TOKEN}

    # Copy result log to ablation directory
    POPE_LOG="./results/pope_llava-v1.6-vicuna-7b_${METHOD}_vtn${TOKEN}.log"
    if [ -f "${POPE_LOG}" ]; then
        LAMBDA_LOG="${RESULT_DIR}/lambda_${lambda}_pope.log"
        cp ${POPE_LOG} ${LAMBDA_LOG}
        echo "  Result saved to: ${LAMBDA_LOG}"

        # Extract F1 score and append to master log
        F1_SCORE=$(grep "Average F1" ${POPE_LOG} | tail -1)
        echo "LAMBDA=${lambda}: ${F1_SCORE}" >> ${MASTER_LOG}
        echo "  ${F1_SCORE}"
    else
        echo "  Warning: Result log not found at ${POPE_LOG}"
    fi

    echo ""
    echo "  Completed LAMBDA=${lambda}"
    echo ""
    sleep 2  # Brief pause between runs
done

echo "================================================================================"
echo "Stage 1 Lambda Ablation Completed"
echo "================================================================================"
echo ""
echo "Summary saved to: ${MASTER_LOG}"
echo ""
cat ${MASTER_LOG}
echo ""
echo "Individual results in: ${RESULT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Analyze results in ${RESULT_DIR}/"
echo "  2. Create documentation file for ablation results"
echo "  3. Update ABLATION_STUDY.md with findings"
echo ""
echo "================================================================================"
