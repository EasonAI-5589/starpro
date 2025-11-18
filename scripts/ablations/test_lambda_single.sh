#!/bin/bash
################################################################################
# Quick Test: Single Lambda Value (λ=0.5)
################################################################################
# Purpose: Verify the λ parameter implementation before running full ablation
################################################################################

set -e

METHOD="star_v3"
TOKEN=128
LAMBDA=0.5

echo "================================================================================"
echo "Quick Test: LAMBDA=${LAMBDA}"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Method: ${METHOD}"
echo "  Token budget: ${TOKEN}"
echo "  Lambda (λ): ${LAMBDA}"
echo "  Formula: L_i(S) = (1-λ)R_i + λD_i(S)"
echo "  Relevance weight (1-λ): 0.5"
echo "  Diversity weight (λ): 0.5"
echo ""
echo "================================================================================"
echo ""

# Export lambda
export LAMBDA=${LAMBDA}

# Run POPE benchmark
echo "Running POPE benchmark..."
bash scripts/v1_6/7b/pope.sh ${METHOD} ${TOKEN}

echo ""
echo "================================================================================"
echo "Test completed!"
echo "================================================================================"
echo ""
echo "Check the debug output for Lambda Configuration section to verify:"
echo "  - λ (lambda): 0.5"
echo "  - Relevance weight (1-λ): 0.5"
echo "  - Diversity weight (λ): 0.5"
echo ""
echo "Result log: ./results/pope_llava-v1.6-vicuna-7b_thcp_vtn128.log"
echo ""
