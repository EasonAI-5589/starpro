#!/bin/bash
# STAR-V5 (S2 Only) Ablation Study
# Purpose: Test Stage 2 progressive pruning only, without Stage 1 THCP
# Usage: bash scripts/ablations/run_s2_only.sh

# ========== Configuration ==========
METHOD="star_v5"  # S2 only ablation
TOKEN=128         # Target token budget
GPU_LIST="0,1,2,3,4,5,6,7"  # Available GPUs

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.6-vicuna-7b"

# ========== Benchmarks to run ==========
BENCHMARKS=("pope" "mme" "textvqa")

echo "=========================================="
echo "STAR-V5 (S2 Only) Ablation Study"
echo "=========================================="
echo "Method: ${METHOD}"
echo "Token Budget: ${TOKEN}"
echo "Benchmarks: ${BENCHMARKS[@]}"
echo "GPUs: ${GPU_LIST}"
echo "=========================================="
echo ""

# ========== Run experiments ==========
for BENCH in "${BENCHMARKS[@]}"; do
    echo ">>> Running ${METHOD} on ${BENCH} benchmark..."
    echo ""

    # Check if benchmark script exists
    SCRIPT_PATH="scripts/v1_6/7b/${BENCH}.sh"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "ERROR: Script not found: $SCRIPT_PATH"
        echo "Skipping ${BENCH}..."
        continue
    fi

    # Run benchmark
    CUDA_VISIBLE_DEVICES=${GPU_LIST} bash ${SCRIPT_PATH} ${METHOD} ${TOKEN}

    if [ $? -eq 0 ]; then
        echo "✓ ${BENCH} completed successfully"
    else
        echo "✗ ${BENCH} failed"
    fi
    echo ""
    echo "=========================================="
    echo ""
done

echo ">>> All experiments completed!"
echo ""
echo "Results should be in:"
echo "  POPE: ./playground/data/eval/pope/answers/"
echo "  MME:  ./playground/data/eval/MME/answers/"
echo "  TextVQA: ./playground/data/eval/textvqa/answers/"
echo ""
echo "To collect results, run the evaluation scripts for each benchmark."
