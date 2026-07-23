#!/bin/bash
# Appendix: Complementarity (1-cos) vs Similarity (cos) ablation
# 比较 NEGATE_RELEVANCE=1 (互补) vs NEGATE_RELEVANCE=0 (相似) 在 GQA/POPE/TextVQA 上的表现

LOG=/mnt/eason/LLaVA-STAR-Pro/results/sign_ablation_$(date +%Y%m%d_%H%M%S).log
TASKS="gqa pope textvqa"
TOKENS="128 64 32"

echo "=== Sign Ablation: 互补性 vs 相似性 ===" | tee $LOG
echo "Benchmarks: $TASKS | Tokens: $TOKENS" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG

for TASK in $TASKS; do
  for T in $TOKENS; do
    for NEGATE in 1 0; do
      SIGN_NAME=$( [ $NEGATE -eq 1 ] && echo "complementary" || echo "similarity" )
      echo "" | tee -a $LOG
      echo "▶ [${TASK^^}] T=$T | NEGATE=$NEGATE ($SIGN_NAME)" | tee -a $LOG
      
      NEGATE_RELEVANCE=$NEGATE \
      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      bash scripts/eval_task.sh $TASK star_pro $T 2>&1 | tee -a $LOG
      
      echo "  Finished: $(date)" | tee -a $LOG
    done
  done
done

echo "" | tee -a $LOG
echo "=== ALL DONE ===" | tee -a $LOG
