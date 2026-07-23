#!/usr/bin/env bash
# ============================================================================
# OCRBench baselines on LLaVA-1.5-7B  —  AIHC 8-GPU eval job
# Author: experiment agent (Claude), 2026-07-24.  Self-identifying wrapper.
# Follows /mnt/eason/starpro_llava_eval_sidecar/OCRBENCH_HANDOFF.md EXACTLY.
# Runs ONLY handoff-verified methods (none/fastv/vscan/HoloV_2).
# Non-STAR baselines: deliberately does NOT export any STAR-Pro-only var
# (STAGE1_SCORER / STAGE1_MULT / STAR_KEEP_POSIDS / STAR_CAUSAL_FIX / TEXT_AGG_MODE).
# NOTE: does NOT write Feishu (handoff rule). Only produces score.json per run.
# ============================================================================
set -uo pipefail

echo "OCRBENCH_BASELINES_START $(date -Iseconds) host=$(hostname)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate fastv || { echo "FATAL: cannot activate fastv env"; exit 3; }
echo "python=$(which python)  ($(python --version 2>&1))"

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
SIDECAR=/mnt/eason/starpro_llava_eval_sidecar
MODEL=/mnt/eason_ckp/models/llava-v1.5-7b
SRC=/mnt/eason_ckp/starpro_eval/llava_native/ocrbench_full_fastv_starproqr_128_20260724
OUTBASE=/mnt/eason_ckp/starpro_eval/llava_native
STAMP=$(date +%Y%m%d_%H%M)
export PYTHONPATH="$ROOT"

# ---- verify reusable materialization (read-only source) ----
for f in images questions.jsonl metadata.jsonl; do
  [ -e "$SRC/$f" ] || { echo "FATAL: SRC missing $f"; exit 4; }
done
QN=$(wc -l < "$SRC/questions.jsonl")
echo "SRC questions=$QN"
[ "$QN" -eq 1000 ] || { echo "FATAL: SRC questions != 1000 ($QN)"; exit 5; }

# ---- one full 8-GPU OCRBench run + completion-gate + score ----
run_one () {
  local METHOD="$1"; local TOKENS="$2"; local TAG="$3"; shift 3
  local EXTRA_ARGS="$*"
  local OUT="$OUTBASE/ocrbench_full_fastv_${TAG}"
  echo "==================== RUN ${TAG}  (method=${METHOD} tokens=${TOKENS} extra='${EXTRA_ARGS}') ===================="
  if [ -e "$OUT" ]; then echo "REFUSE_OVERWRITE_SKIP $OUT"; return 0; fi
  mkdir -p "$OUT/answers"
  ln -s "$SRC/images" "$OUT/images"
  cp "$SRC/questions.jsonl" "$OUT/questions.jsonl"
  cp "$SRC/metadata.jsonl"  "$OUT/metadata.jsonl"

  for IDX in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$IDX PYTHONPATH="$ROOT" \
      python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL" \
        --question-file "$OUT/questions.jsonl" \
        --image-folder "$OUT/images" \
        --answers-file "$OUT/answers/8_$IDX.jsonl" \
        --num-chunks 8 --chunk-idx "$IDX" \
        --pruning_method "$METHOD" \
        --visual_token_num "$TOKENS" \
        $EXTRA_ARGS \
        --temperature 0 --conv-mode vicuna_v1 \
        > "$OUT/worker_$IDX.log" 2>&1 &
  done
  wait

  cat "$OUT"/answers/8_{0,1,2,3,4,5,6,7}.jsonl > "$OUT/answers_merge.jsonl"
  local AQ AM
  AQ=$(wc -l < "$OUT/questions.jsonl"); AM=$(wc -l < "$OUT/answers_merge.jsonl")
  echo "LINES ${TAG}: questions=$AQ answers=$AM"

  if grep -nE "Traceback|RuntimeError|CUDA out of memory|FileNotFoundError" "$OUT"/worker_*.log ; then
    echo "WORKER_ERRORS_FOUND ${TAG}"
  else
    echo "WORKER_LOGS_CLEAN ${TAG}"
  fi

  if [ "$AQ" -ne 1000 ] || [ "$AM" -ne 1000 ]; then
    echo "INCOMPLETE ${TAG} (q=$AQ a=$AM) — NOT scoring, gate FAILED"
    return 1
  fi

  python "$SIDECAR/score_ocrbench.py" \
    --metadata "$OUT/metadata.jsonl" \
    --predictions "$OUT/answers_merge.jsonl" \
    --output "$OUT/score.json"

  if [ -s "$OUT/score.json" ]; then
    touch "$OUT/FULL_DONE"
    echo "SCORE_JSON ${TAG}:"; cat "$OUT/score.json"; echo
    echo "FULL_DONE ${TAG}  OUT=$OUT"
  else
    echo "SCORE_FAILED ${TAG}"
    return 1
  fi
}

# ---- fail-fast smoke (2 samples, method=none) to validate env+mounts+model ----
echo "#### SMOKE (2 samples, method=none) ####"
SMOKE="$OUTBASE/ocrbench_smoke_${STAMP}"
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
head -2 "$SRC/questions.jsonl" > "$SMOKE/q2.jsonl"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" python -m llava.eval.model_vqa_loader \
  --model-path "$MODEL" --question-file "$SMOKE/q2.jsonl" --image-folder "$SRC/images" \
  --answers-file "$SMOKE/a.jsonl" --num-chunks 1 --chunk-idx 0 \
  --pruning_method none --visual_token_num 576 --temperature 0 --conv-mode vicuna_v1 \
  > "$SMOKE/smoke.log" 2>&1
SC=$(wc -l < "$SMOKE/a.jsonl" 2>/dev/null || echo 0)
if [ "${SC:-0}" -lt 2 ]; then
  echo "SMOKE_FAILED (got $SC answers) — env/model/data broken, aborting before full runs:"
  tail -40 "$SMOKE/smoke.log"
  exit 10
fi
echo "SMOKE_OK ($SC answers)"

# ---- full runs: handoff-verified methods only ----
run_one none    576 "vanilla_576_${STAMP}"
run_one fastv   128 "fastv_128_${STAMP}"
run_one fastv    64 "fastv_64_${STAMP}"
run_one fastv    32 "fastv_32_${STAMP}"
run_one HoloV_2 128 "holov2_128_${STAMP}"
run_one HoloV_2  64 "holov2_64_${STAMP}"
run_one HoloV_2  32 "holov2_32_${STAMP}"
run_one vscan   128 "vscan_128to32_${STAMP}" --vscan_stage2_tokens 32 --vscan_prune_layer 16

echo "======================== SUMMARY (${STAMP}) ========================"
for d in "$OUTBASE"/ocrbench_full_fastv_*_"${STAMP}"; do
  if [ -f "$d/score.json" ]; then
    echo "--- $(basename "$d") ---"; cat "$d/score.json"; echo
  else
    echo "--- $(basename "$d") : NO score.json (check logs) ---"
  fi
done
echo "OCRBENCH_BASELINES_ALL_DONE $(date -Iseconds)"
