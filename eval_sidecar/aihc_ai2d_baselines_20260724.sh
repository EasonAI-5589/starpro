#!/usr/bin/env bash
# ============================================================================
# AI2D baselines on LLaVA-1.5-7B  —  8-GPU distributed eval, serial across runs
# Methods: FastV, PDrop, SparseVLM, VisionZip, DivPrune, CDPruner, SCOPE,
#          VScan(2-stage), HoloV  x  {128,64,32} token budgets.
# DUET-VLM and STAR-Pro/-QR are OUT OF SCOPE (do not run here).
# conda env: fastv (per corrected instruction / OCRBENCH_HANDOFF.md).
# Follows aihc_ocrbench_baselines_20260724.sh run_one() pattern exactly:
# 8-way CUDA_VISIBLE_DEVICES parallel chunks, merge, integrity gate, score.
# ============================================================================
set -uo pipefail

echo "AI2D_BASELINES_START $(date -Iseconds) host=$(hostname)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate fastv || { echo "FATAL: cannot activate fastv env"; exit 3; }
echo "python=$(which python)  ($(python --version 2>&1))"

export ENABLE_DEBUG=1

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
SIDECAR=/mnt/eason/starpro_llava_eval_sidecar
MODEL=/mnt/eason_ckp/models/llava-v1.5-7b
SRC=/mnt/eason_ckp/starpro_eval/llava_native/_ai2d_baselines_src_20260724
OUTBASE=/mnt/eason_ckp/starpro_eval/llava_native
STAMP=$(date +%Y%m%d_%H%M)
export PYTHONPATH="$ROOT"

for f in images questions.jsonl metadata.jsonl; do
  [ -e "$SRC/$f" ] || { echo "FATAL: SRC missing $f"; exit 4; }
done
QN=$(wc -l < "$SRC/questions.jsonl")
echo "SRC questions=$QN"
[ "$QN" -eq 3088 ] || { echo "FATAL: SRC questions != 3088 ($QN)"; exit 5; }

# ---- GPU preflight ----------------------------------------------------
gpu_preflight() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '$1 > 300 {n++} END {print n+0}')
  if [ "$busy" -ne 0 ]; then
    echo "REFUSE: a GPU is busy (busy_count=$busy). Not launching."; return 1
  fi
  return 0
}

# ---- one full 8-GPU AI2D run + completion-gate + score ----------------
run_one () {
  local METHOD="$1"; local TOKENS="$2"; local TAG="$3"; shift 3
  local EXTRA_ARGS=("$@")
  local OUT="$OUTBASE/ai2d_full_baseline_${TAG}"
  echo "==================== RUN ${TAG}  (method=${METHOD} tokens=${TOKENS} extra='${EXTRA_ARGS[*]}') ===================="
  if [ -e "$OUT" ]; then echo "REFUSE_OVERWRITE_SKIP $OUT"; return 0; fi

  if ! gpu_preflight; then
    echo "ABORT_RUN ${TAG}: GPU busy at launch time"; return 1
  fi

  mkdir -p "$OUT/answers"
  ln -s "$SRC/images" "$OUT/images"
  cp "$SRC/questions.jsonl" "$OUT/questions.jsonl"
  cp "$SRC/metadata.jsonl"  "$OUT/metadata.jsonl"

  for IDX in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$IDX PYTHONPATH="$ROOT" ENABLE_DEBUG=1 \
      python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL" \
        --question-file "$OUT/questions.jsonl" \
        --image-folder "$OUT/images" \
        --answers-file "$OUT/answers/8_$IDX.jsonl" \
        --num-chunks 8 --chunk-idx "$IDX" \
        --pruning_method "$METHOD" \
        --visual_token_num "$TOKENS" \
        "${EXTRA_ARGS[@]}" \
        --temperature 0 --conv-mode vicuna_v1 --max_new_tokens 128 \
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

  if [ "$AQ" -ne 3088 ] || [ "$AM" -ne 3088 ]; then
    echo "INCOMPLETE ${TAG} (q=$AQ a=$AM) — NOT scoring, gate FAILED"
    return 1
  fi

  python "$SIDECAR/score_ai2d.py" \
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

# ---- fail-fast smoke (2 samples, method=fastv) to validate env+mounts+model
echo "#### SMOKE (2 samples, method=fastv tokens=128) ####"
SMOKE="$OUTBASE/ai2d_baselines_smoke_${STAMP}"
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
head -2 "$SRC/questions.jsonl" > "$SMOKE/q2.jsonl"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" ENABLE_DEBUG=1 python -m llava.eval.model_vqa_loader \
  --model-path "$MODEL" --question-file "$SMOKE/q2.jsonl" --image-folder "$SRC/images" \
  --answers-file "$SMOKE/a.jsonl" --num-chunks 1 --chunk-idx 0 \
  --pruning_method fastv --visual_token_num 128 --temperature 0 --conv-mode vicuna_v1 --max_new_tokens 128 \
  > "$SMOKE/smoke.log" 2>&1
SC=$(wc -l < "$SMOKE/a.jsonl" 2>/dev/null || echo 0)
if [ "${SC:-0}" -lt 2 ]; then
  echo "SMOKE_FAILED (got $SC answers) — env/model/data broken, aborting before full runs:"
  tail -60 "$SMOKE/smoke.log"
  exit 10
fi
echo "SMOKE_OK ($SC answers)"

# ============================= FULL RUNS ==================================
# Order: 128 tier all methods, then 64 tier, then 32 tier (VisionZip/DivPrune/
# CDPruner/SCOPE/VScan/HoloV only for 32 — no FastV/PDrop/SparseVLM row at 32).

# ---- 128 ----
run_one fastv      128 "fastv_128_${STAMP}"
run_one pdrop      128 "pdrop_128_${STAMP}"
run_one sparsevlm  128 "sparsevlm_128_${STAMP}"
run_one visionzip  128 "visionzip_128_${STAMP}"
run_one divprune   128 "divprune_128_${STAMP}"
run_one cdp3       128 "cdpruner_128_${STAMP}"
run_one scope      128 "scope_128_${STAMP}"
run_one vscan      224 "vscan_avg128_${STAMP}" --vscan_stage2_tokens 32 --vscan_prune_layer 16
run_one HoloV_2    128 "holov_128_${STAMP}"

# ---- 64 ----
run_one fastv      64  "fastv_64_${STAMP}"
run_one pdrop      64  "pdrop_64_${STAMP}"
run_one sparsevlm  64  "sparsevlm_64_${STAMP}"
run_one visionzip  64  "visionzip_64_${STAMP}"
run_one divprune   64  "divprune_64_${STAMP}"
run_one cdp3       64  "cdpruner_64_${STAMP}"
run_one scope      64  "scope_64_${STAMP}"
run_one vscan      96  "vscan_avg64_${STAMP}" --vscan_stage2_tokens 32 --vscan_prune_layer 16
run_one HoloV_2    64  "holov_64_${STAMP}"

# ---- 32 ----
run_one visionzip  32  "visionzip_32_${STAMP}"
run_one divprune   32  "divprune_32_${STAMP}"
run_one cdp3       32  "cdpruner_32_${STAMP}"
run_one scope      32  "scope_32_${STAMP}"
run_one vscan      32  "vscan_avg32_${STAMP}" --vscan_stage2_tokens 32 --vscan_prune_layer 16
run_one HoloV_2    32  "holov_32_${STAMP}"

echo "======================== SUMMARY (${STAMP}) ========================"
for d in "$OUTBASE"/ai2d_full_baseline_*_"${STAMP}"; do
  if [ -f "$d/score.json" ]; then
    echo "--- $(basename "$d") ---"; cat "$d/score.json"; echo
  else
    echo "--- $(basename "$d") : NO score.json (check logs) ---"
  fi
done
echo "AI2D_BASELINES_ALL_DONE $(date -Iseconds)"
