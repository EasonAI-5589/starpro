#!/usr/bin/env bash
# AI2D native-LLaVA evaluation for STAR-Pro (reportable path).
#
# Copy prepare_ai2d.py + score_ai2d.py to the SERVER sidecar dir first, e.g.
#   $SIDE = /mnt/eason/starpro_llava_eval_sidecar
# then source this file's blocks (do NOT run vanilla + a pruned job at once:
# each full run occupies all 8 GPUs). GPUs are currently busy -> queue.
set -euo pipefail

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate llava        # canonical for star_pro; 'fastv' is the same known-good path (SEED=66.03)

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
SIDE=/mnt/eason/starpro_llava_eval_sidecar
MODEL=/mnt/eason_ckp/models/llava-v1.5-7b
SRC=/mnt/eason_ckp/LMUData/AI2D_TEST.tsv       # already present (3088 rows)
export PYTHONPATH="$ROOT"

# ---- GPU preflight: refuse to launch a full run while any card is busy -----
gpu_preflight() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '$1 > 300 {n++} END {print n+0}')
  if [ "$busy" -ne 0 ]; then
    echo "QUEUE: a GPU is in use. Do not launch; check again later."; return 75
  fi
}

# ---- shared 8-GPU runner: run_full <OUT> <METHOD> <TOKENS> -----------------
# vanilla: method=vanilla tokens=576 ; star_pro-qr: method=star_pro tokens=128/64/32
run_full() {
  local OUT="$1" METHOD="$2" TOKENS="$3"
  test ! -e "$OUT" || { echo "Refusing to overwrite: $OUT"; return 1; }
  mkdir -p "$OUT/answers"
  python "$SIDE/prepare_ai2d.py" --source "$SRC" --output-dir "$OUT"   # all 3088 rows

  # STAR-Pro-QR env vars ONLY for the star_pro method (never for vanilla).
  local -a QR_ENV=()
  if [ "$METHOD" = "star_pro" ]; then
    QR_ENV=(STAGE1_SCORER=qr STAGE1_MULT=2 STAR_CAUSAL_FIX=1 STAR_KEEP_POSIDS=1 \
            TEXT_AGG_MODE=average_all ENABLE_DEBUG=1)
  fi

  for IDX in $(seq 0 7); do
    env "${QR_ENV[@]}" CUDA_VISIBLE_DEVICES=$IDX PYTHONPATH="$ROOT" \
      python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL" \
        --question-file "$OUT/questions.jsonl" \
        --image-folder "$OUT/images" \
        --answers-file "$OUT/answers/8_$IDX.jsonl" \
        --num-chunks 8 --chunk-idx "$IDX" \
        --pruning_method "$METHOD" \
        --visual_token_num "$TOKENS" \
        --temperature 0 --conv-mode vicuna_v1 --max_new_tokens 128 \
        > "$OUT/worker_$IDX.log" 2>&1 &
  done
  wait

  cat "$OUT"/answers/8_{0,1,2,3,4,5,6,7}.jsonl > "$OUT/answers_merge.jsonl"
  wc -l "$OUT/questions.jsonl" "$OUT/answers_merge.jsonl"
  grep -nE "Traceback|RuntimeError|CUDA out of memory|FileNotFoundError" "$OUT"/worker_*.log || true

  python "$SIDE/score_ai2d.py" \
    --metadata "$OUT/metadata.jsonl" \
    --predictions "$OUT/answers_merge.jsonl" \
    --output "$OUT/score.json"
  cat "$OUT/score.json" | python -c 'import json,sys;d=json.load(sys.stdin);print("Overall %",d["Overall_pct"],"n",d["samples"])'
}

BASE=/mnt/eason_ckp/starpro_eval/llava_native

# =============================== SMOKE (1 GPU, vanilla, tiny) ===============
# Sanity only; never report this number.
smoke() {
  local OUT="$BASE/ai2d_smoke_vanilla_576"
  test ! -e "$OUT" || { echo "Refusing to overwrite: $OUT"; return 1; }
  python "$SIDE/prepare_ai2d.py" --source "$SRC" --output-dir "$OUT" --limit 8
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL" \
    --question-file "$OUT/questions.jsonl" \
    --image-folder "$OUT/images" \
    --answers-file "$OUT/answers.jsonl" \
    --pruning_method vanilla --visual_token_num 576 \
    --temperature 0 --conv-mode vicuna_v1 --max_new_tokens 128
  python "$SIDE/score_ai2d.py" --metadata "$OUT/metadata.jsonl" \
    --predictions "$OUT/answers.jsonl" --output "$OUT/score.json"
}

# =============================== FULL RUNS =================================
# Run ONE at a time, each after gpu_preflight passes.
full_vanilla()  { gpu_preflight && run_full "$BASE/ai2d_full_vanilla_576_$(date +%Y%m%d)" vanilla  576; }
full_qr_128()   { gpu_preflight && run_full "$BASE/ai2d_full_starproqr_128_$(date +%Y%m%d)" star_pro 128; }
full_qr_64()    { gpu_preflight && run_full "$BASE/ai2d_full_starproqr_64_$(date +%Y%m%d)"  star_pro 64;  }
full_qr_32()    { gpu_preflight && run_full "$BASE/ai2d_full_starproqr_32_$(date +%Y%m%d)"  star_pro 32;  }

# Dispatch: ./run_ai2d.sh {smoke|full_vanilla|full_qr_128|full_qr_64|full_qr_32}
"${1:-smoke}"
