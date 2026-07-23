#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
LOG_ROOT="$ROOT/results_aihc_qr_recover_fast_train21_debug_20260723"
TAG=llavaqr_recover_a3_128_fast_train21_debug
OUT="$LOG_ROOT/$TAG"

mkdir -p "$OUT"

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate llava

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
export DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"
export ENABLE_DEBUG=1
export STAR_KEEP_POSIDS=1
export STAGE1_SCORER=qr
export STAGE1_MULT=3
export METHOD=star_pro
export TOKEN=128
export RUN_TAG="$TAG"
export RESULT_DIR="$OUT"

cd "$ROOT"

{
  echo "CONFIG tag=$TAG METHOD=$METHOD TOKEN=$TOKEN STAGE1_SCORER=$STAGE1_SCORER STAGE1_MULT=$STAGE1_MULT STAR_KEEP_POSIDS=$STAR_KEEP_POSIDS RUN_TAG=$RUN_TAG"
  for bench in gqa textvqa; do
    echo "===== BEGIN $TAG $bench ====="
    bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$OUT/${bench}.fulllog"
    echo "===== END $TAG $bench ====="
  done
} 2>&1 | tee "$OUT/suite.log"

python3 - <<'PY'
import os
import re
import sys

root = "/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723/results_aihc_qr_recover_fast_train21_debug_20260723/llavaqr_recover_a3_128_fast_train21_debug"
patterns = {
    "gqa": re.compile(r"Accuracy:\s*([\d.]+)%"),
    "textvqa": re.compile(r"Accuracy:\s*([\d.]+)%"),
}
bad_words = ("Traceback", "RuntimeError", "CUDA out of memory", "ERROR", "WARNING: scorer=qr failed")
failed = False
for bench, pat in patterns.items():
    path = os.path.join(root, f"{bench}.fulllog")
    if not os.path.exists(path):
        print(f"{bench}: MISSING_LOG")
        failed = True
        continue
    text = open(path, errors="replace").read()
    s1 = text.count("[STAR-PRO S1 SCORER] scorer=qr")
    if any(w in text for w in bad_words):
        print(f"{bench}: ERROR_MARKER s1={s1}")
        failed = True
        continue
    matches = pat.findall(text)
    if not matches:
        print(f"{bench}: MISSING_SCORE s1={s1}")
        failed = True
        continue
    print(f"{bench}: {float(matches[-1]):.4f} s1={s1}")
if failed:
    sys.exit(80)
PY

echo "ALL_DONE llava_qr_recover_fast_train21_debug_20260723"
