#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
LOG_ROOT="$ROOT/results_aihc_qr_recover_textpope_train2_debug_20260723"
mkdir -p "$LOG_ROOT"

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
export METHOD=star_pro
export TOKEN=128

cd "$ROOT"

run_one() {
  local tag="$1"
  local mult="$2"
  local out="$LOG_ROOT/$tag"
  mkdir -p "$out"
  export STAGE1_MULT="$mult"
  export RUN_TAG="$tag"
  export RESULT_DIR="$out"
  {
    echo "CONFIG tag=$tag METHOD=$METHOD TOKEN=$TOKEN STAGE1_SCORER=$STAGE1_SCORER STAGE1_MULT=$STAGE1_MULT STAR_KEEP_POSIDS=$STAR_KEEP_POSIDS RUN_TAG=$RUN_TAG"
    for bench in textvqa pope; do
      echo "===== BEGIN $tag $bench ====="
      bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$out/${bench}.fulllog"
      echo "===== END $tag $bench ====="
    done
  } 2>&1 | tee "$out/suite.log"
}

run_one "llavaqr_recover_a2_128_textpope_debug" "2"
run_one "llavaqr_recover_a3_128_textpope_debug" "3"

python3 - <<'PY'
import os
import re
import sys

root = "/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723/results_aihc_qr_recover_textpope_train2_debug_20260723"
tags = ["llavaqr_recover_a2_128_textpope_debug", "llavaqr_recover_a3_128_textpope_debug"]
patterns = {
    "textvqa": re.compile(r"Accuracy:\s*([\d.]+)%"),
    "pope": re.compile(r"Average F1 score:\s*([\d.]+)"),
}
bad_words = ("Traceback", "RuntimeError", "CUDA out of memory", "ERROR", "WARNING: scorer=qr failed", "FileNotFoundError")
failed = False
for tag in tags:
    print(f"SUMMARY {tag}")
    for bench, pat in patterns.items():
        path = os.path.join(root, tag, f"{bench}.fulllog")
        if not os.path.exists(path):
            print(f"  {bench}: MISSING_LOG")
            failed = True
            continue
        text = open(path, errors="replace").read()
        s1 = text.count("[STAR-PRO S1 SCORER] scorer=qr")
        if any(w in text for w in bad_words):
            print(f"  {bench}: ERROR_MARKER s1={s1}")
            failed = True
            continue
        matches = pat.findall(text)
        if not matches:
            print(f"  {bench}: MISSING_SCORE s1={s1}")
            failed = True
            continue
        score = float(matches[-1])
        if bench == "pope":
            score *= 100.0
        print(f"  {bench}: {score:.4f} s1={s1}")
if failed:
    sys.exit(80)
PY

echo "ALL_DONE llava_qr_recover_textpope_train2_debug_20260723"
