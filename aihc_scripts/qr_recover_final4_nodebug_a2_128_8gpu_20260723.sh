#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
TAG="${STARPRO_RUN_TAG:-llavaqr_recover_a2_128_final4_8gpu_20260723}"
LOG_ROOT="${STARPRO_LOG_ROOT:-$ROOT/results_aihc_qr_recover_final4_20260723}"
OUT="$LOG_ROOT/$TAG"
BENCHES="${STARPRO_BENCHES:-mme textvqa pope gqa}"
MME_FIXED_QUESTIONS="$ROOT/aihc_scripts/generated/llava_mme_test_pathfix_${TAG}.jsonl"

mkdir -p "$OUT"

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate llava

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CKPT_DIR="${CKPT_DIR:-/mnt/eason_ckp/models}"
export DATA_DIR="${DATA_DIR:-/mnt/eason_ckp/LLaVA-Eval}"
export ENABLE_DEBUG="${STARPRO_ENABLE_DEBUG:-0}"
export STAR_KEEP_POSIDS=1
export STAR_CAUSAL_FIX=1
export STAGE1_SCORER=qr
export STAGE1_MULT=2
export TEXT_AGG_MODE=average_all
export METHOD=star_pro
export TOKEN=128
export VISUAL_TOKEN_NUM="$TOKEN"
export RUN_TAG="$TAG"
export RESULT_DIR="$OUT"
export STARPRO_BENCHES="$BENCHES"
export MME_QUESTION_FILE="$MME_FIXED_QUESTIONS"
export MME_CONVERTER="$ROOT/aihc_scripts/convert_answer_to_mme_robust_20260723.py"
unset LAMBDA STAGE1_SVD_RANK CUSTOM_PRUNING_SCHEDULE

python3 "$ROOT/aihc_scripts/repair_mme_question_paths_20260723.py" \
  --source "$DATA_DIR/MME/llava_mme.jsonl" \
  --image-root "$DATA_DIR/MME/MME_Benchmark_release_version" \
  --output "$MME_FIXED_QUESTIONS"

cd "$ROOT"

{
  echo "CONFIG tag=$TAG BENCHES=$BENCHES METHOD=$METHOD TOKEN=$TOKEN VISUAL_TOKEN_NUM=$VISUAL_TOKEN_NUM STAGE1_SCORER=$STAGE1_SCORER STAGE1_MULT=$STAGE1_MULT STAR_KEEP_POSIDS=$STAR_KEEP_POSIDS STAR_CAUSAL_FIX=$STAR_CAUSAL_FIX TEXT_AGG_MODE=$TEXT_AGG_MODE ENABLE_DEBUG=$ENABLE_DEBUG CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  for bench in $BENCHES; do
    echo "===== BEGIN $TAG $bench ====="
    bash "scripts/v1_5/7b/${bench}.sh" "$METHOD" "$TOKEN" 2>&1 | tee "$OUT/${bench}.fulllog"
    echo "===== END $TAG $bench ====="
  done
} 2>&1 | tee "$OUT/suite.log"

python3 - <<'PY'
import os
import re
import sys

root = os.environ["RESULT_DIR"]
benches = os.environ["STARPRO_BENCHES"].split()
debug_enabled = os.environ.get("ENABLE_DEBUG", "0") == "1"
bad_words = (
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "FileNotFoundError",
    "WARNING: scorer=qr failed",
    "[ERROR]",
)
failed = False

for bench in benches:
    path = os.path.join(root, f"{bench}.fulllog")
    if not os.path.exists(path):
        print(f"{bench}: MISSING_LOG")
        failed = True
        continue

    text = open(path, errors="replace").read()
    qr_markers = text.count("[STAR-PRO S1 SCORER] scorer=qr")
    causal_markers = text.count("[STAR-PRO CAUSAL FIX]")
    average_all_markers = text.count("Average-all (")
    markers = [word for word in bad_words if word in text]
    if markers:
        print(
            f"{bench}: ERROR_MARKERS={markers} qr_markers={qr_markers} "
            f"causal_markers={causal_markers} "
            f"average_all_markers={average_all_markers}"
        )
        failed = True
        continue
    if (
        qr_markers <= 0
        or (debug_enabled and causal_markers <= 0)
        or (debug_enabled and average_all_markers <= 0)
    ):
        print(
            f"{bench}: FINAL_CONFIG_NOT_PROVEN qr_markers={qr_markers} "
            f"causal_markers={causal_markers} "
            f"average_all_markers={average_all_markers} "
            f"debug_enabled={debug_enabled}"
        )
        failed = True
        continue

    if bench in ("gqa", "textvqa"):
        matches = re.findall(r"Accuracy:\s*([\d.]+)%", text)
        if not matches:
            print(f"{bench}: MISSING_SCORE qr_markers={qr_markers}")
            failed = True
            continue
        print(
            f"{bench}: accuracy={float(matches[-1]):.4f} "
            f"qr_markers={qr_markers} causal_markers={causal_markers} "
            f"average_all_markers={average_all_markers}"
        )
    elif bench == "pope":
        matches = re.findall(r"Average F1 score:\s*([\d.]+)", text)
        if not matches:
            print(f"{bench}: MISSING_SCORE qr_markers={qr_markers}")
            failed = True
            continue
        print(
            f"{bench}: average_f1={100.0 * float(matches[-1]):.4f} "
            f"raw_average_f1={float(matches[-1]):.16f} "
            f"qr_markers={qr_markers} causal_markers={causal_markers} "
            f"average_all_markers={average_all_markers}"
        )
    elif bench == "mme":
        perception = text.find("=========== Perception ===========")
        cognition = text.find("=========== Cognition ===========")
        perception_text = text[perception:cognition] if perception >= 0 else ""
        cognition_text = text[cognition:] if cognition >= 0 else ""
        p_scores = re.findall(r"total score:\s*([\d.]+)", perception_text)
        c_scores = re.findall(r"total score:\s*([\d.]+)", cognition_text)
        if not p_scores or not c_scores:
            print(f"{bench}: MISSING_SCORE qr_markers={qr_markers}")
            failed = True
            continue
        print(
            f"{bench}: perception={float(p_scores[-1]):.4f} "
            f"cognition={float(c_scores[-1]):.4f} "
            f"qr_markers={qr_markers} causal_markers={causal_markers} "
            f"average_all_markers={average_all_markers}"
        )
    else:
        print(f"{bench}: UNSUPPORTED_BENCH")
        failed = True

if failed:
    sys.exit(80)
PY

echo "ALL_DONE llava_qr_recover_final4_a2_128_8gpu_20260723"
