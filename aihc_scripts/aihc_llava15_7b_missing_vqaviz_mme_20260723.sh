#!/usr/bin/env bash
set -euo pipefail

source /mnt/eason/miniconda3/etc/profile.d/conda.sh
conda activate llava

REPO=/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723
cd "${REPO}"

export CKPT_DIR=/mnt/eason_ckp/models
export DATA_DIR=/mnt/eason_ckp/LLaVA-Eval
export ENABLE_DEBUG=1

LOG_DIR="${REPO}/results"
LOG_FILE="${LOG_DIR}/aihc_llava15_7b_missing_vqaviz_mme_20260723.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "START missing VQAv2/VizWiz/MME evaluation"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

for token in 128 64 32; do
  echo "RUN HoloV VQAv2 token=${token}"
  bash scripts/v1_5/7b/vqav2.sh HoloV "${token}"
  echo "RUN HoloV VizWiz token=${token}"
  bash scripts/v1_5/7b/vizwiz.sh HoloV "${token}"
done

export RUN_TAG=missing_20260723
echo "RUN VScan MME stage1=32 stage2=32"
bash scripts/v1_5/7b/mme.sh vscan 32 32
unset RUN_TAG

echo "RUN VScan VizWiz token=32"
bash scripts/v1_5/7b/vizwiz.sh vscan 32

python - <<'PY'
import json
from pathlib import Path

repo = Path("/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723")
files = []
for token in (128, 64, 32):
    files.append(
        repo
        / "playground/data/eval/vqav2/answers_upload"
        / "llava_vqav2_mscoco_test-dev2015/llava-v1.5-7b/HoloV"
        / f"vtn_{token}.json"
    )
    files.append(
        repo
        / "playground/data/eval/vizwiz/answers_upload"
        / "llava_test/llava-v1.5-7b/HoloV"
        / f"vtn_{token}.json"
    )

files.append(
    repo
    / "playground/data/eval/mme/answers"
    / "llava_mme_test/llava-v1.5-7b/vscan"
    / "vtn_32_missing_20260723/merge.jsonl"
)
files.append(
    repo
    / "playground/data/eval/vizwiz/answers_upload"
    / "llava_test/llava-v1.5-7b/vscan/vtn_32.json"
)

for path in files:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty artifact: {path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if len(data) == 0:
            raise SystemExit(f"empty JSON submission: {path}")
    print(f"ARTIFACT_OK bytes={path.stat().st_size} path={path}")
PY

echo "ALL_DONE llava15_7b_missing_vqaviz_mme_20260723"
