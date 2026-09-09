#!/usr/bin/env bash
set -euo pipefail

: "${LLAVA_ROOT:?set LLAVA_ROOT to the patched LLaVA checkout}"
: "${MODEL_PATH:?set MODEL_PATH to local model weights}"
: "${QUESTION_FILE:?set QUESTION_FILE to the benchmark questions}"
: "${IMAGE_FOLDER:?set IMAGE_FOLDER to the benchmark images}"
: "${OUTPUT_FILE:?set OUTPUT_FILE to a new JSONL path}"

T=${T:-64}
ENTRYPOINT=${ENTRYPOINT:-model_vqa_loader}
CONV_MODE=${CONV_MODE:-llava_v1}
case "$ENTRYPOINT" in
  model_vqa|model_vqa_loader|model_vqa_mmbench|model_vqa_science) ;;
  *) printf 'Unsupported ENTRYPOINT: %s\n' "$ENTRYPOINT" >&2; exit 2 ;;
esac
case "$T" in
  128|64|32|640|320|160) ;;
  *) printf 'T must be a paper budget: 128/64/32 for LLaVA-1.5 or 640/320/160 for LLaVA-NeXT\n' >&2; exit 2 ;;
esac

# Paper settings belong to the runner. Extra options may tune a benchmark,
# but must not silently override the method, budget, inputs, or output path.
for arg in "$@"; do
  case "$arg" in
    --model-path|--model-path=*|--model-base|--model-base=*|--question-file|--question-file=*|--image-folder|--image-folder=*|--answers-file|--answers-file=*|--conv-mode|--conv-mode=*|--temperature|--temperature=*|--pruning_method|--pruning_method=*|--visual_token_num|--visual_token_num=*)
      printf 'Use the documented environment settings instead of overriding %s\n' "$arg" >&2
      exit 2
      ;;
    --num-chunks|--num-chunks=*|--chunk-idx|--chunk-idx=*|--num_beams|--num_beams=*|--max_new_tokens|--max_new_tokens=*|--top_p|--top_p=*|--lang|--lang=*|--all-rounds|--single-pred-prompt|--answer-prompter)
      ;;
    -*)
      printf 'Unsupported extra option: %s\n' "$arg" >&2
      exit 2
      ;;
  esac
done

if [[ -e "$OUTPUT_FILE" ]]; then
  printf 'OUTPUT_FILE already exists; use a new path: %s\n' "$OUTPUT_FILE" >&2
  exit 2
fi
mkdir -p "$(dirname -- "$OUTPUT_FILE")"
export PYTHONPATH="$LLAVA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export STAGE1_SCORER=qr
export STAGE1_MULT=2
export TEXT_AGG_MODE=average_all
export STAGE2_SELECTOR=topk
export PRUNING_SCHEDULE_MODE=progressive
export STAGE2_ANCHOR_M=0
export STAR_CAUSAL_FIX=0
export STAR_KEEP_POSIDS=0
unset CUSTOM_PRUNING_SCHEDULE
export ENABLE_DEBUG=${ENABLE_DEBUG:-0}

printf 'integration=llava method=star_pro scorer=qr stage1_mult=2 text_agg=average_all stage2=topk T=%s entrypoint=%s\n' "$T" "$ENTRYPOINT"
python -u -m "llava.eval.$ENTRYPOINT" \
  "$@" \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --image-folder "$IMAGE_FOLDER" \
  --answers-file "$OUTPUT_FILE" \
  --conv-mode "$CONV_MODE" \
  --temperature 0 \
  --pruning_method star_pro \
  --visual_token_num "$T"

test -s "$OUTPUT_FILE"
python - "$OUTPUT_FILE" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
rows = 0
with p.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            json.loads(line)
            rows += 1
if rows == 0:
    raise SystemExit("no JSONL records were produced")
print(f"artifact_ok rows={rows} path={p}")
PY
