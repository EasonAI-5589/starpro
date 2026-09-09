# Evaluation

[Back to STAR-Pro](../README.md) · [Installation](installation.md)

## Run inference

After applying the overlay and activating the environment, set explicit paths:

```bash
export LLAVA_ROOT=/path/to/LLaVA-starpro
export MODEL_PATH=/path/to/llava-v1.5-7b
export QUESTION_FILE=/path/to/questions.jsonl
export IMAGE_FOLDER=/path/to/images
export OUTPUT_FILE=/path/to/new/starpro-answers.jsonl

T=64 bash scripts/run_eval.sh
```

`OUTPUT_FILE` must be new. For the default `model_vqa_loader` entry point,
questions are JSONL objects with `question_id`, `image`, and `text` fields;
image paths are relative to `IMAGE_FOLDER`. Use a small prepared subset to check
inference before running the full official split.

## Question and answer format

For a custom image question, put an image you want to evaluate at
`$IMAGE_FOLDER/example.jpg` and add this single line to `QUESTION_FILE`:

```jsonl
{"question_id":1,"image":"example.jpg","text":"Describe this image."}
```

Use one JSON object per line, without an enclosing list. Each `question_id`
identifies a question; `image` names a file relative to `IMAGE_FOLDER`; `text`
contains the question and any benchmark-specific answer instructions. The
default loader inserts the model's image token and conversation template.

After inference, `OUTPUT_FILE` contains one answer object per question. The
following shows its format; angle-bracketed values are placeholders:

```jsonl
{"question_id":1,"prompt":"Describe this image.","text":"<generated answer>","answer_id":"<generated id>","model_id":"llava-v1.5-7b","metadata":{}}
```

| Field | Meaning |
| --- | --- |
| `question_id` | Input question identifier, used to match predictions to annotations |
| `prompt` | Original question text |
| `text` | Generated answer |
| `answer_id` | Identifier generated for this answer |
| `model_id` | Model name derived from the checkpoint path |
| `metadata` | Additional metadata; empty in the current entry points |

The [default loader](../llava/eval/model_vqa_loader.py) defines this input/output
contract. MMBench accepts TSV input and adds `round_id`, `options`, and
`option_char` to its answer records; ScienceQA accepts the JSON-list format
described below.

## Model settings

| Model family | Nominal T | Conversation mode | Input contract |
| --- | --- | --- | --- |
| LLaVA-1.5 7B / 13B | 128, 64, 32 | `llava_v1` | Standard 336-pixel CLIP input |
| LLaVA-NeXT Vicuna 7B / 13B | 640, 320, 160 | `llava_v1` | Five crop groups: one global crop plus a 2 × 2 local grid |

The NeXT paper configuration uses 2,880 input visual patch tokens before
pruning. In a separate local checkpoint configuration, this corresponds to
`image_grid_pinpoints: [[672, 672]]` with a 336-pixel CLIP tower. Preserve the
original checkpoint configuration for baseline comparisons. The default NeXT
checkpoint permits other grids; this release does not provide general
variable-crop budget allocation and can reject such inputs.

The runner fixes `pruning_method=star_pro`, residual QR (`STAGE1_SCORER=qr`),
`STAGE1_MULT=2`, `TEXT_AGG_MODE=average_all`, progressive schedules, top-k
Stage-2 selection, and greedy decoding. It also clears custom schedules and
sets the optional anchor/causal/position experiment switches to their default
zero values. The schedule is selected from the model size and nominal budget.
Use the budgets for the selected model family.

## Entry points and input formats

| `ENTRYPOINT` | Input | Notes |
| --- | --- | --- |
| `model_vqa_loader` (default) | LLaVA JSONL questions | Image question answering |
| `model_vqa` | LLaVA JSONL questions | Alternative image question-answering loader |
| `model_vqa_mmbench` | Official MMBench TSV | Use `--all-rounds` when preparing circular evaluation |
| `model_vqa_science` | LLaVA-format ScienceQA JSON list: `id`, `image`, and `conversations` | Image-containing subset only; question text comes from `conversations[0].value` |

Example for MMBench:

```bash
ENTRYPOINT=model_vqa_mmbench T=64 bash scripts/run_eval.sh --all-rounds --lang en
```

Optional arguments are limited to benchmark controls: `--num-chunks`,
`--chunk-idx`, `--num_beams`, `--max_new_tokens`, `--top_p`, `--lang`,
`--all-rounds`, `--single-pred-prompt`, and `--answer-prompter`. Availability
varies by entry point. Use the environment variables for model/input/output
paths and T; overriding the runner's fixed arguments is rejected.

## Benchmark preparation and scoring

The links below use the same pinned LLaVA revision as the installation guide.
Start with its [evaluation assets and directory layout](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#scripts),
then follow the dataset-specific preparation instructions. Run STAR-Pro inference
with `scripts/run_eval.sh`; the linked upstream scripts document the subsequent
conversion and scoring steps.

| Benchmark | Data preparation | STAR-Pro entry point | Conversion and scoring reference |
| --- | --- | --- | --- |
| GQA | [Questions and images](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#gqa) | `model_vqa_loader` | [Prediction conversion and GQA evaluator](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/gqa.sh) |
| TextVQA | [Validation data and OCR questions](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#textvqa) | `model_vqa_loader` | [TextVQA answer scorer](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/textvqa.sh) |
| POPE | [COCO questions and annotations](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#pope) | `model_vqa_loader` | [POPE scorer](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/pope.sh) |
| MME | [Images and official evaluation tool](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#mme) | `model_vqa_loader` | [Answer conversion and MME scoring](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/mme.sh) |
| MMBench-EN / CN | [English TSV](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#mmbench) / [Chinese TSV](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#mmbench-cn) | `model_vqa_mmbench` | [Single-round submission preparation](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/mmbench.sh) and the evaluation server linked in the data guide |
| ScienceQA (image subset) | [Problems, splits and images](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#scienceqa) | `model_vqa_science` | [ScienceQA scorer](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/sqa.sh); report the image-subset metric |

Set `QUESTION_FILE` and `IMAGE_FOLDER` to the prepared files for the selected
benchmark. For ScienceQA, retain only records with an image in the LLaVA-format
JSON list. For Chinese MMBench, pass `--lang cn`.

MMBench circular evaluation uses `--all-rounds` and requires scoring each
question across its option rotations. The upstream single-round submission
converter matches only `question_id`; use a round-aware scorer for circular
results instead of passing multi-round records to that converter.

## Score and verify

The runner checks that inference exits successfully and produces non-empty,
valid JSONL. This is an artifact-format check, not a completeness or accuracy
check. Score predictions with the official benchmark-specific scorer from the
upstream evaluation workflow.

For a reproducible result, record the source commit, environment versions,
checkpoint and image configuration, benchmark split, expected/actual sample
counts, nominal T, sharding, command, and scorer. MMBench circular and
single-pass scores must not be interchanged. Report paper comparisons using the
same benchmark protocol and aggregate metric.

## Local release checks

From this repository's root:

```bash
bash scripts/check_public_release.sh
python -m compileall -q llava tests
python -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
git diff --check
```

These checks cover public-file boundaries, Python/shell syntax, and entry-point
contracts using stubs. Full benchmark reproduction and performance measurements
require the actual GPU, models, datasets, and scorers.
