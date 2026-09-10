# Evaluation datasets

[Back to STAR-Pro](../README.md) · [Installation](installation.md) · [Evaluation and scoring](evaluation.md) · [Baselines](baselines.md)

Use the **official LLaVA evaluation assets plus each benchmark's official data**.
The directory name `LLaVA-Eval` is a local data root, not a single benchmark or
a promise that one download contains every dataset. This guide covers the ten
LLaVA image benchmarks in the paper; the other architectures and video datasets
are outside this source overlay.

## 1. Obtain the LLaVA evaluation assets

Download [eval.zip from the official LLaVA instructions](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view).
LLaVA describes this archive as containing prepared questions/annotations,
evaluation scripts, reference predictions and a directory structure. Images and
benchmark-specific scoring assets must also be obtained below. See the
[pinned upstream instructions](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#scripts).

Extract the assets so that `gqa`, `MME`, `textvqa`, etc. are directly inside one
directory. Set `EVAL_ROOT` to that directory:

```bash
export EVAL_ROOT=/path/to/LLaVA-Eval
# Or, if using the original upstream layout:
# export EVAL_ROOT="$LLAVA_ROOT/playground/data/eval"
```

`EVAL_ROOT` is used to construct the paths in the examples; the runner receives
`QUESTION_FILE` and `IMAGE_FOLDER`. Keep datasets outside the source repository.
An existing local dataset collection can be reused if its contents match the
paths and splits below. Do not assume a third-party collection or an
access-restricted mirror includes the complete suite.

## 2. Download each benchmark's data

The links below are the original benchmark sources or links provided by their
maintainers. Download only the benchmarks you intend to evaluate.

| Benchmark | Images / benchmark source | Additional scoring assets |
| --- | --- | --- |
| VQAv2 | COCO [test2015 images](http://images.cocodataset.org/zips/test2015.zip) | [VQA evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission); this recipe uses **test-dev2015** questions |
| GQA | [Official download page](https://cs.stanford.edu/people/dorarad/gqa/download.html) | [Official evaluator and annotations](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) under `gqa/data`; use **testdev balanced** |
| VizWiz | Official [test images](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) and [annotations](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) | [Evaluation server](https://eval.ai/web/challenges/challenge-page/2185/my-submission) |
| ScienceQA | [Official download instructions](https://github.com/lupantech/ScienceQA#ghost-download-the-dataset) | [`problems.json` and `pid_splits.json`](https://github.com/lupantech/ScienceQA/tree/main/data/scienceqa) under `scienceqa`; use the **test image subset** |
| TextVQA | Official [train/val images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) | [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) under `textvqa` |
| POPE | COCO [val2014 images](http://images.cocodataset.org/zips/val2014.zip) | [Official COCO POPE annotations](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) under `pope/coco` |
| MME | [Official MME instructions](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), including the linked [dataset archive](https://huggingface.co/datasets/darkyarding/MME/blob/main/MME_Benchmark_release_version.zip) | Official `eval_tool` under `MME`; retain the original question/answer text files and category layout |
| MMBench-EN | Official [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) | Images are embedded in the TSV; use the matching split and [scoring protocol](evaluation.md#benchmark-preparation-and-scoring) |
| MMBench-CN | Official [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) | Images are embedded in the TSV; do not substitute newer MMBench versions silently |
| MM-Vet | Official [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) | [Original evaluation instructions](https://github.com/yuweihao/MM-Vet); record the evaluator version used |

## 3. Match the input paths

All paths in this table are relative to `EVAL_ROOT`. Question JSONL files and
ScienceQA's original question JSON come from LLaVA's prepared evaluation assets;
raw benchmark annotation files are not interchangeable with those model inputs.

| Benchmark | `QUESTION_FILE` | `IMAGE_FOLDER` | `ENTRYPOINT` |
| --- | --- | --- | --- |
| VQAv2 | `vqav2/llava_vqav2_mscoco_test-dev2015.jsonl` | `vqav2/test2015` | `model_vqa_loader` |
| GQA | `gqa/llava_gqa_testdev_balanced.jsonl` | `gqa/data/images` | `model_vqa_loader` |
| VizWiz | `vizwiz/llava_test.jsonl` | `vizwiz/test` | `model_vqa_loader` |
| ScienceQA | `scienceqa/llava_test_CQM-A_image.json` (prepare below) | `scienceqa/images/test` | `model_vqa_science` |
| TextVQA | `textvqa/llava_textvqa_val_v051_ocr.jsonl` | `textvqa/train_images` | `model_vqa_loader` |
| POPE | `pope/llava_pope_test.jsonl` | `pope/val2014` | `model_vqa_loader` |
| MME | `MME/llava_mme.jsonl` | `MME/MME_Benchmark_release_version` | `model_vqa_loader` |
| MMBench-EN | `mmbench/mmbench_dev_20230712.tsv` | Not needed; TSV embeds images | `model_vqa_mmbench` |
| MMBench-CN | `mmbench_cn/mmbench_dev_cn_20231003.tsv` | Not needed; TSV embeds images | `model_vqa_mmbench` |
| MM-Vet | `mm-vet/llava-mm-vet.jsonl` | `mm-vet/images` | `model_vqa` |

The path spellings follow the
[pinned LLaVA evaluation scripts](https://github.com/haotian-liu/LLaVA/tree/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval):
use **`mmbench_cn`** for the Chinese TSV and **`mm-vet`** for MM-Vet. The upstream
prose uses different directory spellings in places. If an archive adds an extra
outer `eval` or `mm-vet` directory, point `EVAL_ROOT`/`IMAGE_FOLDER` at the actual
contents instead of nesting the paths twice.

For MME, preserve both flat categories and categories containing `images/` and
`questions_answers_YN/`; flattening or deleting these directories can break the
official scorer. For POPE, annotation JSON files alone do not include the COCO
images. For TextVQA, use the OCR-formatted LLaVA questions shown above.

### Prepare the ScienceQA image subset

The public pruning methods require an image. Keep the original
`llava_test_CQM-A.json` and derive a separate input containing only image-bearing
questions. This command creates a new file and refuses to overwrite an existing one:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"]) / "scienceqa"
rows = json.loads((root / "llava_test_CQM-A.json").read_text())
assert isinstance(rows, list), "Expected the LLaVA-format ScienceQA JSON list"
image_rows = [row for row in rows if row.get("image")]
assert image_rows, "No image-bearing questions found"
destination = root / "llava_test_CQM-A_image.json"
with destination.open("x", encoding="utf-8") as handle:
    json.dump(image_rows, handle, ensure_ascii=False, indent=2)
print(f"Prepared {len(image_rows)} of {len(rows)} questions: {destination}")
PY
```

Use `scienceqa/images/test` as `IMAGE_FOLDER`, pass `--single-pred-prompt`, and
report the image-subset score from the official ScienceQA evaluator.

## 4. Run with explicit paths

After [installing the overlay](installation.md), set `LLAVA_ROOT` and `MODEL_PATH`.
For GQA:

```bash
export QUESTION_FILE="$EVAL_ROOT/gqa/llava_gqa_testdev_balanced.jsonl"
export IMAGE_FOLDER="$EVAL_ROOT/gqa/data/images"
export OUTPUT_FILE=./answers/starpro-gqa-T64.jsonl

ENTRYPOINT=model_vqa_loader METHOD=star_pro T=64 bash scripts/run_eval.sh
```

For MMBench-CN circular evaluation:

```bash
QUESTION_FILE="$EVAL_ROOT/mmbench_cn/mmbench_dev_cn_20231003.tsv" \
OUTPUT_FILE=./answers/starpro-mmbench-cn-circular-T64.jsonl \
ENTRYPOINT=model_vqa_mmbench METHOD=star_pro T=64 \
  bash scripts/run_eval.sh --lang cn --single-pred-prompt --all-rounds
```

`IMAGE_FOLDER` is unused by MMBench. Use `--lang en` and the English TSV for
MMBench-EN. The four [baselines](baselines.md) use the same dataset paths; change
`METHOD` and choose a new output file for each run.

The runner rejects a missing/empty question file or a missing required image
directory before loading the model. This checks entry paths, not every image,
record, annotation or expected sample count. Run a small image-containing subset
first, then score the full split with the [benchmark-specific evaluator](evaluation.md#benchmark-preparation-and-scoring).
Record the data version, split and expected/actual record counts with the result.
