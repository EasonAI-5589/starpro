# Baseline evaluation

[Back to STAR-Pro](../README.md) · [Installation](installation.md) · [Datasets](datasets.md) · [Evaluation](evaluation.md) · [Original paper tables](tables.md)

This overlay includes the LLaVA-1.5 and LLaVA-NeXT Vicuna 7B/13B integrations
used to compare STAR-Pro with DivPrune, CDPruner, FastV and SparseVLM. All methods
use the same inference entry points and benchmark scorers. The release supports
one image per question and batch size 1, with standard CLIP visual features.
Use the runner's default greedy decoding (`--num_beams 1`).

## Choose a method

Set `METHOD` when invoking `scripts/run_eval.sh`:

| `METHOD` | Selection | Original project |
| --- | --- | --- |
| `star_pro` (default) | Adaptive QR candidate pool, then progressive attention pruning | [STAR-Pro](https://arxiv.org/abs/2609.05916) |
| `divprune` | Diversity selection before the decoder | [DivPrune](https://github.com/vbdi/divprune) |
| `cdpruner` | Conditional diversity selection before the decoder | [CDPruner](https://github.com/Theia-4869/CDPruner) |
| `fastv` | Attention-based pruning at decoder layer K=2 | [FastV](https://github.com/pkunlp-icler/FastV) |
| `sparsevlm` | Text-guided progressive pruning with token merging | [SparseVLM](https://github.com/Gumpest/SparseVLMs) |
| `vanilla` | Unpruned reference | [LLaVA](https://github.com/haotian-liu/LLaVA) |

The runner maps `cdpruner` to the existing internal key `cdp3` and automatically
enables its CLIP text tower. `DivPrune`, `CDPruner`, `FastV`, and `SparseVLM`
are accepted aliases. This release uses the original SparseVLM method, not the
later SparseVLM+ variant.

## Token budgets

`T` is the **total nominal budget across all image crop groups**. Pass it unchanged
to the runner. The model derives the crop count from its image configuration and
converts the total to each method's internal per-crop setting.

| Method | LLaVA-1.5 (one crop) | NeXT (five crop groups) |
| --- | --- | --- |
| STAR-Pro, DivPrune, CDPruner | 32, 64, 128 | 160, 320, 640 |
| FastV, SparseVLM | 64, 128 | 320, 640 |
| Vanilla | Unpruned; T is ignored | Unpruned; T is ignored |

DivPrune and CDPruner retain a fixed number of visual tokens before decoding.
STAR-Pro, FastV and SparseVLM use layer-dependent schedules: nominal T describes
the target layer-average budget, while integer rounding and SparseVLM token
merging affect the realized average. For example, the 7B T=64 schedules have
layer averages of 63.1875 for FastV and 63.21875 for SparseVLM.

The public FastV/SparseVLM schedules do not include T=32 (or NeXT T=160), matching
the unavailable entries in [paper Table 1](tables.md#table-1). Those settings are
rejected before inference. The NeXT configuration must produce the documented
five crop groups; see [model settings](evaluation.md#model-settings). General
variable-crop allocation is not provided.

## Run the comparisons

Install the overlay and set `LLAVA_ROOT`, `MODEL_PATH`, `QUESTION_FILE`, and
`IMAGE_FOLDER` as described in the [quick start](../README.md#quick-start).
For LLaVA-1.5 at T=64:

```bash
mkdir -p ./baseline-answers
for method in star_pro divprune cdpruner fastv sparsevlm; do
  METHOD="$method" T=64 OUTPUT_FILE="./baseline-answers/${method}-T64.jsonl" \
    bash scripts/run_eval.sh
done

METHOD=vanilla OUTPUT_FILE=./baseline-answers/vanilla.jsonl \
  bash scripts/run_eval.sh
```

Use T=320 for the equivalent NeXT setting. Each output path must be new; choose
a different directory when repeating a run. For a different benchmark, set
`ENTRYPOINT` and its benchmark options as in the [evaluation guide](evaluation.md).
For example, to run CDPruner with MMBench circular evaluation:

```bash
QUESTION_FILE="$EVAL_ROOT/mmbench/mmbench_dev_20230712.tsv" \
  METHOD=cdpruner T=64 ENTRYPOINT=model_vqa_mmbench \
  OUTPUT_FILE=./baseline-answers/cdpruner-mmbench-circular.jsonl \
  bash scripts/run_eval.sh --all-rounds --lang en --single-pred-prompt
```

Keep checkpoint weights, image configuration, prompts, decoding settings,
benchmark splits and scorers identical when comparing methods. Record both the
nominal budget and realized layer trajectory when comparing compute. The
decoder integrations expose `layer_visual_tokens` after prefill for this purpose.

## Implementation and validation

DivPrune and CDPruner selectors are in
[`llava_arch.py`](../llava/model/llava_arch.py). FastV and SparseVLM use their
respective decoder modules in [`language_model`](../llava/model/language_model).
The wrapper measures the actual visual-token span instead of assuming a fixed
text-prefix length. The shared baseline runtime maintains masks and caches for
each decoder layer as pruning changes the sequence length.

[CPU tests](../tests/test_baseline_models.py) exercise real selectors, token
counts, crop and budget guards, layer trajectories, explicit attention masks,
prefill, repeated generation and multiple cached decode steps with small random
Llama models. CI runs these tests using the pinned PyTorch/Transformers runtime,
without downloading checkpoints. They check integration behavior; the original
paper tables are published experimental results, and this release update does
not establish a new GPU benchmark reproduction.

Please cite the corresponding original method when reporting its results and
preserve the [third-party notices](../THIRD_PARTY_NOTICES.md).
