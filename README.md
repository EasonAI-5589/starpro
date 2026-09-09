<div align="center">

# <img src="assets/starpro-logo.png" width="56" height="49" align="absmiddle" alt=""> STAR-Pro

**Stage-Wise Token Adaptive Reduction with Progressive Refinement<br>for Efficient Large Vision-Language Models**

<p><sub>Yichen Guo<sup>1,2,*</sup>, Tinghao Wang<sup>1,3,*</sup>, Qizhe Zhang<sup>1,*</sup>, Lingbei Meng<sup>5</sup>, Yuan Zhang<sup>1</sup>, Jiajun Cao<sup>1</sup>, Hao Jiang<sup>3</sup>, Chenwei Wu<sup>4</sup>, Jixian Wu<sup>1</sup>, Sixiang Chen<sup>1</sup>, Tao Luo<sup>3</sup>, Hongyang Cheng<sup>1</sup>, Kai Tang<sup>1,2</sup>, Chenxi Li<sup>5</sup>, Renyuan Li<sup>3</sup>, Xiande Huang<sup>5</sup>, Wenya Wang<sup>2</sup>, Shanghang Zhang<sup>1,†</sup></sub></p>

<p><sub><sup>1</sup> Peking University · <sup>2</sup> Nanyang Technological University<br>
<sup>3</sup> University of Electronic Science and Technology of China<br>
<sup>4</sup> University of Michigan, Ann Arbor · <sup>5</sup> De Artificial Intelligence Lab</sub></p>

<sub>* Equal contribution. † Corresponding author.</sub>

[Paper](https://arxiv.org/abs/2609.05916) · [PDF](https://arxiv.org/pdf/2609.05916) · [Installation](#quick-start) · [Evaluation](docs/evaluation.md) · [Baselines](docs/baselines.md) · [Figures](docs/figures.md) · [Tables](docs/tables.md) · [Citation](#citation)

[![Public release checks](https://github.com/EasonAI-5589/starpro/actions/workflows/public-release.yml/badge.svg?branch=main)](https://github.com/EasonAI-5589/starpro/actions/workflows/public-release.yml)

**Training-free visual token pruning · 7 LVLMs · 18 image and video benchmarks**

</div>

[Method](#overview) · [Quick start](#quick-start) · [Model checkpoints](#supported-models) · [Baseline evaluation](#baseline-evaluation) · [Results](#performance-across-models) · [Efficiency](#efficiency)

## Overview

STAR-Pro accelerates large vision-language models without additional training.
It preserves broad visual coverage before multimodal fusion, then refines the
surviving tokens as text-to-visual attention evolves through the decoder.

- **94.4% fewer visual tokens, 97.5% relative performance** on LLaVA-NeXT-13B
  at T = 160 ([results](#performance-across-models)).
- **2.24× measured inference speedup** on 64-frame LLaVA-Video-7B under the
  paper's one-generated-token protocol ([efficiency](#efficiency)).

<p align="center">
  <a href="assets/figures/fig3-method.png"><img src="assets/figures/fig3-method.png" width="100%" alt="STAR-Pro method: a pivoted-QR candidate pool followed by progressive text-guided pruning, illustrated with an image question-answering example."></a>
</p>

**Figure 3 · The STAR-Pro framework.** The lower row shows how the visual evidence
changes from the candidate pool to later pruning stages.

- **Adaptive Stage:** residual pivoted QR builds a query-agnostic feature-coverage
  pool with a default size of **2T**.
- **Progressive Stage:** attention is recomputed at selected decoder layers to
  retain a nested set of visual tokens.

**T is the layer-average visual token budget.** It is not a fixed token count at
every layer. See [the method and schedules](https://arxiv.org/pdf/2609.05916#page=3)
for architecture-specific details.

## Quick start

The public implementation is a **LLaVA source overlay**. Use Linux, an NVIDIA GPU
and Python 3.10 for inference. Start from a clean, pinned LLaVA checkout:

```bash
git clone https://github.com/EasonAI-5589/starpro.git
cd starpro
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

git clone https://github.com/haotian-liu/LLaVA.git ../LLaVA-starpro
git -C ../LLaVA-starpro checkout c121f0432da27facab705978f83c4ada465e46fd
bash scripts/install_overlay.sh ../LLaVA-starpro
export LLAVA_ROOT="$(cd ../LLaVA-starpro && pwd)"
```

Choose a [supported checkpoint](#supported-models), prepare an image folder and
[JSONL questions](docs/evaluation.md#question-and-answer-format), then run:

```bash
export MODEL_PATH=/path/to/llava-v1.5-7b
export QUESTION_FILE=/path/to/questions.jsonl
export IMAGE_FOLDER=/path/to/images
export OUTPUT_FILE=/path/to/new/starpro-answers.jsonl

T=64 bash scripts/run_eval.sh
```

The output path must be new. Follow the [installation guide](docs/installation.md)
for prerequisites and the [evaluation guide](docs/evaluation.md) for input formats,
benchmark scoring and NeXT image configuration.

## Supported models

The public code supports the following full LLaVA checkpoints. Each model name
links to the original provider's download page.

| Model | Checkpoint | STAR-Pro nominal layer-average T |
| --- | --- | --- |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | 128, 64, 32 |
| LLaVA-1.5-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | 128, 64, 32 |
| LLaVA-NeXT-7B | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) | 640, 320, 160 |
| LLaVA-NeXT-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) | 640, 320, 160 |

NeXT uses the [documented five-crop configuration](docs/evaluation.md#model-settings).
The paper also evaluates **Qwen3-VL, InternVL3 and LLaVA-Video**; their results and
figures are included here, while their code is not part of this LLaVA overlay.

For data preparation and scoring, follow the
[benchmark guide](docs/evaluation.md#benchmark-preparation-and-scoring).

## Baseline evaluation

The same overlay also includes **DivPrune, CDPruner, FastV and SparseVLM**, plus
an unpruned reference. Select the method with `METHOD`; the default is
`star_pro`. Use the same checkpoint, images, questions and scorer for comparisons.

After setting the paths in [Quick start](#quick-start), evaluate the four
baselines on LLaVA-1.5 with a nominal budget of 64 tokens:

```bash
mkdir -p ./baseline-answers
for method in divprune cdpruner fastv sparsevlm; do
  METHOD="$method" T=64 OUTPUT_FILE="./baseline-answers/${method}-T64.jsonl" \
    bash scripts/run_eval.sh
done

METHOD=vanilla OUTPUT_FILE=./baseline-answers/vanilla.jsonl \
  bash scripts/run_eval.sh
```

For the documented five-crop NeXT configuration, use **T=320** instead.
`T` always denotes the **total** nominal visual-token budget across crop groups;
the model converts it to each baseline's internal schedule. FastV and SparseVLM
support T=64/128 on LLaVA-1.5 and T=320/640 on NeXT. DivPrune and CDPruner also
support the smaller T=32/160 setting. See the [baseline guide](docs/baselines.md)
for method keys, original implementations and validation scope.

## Performance across models

<p align="center">
  <a href="assets/figures/fig1-performance.png"><img src="assets/figures/fig1-performance.png" width="680" alt="Paper Figure 1: benchmark radar plots for LLaVA-NeXT, LLaVA-Video, Qwen3-VL and InternVL3 at approximately 90 percent visual token reduction."></a>
</p>

**Figure 1 · Performance at approximately 90% token reduction.** Each radar axis
is normalized to the strongest plotted method. Panel summaries compare STAR-Pro
with the strongest competing method, relative to the unpruned model.

The tables below are taken directly from the [published paper](https://arxiv.org/pdf/2609.05916v1),
including their original captions, values and formatting. Click an image to
open the full-resolution table.

### LLaVA-1.5 and LLaVA-NeXT

[![Table 1: original paper results for STAR-Pro and pruning baselines on LLaVA-1.5 and LLaVA-NeXT, with 7B and 13B models.](assets/tables/table01-llava-series.png)](assets/tables/table01-llava-series.png)

<details>
<summary><strong>Video, Qwen3-VL and InternVL3 results — Tables 2–3</strong></summary>

[![Table 2: original paper results on LLaVA-Video-7B.](assets/tables/table02-video.png)](assets/tables/table02-video.png)

[![Table 3: original paper results on Qwen3-VL-8B and InternVL3-8B.](assets/tables/table03-advanced-architectures.png)](assets/tables/table03-advanced-architectures.png)

These architectures are evaluated in the paper; their code is outside the
[released LLaVA overlay](#supported-models).

</details>

The [complete gallery of all 16 paper tables](docs/tables.md) includes
[per-benchmark scores](docs/tables.md#table-6),
[matched-compute comparisons](docs/tables.md#table-11),
[token allocation](docs/tables.md#table-12) and
[candidate-pool sweeps](docs/tables.md#table-13).
Relative performance follows each original table's aggregate metric.

## Why two stages?

<p align="center">
  <a href="assets/figures/fig2-empirical-study.png"><img src="assets/figures/fig2-empirical-study.png" width="100%" alt="Paper Figure 2: feature-space coverage of retained tokens and changing visual-token importance across decoder layers."></a>
</p>

**Figure 2 · The evidence behind the design.** The left panel measures feature
coverage before fusion; the right panel tracks how attended token sets change
with decoder depth. Together, they motivate preserving coverage early and
refining the retained set progressively.

## Efficiency

<p align="center">
  <a href="assets/figures/fig4-efficiency.png"><img src="assets/figures/fig4-efficiency.png" width="100%" alt="Paper Figure 4: latency, operator-counted compute, peak GPU memory, speedup and relative performance on LLaVA-NeXT-7B and LLaVA-Video-7B."></a>
</p>

**Figure 4 · Accuracy–efficiency trade-offs.** The paper reports **2.24× inference
speedup** on LLaVA-Video-7B at 16 nominal tokens per frame over 64 frames.

The timing protocol uses one A800-80GB, batch size 1, one generated token,
10 warm-up samples and 50 measured samples; CPU video decoding and preprocessing
are excluded. Operator-counted TFLOPs exclude unexposed fused-attention operations
and top-k/QR selection. See [Appendix C.3](https://arxiv.org/pdf/2609.05916#page=17).

<details>
<summary><strong>Ablation studies: stage contributions, candidate budget and pruning layers</strong></summary>

![Figure 5: stage and schedule ablations](assets/figures/fig5-ablations.png)

**Figure 5.** Ablations on LLaVA-1.5-7B examine the two stages, the candidate-pool
multiplier and matched-compute pruning-layer choices.

The [complete paper figure gallery](docs/figures.md) also includes Figures 6–7
on schedule robustness and candidate-pool sensitivity across models and budgets.

</details>

<details>
<summary><strong>Code map and contributor checks</strong></summary>

| Location | Contents |
| --- | --- |
| [llava_arch.py](llava/model/llava_arch.py) | Adaptive-Stage QR selection and visual token construction |
| [modelling_llama_star.py](llava/model/language_model/modelling_llama_star.py) | Progressive pruning and paper schedules |
| [modeling_llama_fastv.py](llava/model/language_model/modeling_llama_fastv.py) / [modeling_llama_sparsevlm.py](llava/model/language_model/modeling_llama_sparsevlm.py) | FastV and SparseVLM decoder integration |
| [llava/eval](llava/eval) | Benchmark inference entry points |
| [assets/figures](assets/figures) | Renderings of all seven paper figures |
| [assets/tables](assets/tables) | All 16 original paper tables, with extraction metadata |
| [docs](docs) | Installation, evaluation, baselines and paper galleries |
| [tests](tests) | Entry-point checks and CPU tensor/prefill/decode tests |

Before contributing, run the [release checks](docs/evaluation.md#local-release-checks).
The checks verify source boundaries, method dispatch, token selection and cached
decoding on small CPU models; they do not rerun
the paper's GPU experiments. See [CONTRIBUTING.md](CONTRIBUTING.md).

</details>

## Citation

```bibtex
@article{guo2026starpro,
  title = {STAR-Pro: Stage-Wise Token Adaptive Reduction with Progressive Refinement for Efficient Large Vision-Language Models},
  author = {Guo, Yichen and Wang, Tinghao and Zhang, Qizhe and Meng, Lingbei and Zhang, Yuan and Cao, Jiajun and Jiang, Hao and Wu, Chenwei and Wu, Jixian and Chen, Sixiang and Luo, Tao and Cheng, Hongyang and Tang, Kai and Li, Chenxi and Li, Renyuan and Huang, Xiande and Wang, Wenya and Zhang, Shanghang},
  journal = {arXiv preprint arXiv:2609.05916},
  year = {2026},
  eprint = {2609.05916},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2609.05916},
  url = {https://arxiv.org/abs/2609.05916}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements and license

Our implementation builds on [LLaVA](https://github.com/haotian-liu/LLaVA).
We thank the authors and maintainers of the models, benchmarks and pruning
baselines used in the paper, including
[DivPrune](https://github.com/vbdi/divprune),
[CDPruner](https://github.com/Theia-4869/CDPruner),
[FastV](https://github.com/pkunlp-icler/FastV), and
[SparseVLM](https://github.com/Gumpest/SparseVLMs).
The STAR-Pro logo is the original artwork used in our paper's method figure.

STAR-Pro contributions use the [Apache License 2.0](LICENSE). Preserve the
[third-party notices](THIRD_PARTY_NOTICES.md) and follow the licenses of the
upstream code, model weights and datasets you use.
For sensitive reports, follow [SECURITY.md](SECURITY.md).
