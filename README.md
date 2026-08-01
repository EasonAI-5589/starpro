# STAR-Pro

Official LLaVA implementation of **STAR-Pro: Stage-Wise Token Adaptive
Reduction with Progressive Refinement for Efficient Large Vision-Language
Models**.

STAR-Pro is a training-free, two-stage visual-token pruning method:

1. the **Adaptive Stage** uses residual pivoted QR to build an over-budget
   feature-coverage candidate pool before multimodal fusion; and
2. the **Progressive Stage** uses evolving text-to-visual attention at selected
   decoder layers to refine a nested survivor set under a target layer-average
   token budget.

> **Release scope.** This first public release contains the audited overlay for
> LLaVA-1.5 and LLaVA-NeXT. Integrations for the other architectures evaluated
> in the paper will be released separately. Generated answers, private cluster
> scripts, submission spreadsheets, and model weights are intentionally not
> part of this repository.

## Repository layout

- `llava/model/llava_arch.py`: Adaptive-Stage residual QR selection and
  multimodal token plumbing.
- `llava/model/language_model/modelling_llama_star.py`: Progressive-Stage
  pruning and the paper schedules.
- `llava/model/language_model/llava_llama.py`: model construction and STAR-Pro
  dispatch.
- `llava/model/builder.py`: full-checkpoint loader with fail-closed STAR-Pro
  wiring.
- `llava/model/multimodal_encoder/clip_encoder.py`: vision/text features used by
  residual QR.
- `llava/eval/`: evaluation entry points compatible with STAR-Pro's generated
  output and visual-token accounting.
- `scripts/run_eval.sh`: guarded reproduction runner for the paper settings.
- `scripts/install_overlay.sh`: installer for a clean compatible LLaVA checkout.

## Installation

The overlay targets the LLaVA package version `1.2.2.post1`. The paper
environment used Python 3.10, PyTorch 2.1.2, Transformers 4.37.2, and
Accelerate 0.21.0. Python 3.10 or 3.11 is recommended because the pinned
PyTorch release does not provide Python 3.12 wheels.

```bash
git clone https://github.com/EasonAI-5589/starpro.git
cd starpro

python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Prepare a clean compatible LLaVA checkout first, then install the overlay.
bash scripts/install_overlay.sh /path/to/clean/llava
```

The installer modifies only the listed overlay files. Keep an untouched LLaVA
checkout when comparing against the unpruned baseline. This release expects a
full LLaVA checkpoint; LoRA and projector-only loading are outside the current
release contract.

## Evaluation

Set the paths for one benchmark and run the guarded entry point:

```bash
export LLAVA_ROOT=/path/to/patched/llava
export MODEL_PATH=/path/to/llava-v1.5-7b
export QUESTION_FILE=/path/to/questions.jsonl
export IMAGE_FOLDER=/path/to/images
export OUTPUT_FILE=/path/to/new/starpro-answers.jsonl

T=64 bash scripts/run_eval.sh
```

`OUTPUT_FILE` must not already exist. The runner validates both the selected
paper configuration and the resulting non-empty JSONL artifact, so stale output
cannot be mistaken for a completed run.

Paper settings selected by the runner are:

| Component | Setting |
| --- | --- |
| pruning method | `star_pro` |
| Adaptive-Stage selector | residual pivoted QR (`STAGE1_SCORER=qr`) |
| candidate multiplier | `STAGE1_MULT=2` |
| Progressive-Stage selector | text-guided top-k |
| text aggregation | average over text queries and attention heads |
| decoding | greedy (`temperature=0`) |

Supported nominal budgets are `T=128,64,32` for LLaVA-1.5 and
`T=640,320,160` for LLaVA-NeXT. NeXT applies the exact `2T` candidate budget
across its five natural crop groups. Use `ENTRYPOINT=model_vqa_loader` by
default; the runner also accepts the other evaluation modules included here.

## Reproducibility and issues

Please include the model, benchmark, nominal budget, command, environment
versions, and complete traceback when reporting a problem. Do not attach model
weights, benchmark data, API keys, or private infrastructure logs to an issue.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@article{guo2026starpro,
  title   = {STAR-Pro: Stage-Wise Token Adaptive Reduction with Progressive Refinement for Efficient Large Vision-Language Models},
  author  = {Guo, Yichen and Wang, Tinghao and Zhang, Qizhe and Meng, Lingbei and Zhang, Yuan and Cao, Jiajun and Jiang, Hao and Wu, Chenwei and Wu, Jixian and Chen, Sixiang and Luo, Tao and Cheng, Hongyang and Tang, Kai and Li, Chenxi and Li, Renyuan and Huang, Xiande and Wang, Wenya and Zhang, Shanghang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

The direct arXiv identifier will be added here as soon as it is visible in the
public arXiv index.

## License and acknowledgement

This repository is released under the [Apache License 2.0](LICENSE). It is an
overlay derived from the open-source [LLaVA](https://github.com/haotian-liu/LLaVA)
codebase; please also follow the licenses of LLaVA, its base models, and all
datasets used in evaluation.
