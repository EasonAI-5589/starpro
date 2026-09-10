# Installation

[Back to STAR-Pro](../README.md)

Use Linux with an NVIDIA GPU for inference. The pinned runtime uses Python 3.10,
PyTorch 2.1.2, Transformers 4.37.2, and Accelerate 0.21.0. The release checks can
run on a CPU; they do not establish GPU inference or benchmark accuracy.

## 1. Create the environment

```bash
git clone https://github.com/EasonAI-5589/starpro.git
cd starpro
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Prepare the base framework

This repository contains replacement source files, not the entire LLaVA package.
The base checkout below is LLaVA `1.2.2.post1`, pinned to an immutable commit.
Run these commands from the STAR-Pro directory:

```bash
git clone https://github.com/haotian-liu/LLaVA.git ../LLaVA-starpro
git -C ../LLaVA-starpro checkout c121f0432da27facab705978f83c4ada465e46fd
bash scripts/install_overlay.sh ../LLaVA-starpro
export LLAVA_ROOT="$(cd ../LLaVA-starpro && pwd)"
```

The installer copies the files under `llava/` into the base checkout. Select
`METHOD=vanilla` for unpruned comparisons with the same configuration. The evaluation runner puts
`LLAVA_ROOT` on `PYTHONPATH`, so installing the upstream training or web-demo
extras is not required for this inference path.

The overlay supplies its own vision-tower builder and supports standard CLIP.
S2 vision towers, LoRA adapters, projector-only weights, quantized inference,
and training are outside the documented release configuration.

## 3. Prepare model weights and a benchmark

Choose one of the full Vicuna checkpoints supported by this overlay:

| Model | Official checkpoint | STAR-Pro nominal layer-average T |
| --- | --- | --- |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | 128, 64, 32 |
| LLaVA-1.5-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | 128, 64, 32 |
| LLaVA-NeXT-7B | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) | 640, 320, 160 |
| LLaVA-NeXT-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) | 640, 320, 160 |

Download the chosen checkpoint locally and set `MODEL_PATH` to that directory.
Keep `llava` in its directory name, as the loader uses the name to identify the
model family. Obtain the configured CLIP vision/text weights as well; the model
library can download them if they are not cached.

For NeXT, these budgets apply to **five crop groups**: one global crop and a
2 × 2 local grid, giving 2,880 visual patch tokens before pruning. Prepare the
[five-crop checkpoint configuration](evaluation.md#model-settings) before
running. Models based on other language-model families are not supported by this
Llama-specific overlay.

DivPrune, CDPruner, FastV and SparseVLM share this installation. Their available
budgets and method commands are listed in the [baseline guide](baselines.md).

Prepare benchmark files using the [dataset guide](datasets.md), which separates
LLaVA's `eval.zip` assets from each benchmark's images and scoring annotations
and gives the exact paths expected by this overlay.
Then follow the [question format example](evaluation.md#question-and-answer-format)
or the [benchmark preparation and scoring table](evaluation.md#benchmark-preparation-and-scoring)
to run your first evaluation.
