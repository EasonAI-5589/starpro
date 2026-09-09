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

The installer copies the files under `llava/` into the base checkout. Use a
separate checkout for unpruned comparisons. The evaluation runner puts
`LLAVA_ROOT` on `PYTHONPATH`, so installing the upstream training or web-demo
extras is not required for this inference path.

The overlay supplies its own vision-tower builder and supports standard CLIP.
S2 vision towers, LoRA adapters, projector-only weights, quantized inference,
and training are outside the documented release configuration.

## 3. Prepare model weights and a benchmark

Obtain a full checkpoint such as
[llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) from its provider.
The local checkpoint directory name should retain `llava`, as the loader uses
that name to identify the model family. Obtain the configured CLIP vision/text
weights as well; the model library can download them if they are not cached.

For NeXT, use the Vicuna 7B or 13B variant and read the five-crop requirement in
[Evaluation](evaluation.md#model-settings) before running. Models based on other
language-model families are not supported by this Llama-specific overlay.

Prepare benchmark files using the upstream
[LLaVA evaluation instructions](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md).
Then continue with [Evaluation](evaluation.md).
