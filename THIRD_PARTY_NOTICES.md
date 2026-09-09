# Third-party notices

STAR-Pro is a source overlay for LLaVA. Existing upstream copyright and license
notices remain applicable to their respective code; the repository's
[Apache License 2.0](LICENSE) does not replace licenses of separately downloaded
projects, model weights or datasets.

## LLaVA and Transformers

- **LLaVA** — Copyright 2023 Haotian Liu. The overlay retains LLaVA's Apache-2.0
  notices. The supported base revision is
  [`c121f0432da27facab705978f83c4ada465e46fd`](https://github.com/haotian-liu/LLaVA/tree/c121f0432da27facab705978f83c4ada465e46fd).
- **Hugging Face Transformers** — Copyright 2022 EleutherAI and the HuggingFace
  Inc. team. The Llama decoder integrations build on the Apache-2.0
  [`modeling_llama.py` from Transformers v4.37.2](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py).

The STAR-Pro adaptations add pruning schedules, visual-span handling and
per-layer mask/cache management for inference.

## Pruning baselines

The baseline integrations follow the corresponding original methods. Please
credit and cite them when reporting their results:

- **DivPrune** — [original project](https://github.com/vbdi/divprune).
  Its top-level distribution uses [CC BY-NC 4.0](https://github.com/vbdi/divprune/blob/main/LICENSE).
  Its LLaVA subtree separately includes an [Apache-2.0 license](https://github.com/vbdi/divprune/blob/main/LLaVA/LICENSE)
  and an Apache-2.0 header in
  [`llava_arch.py`](https://github.com/vbdi/divprune/blob/main/LLaVA/llava/model/llava_arch.py).
  The original project's top-level noncommercial license is not replaced by
  STAR-Pro's license.
- **CDPruner** — [original project](https://github.com/Theia-4869/CDPruner),
  [Apache-2.0 license](https://github.com/Theia-4869/CDPruner/blob/main/LICENSE).
  The conditional-diversity selector is integrated in `llava_arch.py`.
- **FastV** — [original project](https://github.com/pkunlp-icler/FastV).
  The project's Transformers subtree contains an
  [Apache-2.0 license](https://github.com/pkunlp-icler/FastV/blob/main/src/transformers/LICENSE)
  and retains the EleutherAI/Hugging Face notice in its Llama decoder source.
  STAR-Pro's integration uses the paper comparison's K=2 budget schedules.
- **SparseVLM** — [original project](https://github.com/Gumpest/SparseVLMs),
  [Apache-2.0 license](https://github.com/Gumpest/SparseVLMs/blob/main/LICENSE).
  The integration uses SparseVLM's text-guided pruning and clustering/merging
  rules with the STAR-Pro comparison schedules.

The public baseline runtime has been adapted for measured visual spans, the
documented crop layouts, explicit attention masks, cached decoding and CPU
contract tests. It does not imply that the baseline authors endorse these
adaptations or the reported STAR-Pro comparisons.
