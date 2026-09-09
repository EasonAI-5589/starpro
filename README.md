<div align="center">

# STAR-Pro

**Stage-Wise Token Adaptive Reduction with Progressive Refinement<br>for Efficient Large Vision-Language Models**

[Paper](https://arxiv.org/abs/2609.05916) · [PDF](https://arxiv.org/pdf/2609.05916) · [Installation](docs/installation.md) · [Evaluation](docs/evaluation.md) · [Citation](#citation)

</div>

STAR-Pro is a **training-free visual token pruning method**. It preserves broad
visual coverage before multimodal fusion, then progressively refines the retained
tokens using text-to-visual attention inside the language model.

## How it works

1. **Adaptive Stage:** query-agnostic residual pivoted QR selects a candidate pool
   with a default budget of **2T**.
2. **Progressive Stage:** text-to-visual attention is recomputed at selected decoder
   layers to prune a nested set of surviving tokens.

Here **T is the layer-average visual token budget**, rather than the number of
tokens retained at every layer or after the final pruning step. The method does
not require additional training or merge discarded tokens into survivors.
See [Section 3 and Appendix B](https://arxiv.org/pdf/2609.05916#page=3) for the
method and architecture-specific schedules.

## Results reported in the paper

The following rows summarize the strongest compression setting for the LLaVA
models in [Table 1](https://arxiv.org/pdf/2609.05916#page=5).

| Model | Nominal T | Visual token reduction | Relative aggregate performance |
| --- | ---: | ---: | ---: |
| LLaVA-1.5-7B | 32 | 94.4% | 94.9% |
| LLaVA-1.5-13B | 32 | 94.4% | 96.1% |
| LLaVA-NeXT-7B | 160 | 94.4% | 97.0% |
| LLaVA-NeXT-13B | 160 | 94.4% | 97.5% |

Relative performance is normalized to the unpruned model using the paper's
aggregate metric; it is not absolute benchmark accuracy. These are published
paper results, not measurements from the repository maintenance checks.

## Code release

This repository provides a **source overlay for LLaVA-1.5 and LLaVA-NeXT**.
It includes the STAR-Pro implementation, evaluation entry points, pinned runtime
dependencies, and scripts for applying the overlay to a compatible LLaVA checkout.

| Integration | Current public scope |
| --- | --- |
| LLaVA-1.5 | Full 7B/13B checkpoints; T = 128, 64, 32 |
| LLaVA-NeXT | Vicuna 7B/13B checkpoints; T = 640, 320, 160; five crop groups as described in the [evaluation guide](docs/evaluation.md) |
| Qwen3-VL, InternVL3, LLaVA-Video | Evaluated in the paper; integration code is not included in this overlay |

Start with the [installation guide](docs/installation.md), then the
[evaluation guide](docs/evaluation.md). Model weights and benchmark data must be
obtained from their original providers.

## Repository layout

```text
llava/
  model/                    STAR-Pro selection, decoder, and checkpoint loader
  eval/                     Benchmark inference entry points
scripts/
  install_overlay.sh        Apply the overlay to a clean LLaVA checkout
  run_eval.sh               Run a supported paper configuration
  check_public_release.sh   Check the public source tree before publishing
docs/                      Installation and evaluation instructions
tests/                     CPU-only entry-point contract checks
```

The Adaptive Stage is implemented in [llava_arch.py](llava/model/llava_arch.py);
the Progressive Stage and schedules are in
[modelling_llama_star.py](llava/model/language_model/modelling_llama_star.py).

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

## License and contributions

The code is distributed under the [Apache License 2.0](LICENSE) and builds on
[LLaVA](https://github.com/haotian-liu/LLaVA). Preserve upstream license notices
and follow the licenses of the model weights and datasets you use.

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue and pull request guidance, and
[SECURITY.md](SECURITY.md) for reporting sensitive issues privately.
