# Stage 2 Text Aggregation Ablation Study

## Overview

This ablation study analyzes different text token aggregation strategies in **Stage 2 (Progressive Pruning)** of the STAR-Pro framework. We compare three methods for determining which visual tokens to keep based on text guidance.

## Background

In Stage 2, the model progressively prunes visual tokens across layers using text-to-visual attention. The key question is: **how should we aggregate information from multiple text tokens to guide visual token selection?**

## Three Text Aggregation Strategies

### 1. Last Token Only (Baseline)
**Implementation:** PDrop-style single-token attention
```python
# Use only the last text token's attention to visual tokens
visual_attention = attn_avg[0, -1, visual_start:visual_end]
```

**Characteristics:**
- Uses only the final token position from the text sequence
- Simple and computationally efficient
- Vulnerable to noise in the last token
- May miss key information that appears early in the query

**Usage:**
```bash
export TEXT_AGG_MODE=last_token
```

### 2. Multi-Token Guidance [Ours] (Default)
**Implementation:** Importance-weighted multi-token aggregation
```python
# Step 1: Identify important text tokens via text-visual similarity
text_visual_sim = torch.matmul(text_hidden, visual_hidden.transpose(1, 2))
text_importance = text_visual_sim.softmax(dim=0).mean(dim=1)

# Step 2: Select above-average importance tokens as "raters"
text_rater_mask = text_importance > text_importance.mean()
text_rater_indices = torch.where(text_rater_mask)[0]

# Step 3: Average attention from selected raters to visual tokens
visual_attention = rater_to_visual_attn.mean(dim=0)
```

**Characteristics:**
- Identifies important text tokens based on their similarity to visual content
- Filters out less relevant tokens (e.g., stopwords, generic terms)
- Aggregates attention from multiple important tokens
- Balances coverage of key concepts with computational efficiency

**Usage:**
```bash
export TEXT_AGG_MODE=multi_token  # or omit (default)
```

### 3. Average All (Baseline)
**Implementation:** Simple averaging of all text tokens
```python
# Average attention from ALL text tokens without filtering
all_text_to_visual_attn = attn_avg[0, visual_end:, visual_start:visual_end]
visual_attention = all_text_to_visual_attn.mean(dim=0)
```

**Characteristics:**
- Treats all text tokens equally
- No importance weighting or filtering
- May dilute signal from important tokens with noise from generic words
- Simpler than multi-token but less selective

**Usage:**
```bash
export TEXT_AGG_MODE=average_all
```

## Experimental Setup

### Configuration
- **Model:** LLaVA-1.5-7B
- **Method:** star_v3 (Stage 1 + Stage 2 full pipeline)
- **Token Budget:** T = 128
- **Lambda:** λ = 0.5 (fixed, from Stage 1 ablation)
- **Benchmarks:** MME, GQA, POPE, TextVQA

### Running the Ablation

```bash
cd /path/to/STAR-LLaVA
bash scripts/ablations/run_stage2_text_agg.sh
```

The script will:
1. Test all three text aggregation modes
2. Save results to `./results/ablation_stage2_text_agg/ablation_text_agg_<timestamp>.log`
3. Output comparison table

## Expected Results

The multi-token guidance method (Ours) is expected to outperform both baselines by:
- **vs. Last Token Only:** Better coverage of diverse concepts in the query
- **vs. Average All:** Better signal-to-noise ratio through importance filtering

Example output:
```
Stage 2 Text Aggregation Ablation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method              POPE    MME      GQA     Avg.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
last_token          XX.X    XXXX.X   XX.X    XX.X
multi_token [Ours]  87.3    1444.0   60.4    XX.X
average_all         XX.X    XXXX.X   XX.X    XX.X
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Implementation Details

### File Modified
- **`llava/model/language_model/modelling_llama_star.py`** (lines 473-544)
  - Added `TEXT_AGG_MODE` environment variable support
  - Reorganized text aggregation logic into three branches
  - Maintained backward compatibility (defaults to `multi_token`)

### Key Code Sections

**Reading environment variable:**
```python
text_agg_mode = os.environ.get('TEXT_AGG_MODE', 'multi_token')
```

**Branch selection:**
```python
if text_agg_mode == 'last_token':
    # PDrop baseline
elif text_agg_mode == 'average_all':
    # Simple averaging baseline
else:  # 'multi_token'
    # Our method (default)
```

## Related Work

This ablation is inspired by:
- **PDrop** (Yin et al., 2024): Uses last-token attention
- **SparseVLM** (Yuan et al., 2024): Multi-token approach for encoder-decoder models
- **STAR-V3** (This work): Adapted multi-token guidance for decoder-only LVLMs

## Citation

If you use this ablation study, please cite:

```bibtex
@article{star2024,
  title={STAR: Spatial-Temporal Augmentation with Text-to-visual Attention for Referring video object segmentation},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```
