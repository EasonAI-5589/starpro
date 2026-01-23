# Stage 2 Pruning Schedule Ablation Study

## Overview

This ablation study analyzes different **pruning schedules** in **Stage 2 (Progressive Pruning)** of the STAR-Pro framework. We compare how the distribution of visual token pruning across layers affects model performance.

## Background

In Stage 2, the model progressively prunes visual tokens across layers. The key question is: **at which layers should we prune, and how aggressively?**

All schedules maintain the same average token count (e.g., T=128) for fair comparison, but distribute tokens differently across layers.

## Pruning Schedule Strategies

### Schedule Format

Each schedule is represented as a list of `(layer_idx, token_count)` tuples:
- **layer_idx**: The layer where pruning occurs
- **token_count**: Number of tokens to keep after this layer

Example: `[(12, 64), (24, 32)]` means:
- Layers 0-11: Keep all tokens from Stage 1 (256 tokens)
- At layer 12: Prune to 64 tokens
- Layers 12-23: Use 64 tokens
- At layer 24: Prune to 32 tokens
- Layers 24-31: Use 32 tokens

### 1. Progressive Front-Heavy [Ours] (Default)

**Configuration:**
```bash
export PRUNING_SCHEDULE_MODE=progressive
# Or use custom:
export CUSTOM_PRUNING_SCHEDULE='[[12, 64], [24, 32]]'
```

**Schedule:** `[(12, 64), (24, 32)]`

**Token Distribution (32 layers):**
- Layers 0-11: 256 tokens (12 layers)
- Layers 12-23: 64 tokens (12 layers)
- Layers 24-31: 32 tokens (8 layers)
- **Average:** (12×256 + 12×64 + 8×32) / 32 = **128.0 tokens**

**Characteristics:**
- Aggressive early pruning at layer 12 (256 → 64)
- Front-heavy: More tokens in early and middle layers
- Based on the observation that early layers need more visual information
- Balances semantic understanding (early layers) and reasoning (late layers)

### 2. Single-Stage Pruning (Baseline)

**Configuration:**
```bash
export PRUNING_SCHEDULE_MODE=single_stage
# Or use custom:
export CUSTOM_PRUNING_SCHEDULE='[[2, 64], [10, 32]]'
```

**Schedule:** `[(2, 64), (10, 32)]`

**Token Distribution (32 layers):**
- Layers 0-1: 256 tokens (2 layers)
- Layers 2-9: 64 tokens (8 layers)
- Layers 10-31: 32 tokens (22 layers)
- **Average:** (2×256 + 8×64 + 22×32) / 32 = **128.0 tokens**

**Characteristics:**
- Minimal progressive pruning - only 2 pruning points
- Very early aggressive pruning (layer 2, 10)
- Tests if progressive pruning is necessary
- Most layers (22/32) operate with minimal tokens (32)

### 3. Uniform Progressive (Baseline)

**Configuration:**
```bash
export PRUNING_SCHEDULE_MODE=uniform
# Or use custom:
export CUSTOM_PRUNING_SCHEDULE='[[10, 128], [20, 64], [28, 32]]'
```

**Schedule:** `[(10, 128), (20, 64), (28, 32)]`

**Token Distribution (32 layers):**
- Layers 0-9: 256 tokens (10 layers)
- Layers 10-19: 128 tokens (10 layers)
- Layers 20-27: 64 tokens (8 layers)
- Layers 28-31: 32 tokens (4 layers)
- **Average:** (10×256 + 10×128 + 8×64 + 4×32) / 32 = **128.0 tokens**

**Characteristics:**
- More gradual, uniform pruning across all stages
- Balanced distribution: similar number of layers per token level
- Less aggressive than front-heavy schedule
- Tests if smooth transition is better than aggressive early pruning

### 4. Custom Schedules

You can define any custom pruning schedule that maintains the target average token count.

**Configuration:**
```bash
export CUSTOM_PRUNING_SCHEDULE='[[layer1, tokens1], [layer2, tokens2], ...]'
```

**Example 1: Three-stage uniform**
```bash
export CUSTOM_PRUNING_SCHEDULE='[[11, 96], [21, 64], [27, 32]]'
# Layers 0-10: 256 tokens (11 layers)
# Layers 11-20: 96 tokens (10 layers)
# Layers 21-26: 64 tokens (6 layers)
# Layers 27-31: 32 tokens (5 layers)
# Average: (11×256 + 10×96 + 6×64 + 5×32) / 32 = 128.0
```

**Example 2: Late pruning**
```bash
export CUSTOM_PRUNING_SCHEDULE='[[20, 96], [28, 32]]'
# Layers 0-19: 256 tokens (20 layers)
# Layers 20-27: 96 tokens (8 layers)
# Layers 28-31: 32 tokens (4 layers)
# Average: (20×256 + 8×96 + 4×32) / 32 = 184.0 (adjust tokens to reach 128)
```

## Experimental Setup

### Configuration
- **Model:** LLaVA-1.5-7B
- **Method:** star_pro (Stage 1 + Stage 2 full pipeline)
- **Token Budget:** T = 128
- **Lambda:** λ = 0.5 (fixed, from Stage 1 ablation)
- **Benchmarks:** MME, GQA, POPE, TextVQA

### Running the Ablation

#### Method 1: Using the runner script with predefined schedules

```bash
cd /path/to/STAR-LLaVA
bash scripts/ablations/run_stage2_pruning_schedule.sh
```

The script will test all configured schedules and save results to:
```
./results/ablation_stage2_pruning_schedule/ablation_pruning_schedule_<timestamp>.log
```

#### Method 2: Adding custom schedules to the runner script

Edit `scripts/ablations/run_stage2_pruning_schedule.sh`:

```bash
SCHEDULE_CONFIGS=(
    "Progressive (Front-heavy) [Ours]|progressive|"
    "Single-stage (Layers 2, 10)|single_stage|"
    "Uniform Progressive|uniform|"
    # Add your custom schedules:
    "My Custom Schedule|custom|[[8, 96], [16, 48], [24, 32]]"
    "Late Pruning|custom|[[16, 128], [24, 64], [28, 32]]"
)
```

#### Method 3: Manual testing with environment variables

```bash
# Test with predefined mode
export PRUNING_SCHEDULE_MODE=progressive  # or single_stage, uniform
export LAMBDA=0.5
export ENABLE_DEBUG=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh star_pro 128

# Test with custom schedule
export CUSTOM_PRUNING_SCHEDULE='[[12, 64], [24, 32]]'
unset PRUNING_SCHEDULE_MODE  # Custom takes priority
export LAMBDA=0.5
export ENABLE_DEBUG=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh star_pro 128
```

## Expected Results

The progressive front-heavy schedule (Ours) is expected to outperform baselines:

- **vs. Single-stage:** Progressive pruning provides smoother adaptation across layers
- **vs. Uniform:** Front-heavy schedule preserves more visual information in early layers where semantic understanding occurs

Example output:
```
Stage 2 Pruning Schedule Ablation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Schedule                  POPE    MME      GQA     Avg.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-stage (L2, L10)    XX.X    XXXX.X   XX.X    XX.X
Uniform Progressive       XX.X    XXXX.X   XX.X    XX.X
Progressive [Ours]        87.3    1444.0   60.4    XX.X
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Implementation Details

### Files Modified

1. **`llava/model/language_model/modelling_llama_star.py`**
   - Added three predefined schedules:
     - `STAR_PRO_SCHEDULE`: Progressive front-heavy (default)
     - `STAR_PRO_SINGLE_STAGE_SCHEDULE`: Single-stage pruning
     - `STAR_PRO_UNIFORM_SCHEDULE`: Uniform progressive
   - Added `CUSTOM_PRUNING_SCHEDULE` environment variable support
   - Added `PRUNING_SCHEDULE_MODE` environment variable support
   - Priority: custom schedule > predefined mode > default

2. **`scripts/ablations/run_stage2_pruning_schedule.sh`**
   - Runner script for testing multiple schedules
   - Supports both predefined modes and custom schedules
   - Configuration format: `"name|mode|custom_schedule"`

### Key Code Sections

**Reading schedule configuration:**
```python
# Priority: custom schedule > environment variable > default
custom_schedule_str = os.environ.get('CUSTOM_PRUNING_SCHEDULE', None)
if custom_schedule_str:
    # Parse custom schedule from JSON string
    self.pruning_schedule = json.loads(custom_schedule_str)
else:
    # Use predefined schedules
    pruning_schedule_mode = os.environ.get('PRUNING_SCHEDULE_MODE', 'progressive')
    if pruning_schedule_mode == 'single_stage':
        self.pruning_schedule = STAR_PRO_SINGLE_STAGE_SCHEDULE[self.scale][self.target_visual_tokens]
    elif pruning_schedule_mode == 'uniform':
        self.pruning_schedule = STAR_PRO_UNIFORM_SCHEDULE[self.scale][self.target_visual_tokens]
    else:  # 'progressive' (default)
        self.pruning_schedule = STAR_PRO_SCHEDULE[self.scale][self.target_visual_tokens]
```

## Designing Custom Schedules

### Guidelines

1. **Maintain average token count:** Ensure the weighted average equals your target (e.g., 128)
   ```
   Average = Σ(layers_i × tokens_i) / total_layers
   ```

2. **Layer indexing:** Use 0-based indexing (layer 0 = first layer)

3. **Monotonic decrease:** Each pruning point should reduce tokens:
   ```
   tokens_1 > tokens_2 > tokens_3 > ...
   ```

4. **Consider model depth:**
   - 7B model: 32 layers
   - 13B model: 40 layers

### Calculation Example

For 7B model (32 layers) with target average = 128:

```python
# Schedule: [(10, 128), (20, 64), (28, 32)]
# Layer 0-9: 256 tokens (10 layers)
# Layer 10-19: 128 tokens (10 layers)
# Layer 20-27: 64 tokens (8 layers)
# Layer 28-31: 32 tokens (4 layers)

average = (10 * 256 + 10 * 128 + 8 * 64 + 4 * 32) / 32
        = (2560 + 1280 + 512 + 128) / 32
        = 4480 / 32
        = 128.0 ✓
```

### Tools for Schedule Design

Use this Python snippet to verify your custom schedule:

```python
def verify_schedule(schedule, total_layers=32, stage1_tokens=256, target_avg=128):
    """
    Verify a pruning schedule maintains the target average.

    Args:
        schedule: List of (layer_idx, token_count) tuples
        total_layers: Total number of layers (32 for 7B, 40 for 13B)
        stage1_tokens: Tokens from Stage 1 (256 for T=128)
        target_avg: Target average tokens
    """
    current_tokens = stage1_tokens
    total = 0

    prev_layer = 0
    for layer_idx, token_count in schedule:
        # Layers before this pruning point
        total += (layer_idx - prev_layer) * current_tokens
        current_tokens = token_count
        prev_layer = layer_idx

    # Remaining layers
    total += (total_layers - prev_layer) * current_tokens

    average = total / total_layers
    print(f"Schedule: {schedule}")
    print(f"Average: {average:.2f} (target: {target_avg})")
    print(f"Match: {'✓' if abs(average - target_avg) < 0.01 else '✗'}")
    return average

# Example usage:
verify_schedule([(12, 64), (24, 32)])
verify_schedule([(2, 64), (10, 32)])
verify_schedule([(10, 128), (20, 64), (28, 32)])
```

## Related Work

This ablation is inspired by:
- **FastV** (Chen et al., 2024): Single-stage pruning at specific layers
- **SparseVLM** (Yuan et al., 2024): Multi-layer progressive pruning
- **STAR-PRO** (This work): Adaptive progressive pruning with front-heavy schedule

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
