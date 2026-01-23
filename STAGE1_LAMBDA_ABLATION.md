# Stage 1 (THCP) Lambda Ablation Study

## Overview

This document describes the λ parameter ablation implementation for Stage 1 (Text-Concept Hierarchical Coverage Pruning) of STAR-Pro.

## Purpose

To isolate and evaluate the contribution of the relevance-diversity tradeoff parameter in the THCP component, addressing the research question in the paper's ablation study section.

## Formula

**Corrected Formula:**
```
L_i(S) = (1-λ)R_i + λD_i(S)
```

where:
- `R_i`: Relevance score (text-visual similarity)
- `D_i(S)`: Diversity score (visual novelty relative to selected tokens)
- `λ ∈ [0, 1]`: Tradeoff parameter

**Note:** The paper's original formula `L_i(S) = R_i + λD_i(S)` will be corrected in the camera ready version.

## Lambda Value Interpretations

| λ Value | Relevance Weight (1-λ) | Diversity Weight (λ) | Interpretation |
|---------|------------------------|----------------------|----------------|
| 0.0     | 1.0                   | 0.0                  | Pure relevance |
| 0.25    | 0.75                  | 0.25                 | Relevance-heavy |
| 0.5     | 0.5                   | 0.5                  | Balanced |
| 0.75    | 0.25                  | 0.75                 | Diversity-heavy |
| 1.0     | 0.0                   | 1.0                  | Pure diversity |

## Implementation

### 1. Code Modifications

**File:** `llava/model/llava_arch.py` (lines 1035-1044)

```python
# Stage 1 (THCP) Lambda Ablation Support
# Formula: L_i(S) = (1-λ)R_i + λD_i(S)
# λ=0.0: pure relevance, λ=1.0: pure diversity, λ=0.5: balanced
import os
lambda_val = float(os.environ.get('LAMBDA', '0.5'))

# Unified weights using (1-λ)R + λD formula
coverage_weight = 1.0 - lambda_val   # (1-λ) for coverage in M>1 mode
relevance_weight = 1.0 - lambda_val  # (1-λ) for relevance in M=1 mode
diversity_weight = lambda_val        # λ for diversity in both modes
```

**Key Features:**
- Environment variable control: `export LAMBDA=0.5`
- Default value: λ=0.5 (balanced)
- Unified weights for both Coverage Mode (M>1) and Relevance Mode (M=1)
- Debug output shows λ configuration

### 2. Ablation Scripts

**Quick Test (Single λ):**
```bash
bash scripts/ablations/test_lambda_single.sh
```
- Tests λ=0.5 only
- Quick verification of implementation
- Check debug output for parameter confirmation

**Full Ablation:**
```bash
bash scripts/ablations/stage1_lambda_ablation.sh
```
- Tests all 5 λ values: 0.0, 0.25, 0.5, 0.75, 1.0
- Runs POPE benchmark for each
- Saves results to `./results/ablation_stage1_lambda/`
- Generates summary log with F1 scores

### 3. Manual Testing

For individual λ values:
```bash
export LAMBDA=0.5
bash scripts/v1_6/7b/pope.sh star_pro 128
```

## Configuration

- **Method:** `star_pro` (S1+S2 Full STAR-Pro pipeline)
- **Token Budget:** T=128 (target average visual tokens)
- **Benchmark:** POPE (Polling-based Object Probing Evaluation)
- **Model:** llava-v1.6-vicuna-7b
- **Baseline Expected:** ~98.4% retention @ λ=0.5 (per paper)

## Expected Results

Based on paper claims (Table 6):
- **S1 Only (THCP, λ=0.5):** ~98.0% retention @ T=128
- **S2 Only (front-heavy):** 77.7% F1 @ T=128
- **S1+S2 (STAR-Pro):** ~98.4% retention @ T=128

The ablation should show:
1. **λ=0.0 (pure relevance):** May suffer from redundancy
2. **λ=1.0 (pure diversity):** May lose text relevance
3. **λ=0.5 (balanced):** Should achieve best performance
4. **Trend:** Performance should peak around λ=0.5

## Results Directory Structure

```
results/ablation_stage1_lambda/
├── ablation_summary_<timestamp>.log    # Master summary with all F1 scores
├── lambda_0.0_pope.log                 # Individual results
├── lambda_0.25_pope.log
├── lambda_0.5_pope.log
├── lambda_0.75_pope.log
└── lambda_1.0_pope.log
```

## Debug Verification

When running with the implementation, you should see in the output:

```
================================================================================
THCP Debug - Mode: RELEVANCE
================================================================================

[Lambda Configuration]
  λ (lambda): 0.5
  Formula: L_i(S) = (1-λ)R_i + λD_i(S)
  Relevance weight (1-λ): 0.5
  Diversity weight (λ): 0.5

[Text Embeddings]
  Shape: torch.Size([...])
  M (num tokens): ...
  Using Relevance strategy
```

## Usage Workflow

### Step 1: Quick Test
```bash
# Verify implementation
bash scripts/ablations/test_lambda_single.sh
```

Check debug output confirms:
- λ = 0.5
- Relevance weight = 0.5
- Diversity weight = 0.5

### Step 2: Full Ablation
```bash
# Run all λ values
bash scripts/ablations/stage1_lambda_ablation.sh
```

Wait for completion (~30-60 minutes depending on hardware).

### Step 3: Analyze Results
```bash
# View summary
cat results/ablation_stage1_lambda/ablation_summary_*.log

# View individual results
cat results/ablation_stage1_lambda/lambda_0.5_pope.log
```

### Step 4: Documentation
Create formal results document similar to S2 ablation format.

## Camera Ready Corrections

### Equation (5) Update

**Current (incorrect):**
```
L_i(S) = R_i + λD_i(S)
```

**Corrected:**
```
L_i(S) = (1-λ)R_i + λD_i(S), where λ ∈ [0, 1]
```

**Additional clarification to add:**
"The parameter λ controls the balance between text relevance and visual diversity. λ=0 selects tokens purely by text relevance, λ=1 selects purely by visual diversity, and λ=0.5 provides balanced weighting."

## Notes

1. **Consistency:** The implementation uses the corrected formula, which will match the camera ready version
2. **Supplementary:** Ablation results in supplementary should reference the corrected formula
3. **Default:** The code defaults to λ=0.5 (balanced) when LAMBDA env var is not set
4. **Both Modes:** The same λ applies to both Coverage Mode (M>1) and Relevance Mode (M=1) for consistency

## References

- Paper Section: "Stage 1 Component Analysis"
- Related: Table 6 (ablation results)
- Code: `llava/model/llava_arch.py` (THCP implementation)
