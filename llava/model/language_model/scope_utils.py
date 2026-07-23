# SCOPE: Saliency-Coverage Oriented Token Pruning
# Reference: https://github.com/kinredon/SCOPE
# Paper: "SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs" (NeurIPS 2025)

import os
import torch


def scope_select(visual_features, num_tokens, cls_attn=None, alpha=1.0, combined='multi'):
    """
    SCOPE token selection algorithm.

    Jointly models saliency (CLS attention) and coverage (token diversity) to select
    the most representative visual tokens.

    Args:
        visual_features: [B, N, D] visual token features (after removing CLS)
        num_tokens: number of tokens to select
        cls_attn: [B, N] CLS attention scores (saliency)
        alpha: exponent for attention scores (default: 1.0)
        combined: 'multi' for multiplication, 'add' for addition

    Returns:
        selected_idx: [B, K] indices of selected tokens
        cosine_simi: [B, N, N] cosine similarity matrix
    """
    # Compute cosine similarity matrix for coverage
    norm_vectors = visual_features / visual_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))

    B, N = visual_features.shape[:2]
    device = visual_features.device
    dtype = visual_features.dtype

    # Pre-allocate tensors
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_idx = torch.empty(B, num_tokens, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, dtype=dtype, device=device)

    # Prepare saliency scores
    if cls_attn is not None:
        cls_attn_powered = (cls_attn ** alpha).to(dtype)
    else:
        cls_attn_powered = torch.ones(B, N, dtype=dtype, device=device)

    # Greedy selection: iteratively pick token with highest SCOPE score
    for i in range(num_tokens):
        # Calculate coverage gains for all unselected tokens
        # Gain = sum of max(0, similarity to unselected - current coverage)
        unselected_mask = ~selected

        gains = torch.maximum(
            torch.zeros(1, dtype=dtype, device=device),
            cosine_simi.masked_fill(~unselected_mask.unsqueeze(1), 0) - cur_max.unsqueeze(2)
        ).sum(dim=1)

        # Combine coverage gain with saliency score
        if combined == 'multi':
            scope_score = gains * cls_attn_powered
        elif combined == 'add':
            scope_score = gains + cls_attn_powered
        else:
            raise ValueError(f"Unknown combined mode: {combined}")

        # Mask out already selected tokens
        scope_score = scope_score.masked_fill(~unselected_mask, float('-inf'))

        # Select token with highest SCOPE score
        best_idx = scope_score.argmax(dim=1)

        # Update states
        selected[torch.arange(B, device=device), best_idx] = True
        selected_idx[:, i] = best_idx
        cur_max = torch.maximum(cur_max, cosine_simi[torch.arange(B, device=device), best_idx])

    return selected_idx, cosine_simi


def scope_select_with_env(visual_features, num_tokens, cls_attn=None):
    """
    SCOPE selection with environment variable configuration.

    Environment variables:
        ALPHA: exponent for attention scores (default: 1.0)
        COMBINED: 'multi' or 'add' (default: 'multi')

    Note: Uses same env var names as official SCOPE implementation.
    """
    alpha = float(os.environ.get('ALPHA', '1.0'))
    combined = os.environ.get('COMBINED', 'multi')

    return scope_select(visual_features, num_tokens, cls_attn, alpha=alpha, combined=combined)
