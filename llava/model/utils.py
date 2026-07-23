from transformers import AutoConfig
import torch
import math

def HoloV(image_tokens, attention, num_patches, new_image_token_num, esp=1e-6):
    """Official HoloV (arXiv:2510.02912), faithful port of obananas/HoloV.

    Crops are contiguous slices of the raster-ordered token sequence, i.e.
    horizontal bands -- not 2D blocks. The 2D-block variant is HoloV_2.

    attention: (B, N) CLS->patch attention, already averaged over heads.
    Returns (tokens [B, new_image_token_num, D], valid_mask [B, new_image_token_num]);
    valid_mask is False on the zero-padding emitted when allocation underfills.
    """
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 0.09
    pruned_image_tokens_list = []
    valid_masks_list = []

    for b in range(B):
        image_token = image_tokens[b]  # [N, D]
        image_attention = attention[b]  # [N]

        patch_size = N // num_patches
        remainder = N % num_patches

        image_tokens_patches = []
        attention_patches = []
        start_idx = 0
        for p in range(num_patches):
            current_patch_size = patch_size + (1 if p < remainder else 0)
            end_idx = start_idx + current_patch_size
            if current_patch_size > 0:
                image_tokens_patches.append(image_token[start_idx:end_idx])
                attention_patches.append(image_attention[start_idx:end_idx])
            start_idx = end_idx

        patch_scores = []
        all_patches = []
        for p in range(len(image_tokens_patches)):
            patch_tokens = image_tokens_patches[p]
            patch_attn = attention_patches[p]
            current_patch_size = len(patch_tokens)

            if current_patch_size <= 1:
                patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
                all_patches.append(patch_tokens)
                continue

            with torch.no_grad():
                F_normalized = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + esp)
                S = torch.mm(F_normalized, F_normalized.transpose(0, 1))
                eye_mask = 1 - torch.eye(current_patch_size, device=device)
                S_masked = S * eye_mask

                valid_entries = current_patch_size - 1
                mean_sim = S_masked.sum(dim=1) / valid_entries
                var_sim = ((S_masked - mean_sim.unsqueeze(1))**2).sum(dim=1) / valid_entries

                patch_attn_scaled = patch_attn * 1e3
                var_scaling = (torch.mean(torch.abs(patch_attn_scaled)) /
                               (torch.mean(torch.abs(var_sim)) + esp))
                var_sim_scaled = var_sim * var_scaling

                token_scores = patch_attn_scaled + alpha * var_sim_scaled

                patch_scores.append(token_scores.mean())
                all_patches.append(patch_tokens)

        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)

        if len(patch_scores) > 0:
            weights = patch_scores / (patch_scores.sum() + esp)
            allocated = (weights * new_image_token_num).floor().long()

            remaining = new_image_token_num - allocated.sum()
            if remaining > 0 and len(weights) > 0:
                _, indices = torch.topk(weights, k=min(remaining.item(), len(weights)))
                for idx in indices[:remaining]:
                    allocated[idx] += 1

            new_patches = []
            for i, (patch, alloc) in enumerate(zip(all_patches, allocated)):
                patch_size = len(patch)
                if alloc <= 0:
                    continue
                elif alloc >= patch_size:
                    new_patches.append(patch)
                else:
                    # Selection within a crop is by CLS attention alone -- the variance
                    # term only shapes the per-crop budget. This mirrors the reference
                    # implementation, which differs from Eq.5 of the paper.
                    patch_attn = attention_patches[i]
                    _, top_indices = torch.topk(patch_attn, k=min(alloc.item(), patch_size))
                    new_patches.append(patch[top_indices])

            new_image_tokens = torch.cat(new_patches, dim=0) if new_patches \
                else torch.zeros((0, D), device=device)
        else:
            new_image_tokens = torch.zeros((0, D), device=device)

        actual_tokens = new_image_tokens.size(0)
        valid_mask = torch.ones(new_image_token_num, dtype=torch.bool, device=device)
        if actual_tokens < new_image_token_num:
            padding = torch.zeros((new_image_token_num - actual_tokens, D), device=device)
            new_image_tokens = torch.cat([new_image_tokens, padding], dim=0)
            valid_mask[actual_tokens:] = False
        elif actual_tokens > new_image_token_num:
            new_image_tokens = new_image_tokens[:new_image_token_num]

        pruned_image_tokens_list.append(new_image_tokens)
        valid_masks_list.append(valid_mask)

    return (torch.stack(pruned_image_tokens_list, dim=0).to(image_tokens.dtype),
            torch.stack(valid_masks_list, dim=0))


def HoloV_2(image_tokens, attention, num_patches, new_image_token_num, esp=1e-6):
    attention = attention.unsqueeze(0)
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 1
    beta = 0.09
    power = 1
    pruned_image_tokens_list = []
    final_positions_list = []  

    for b in range(B):
        image_token = image_tokens[b]  # [N, D]
        image_attention = attention[b]  # [N]

        # Reshape tokens to 2D grid for spatial patch construction
        H = int(math.sqrt(N))
        W = N // H
        assert H * W == N, f"Token count {N} cannot form a 2D grid"

        image_token_2d = image_token.reshape(H, W, D)
        image_attention_2d = image_attention.reshape(H, W)

        # Determine patch grid dimensions (most square-like factorization)
        grid_h = int(math.sqrt(num_patches))
        while num_patches % grid_h != 0:
            grid_h -= 1
        grid_w = num_patches // grid_h

        # Create 2D grid patches
        image_tokens_patches = []
        attention_patches = []
        patch_indices_list = []

        ph = H // grid_h
        pw = W // grid_w
        h_remainder = H % grid_h
        w_remainder = W % grid_w

        row_start = 0
        for i in range(grid_h):
            current_ph = ph + (1 if i < h_remainder else 0)
            col_start = 0
            for j in range(grid_w):
                current_pw = pw + (1 if j < w_remainder else 0)

                patch = image_token_2d[row_start:row_start + current_ph,
                                       col_start:col_start + current_pw]
                attn_patch = image_attention_2d[row_start:row_start + current_ph,
                                               col_start:col_start + current_pw]

                # Compute original 1D indices for this 2D patch
                rows = torch.arange(row_start, row_start + current_ph, device=device)
                cols = torch.arange(col_start, col_start + current_pw, device=device)
                grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
                indices_1d = grid_rows * W + grid_cols

                patch_flat = patch.reshape(-1, D)
                attn_flat = attn_patch.reshape(-1)
                indices_flat = indices_1d.reshape(-1)

                if len(patch_flat) > 0:
                    image_tokens_patches.append(patch_flat)
                    attention_patches.append(attn_flat)
                    patch_indices_list.append(indices_flat)

                col_start += current_pw
            row_start += current_ph

        # Process each patch separately
        patch_scores = []
        all_patches = []
        all_patch_indices = []

        for p in range(len(image_tokens_patches)):
            patch_tokens = image_tokens_patches[p]  # [current_patch_size, D]
            patch_attn = attention_patches[p]  # [current_patch_size]
            current_patch_size = len(patch_tokens)

            patch_indices = patch_indices_list[p]

            if current_patch_size <= 1:
                # If patch has only one token or is empty, handle specially
                patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
                all_patches.append(patch_tokens)
                all_patch_indices.append(patch_indices)
                continue

            with torch.no_grad():
                # Normalize patch tokens
                F_normalized = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + esp)

                # Compute similarity matrix
                S = torch.mm(F_normalized, F_normalized.transpose(0, 1))

                # Create eye mask of appropriate size
                eye_mask = 1 - torch.eye(current_patch_size, device=device)
                S_masked = S * eye_mask

                # Compute mean and variance
                valid_entries = current_patch_size - 1
                mean_sim = S_masked.sum(dim=1) / valid_entries
                var_sim = ((S_masked - mean_sim.unsqueeze(1))**2).sum(dim=1) / valid_entries

                # Scale attention
                patch_attn_scaled = patch_attn * 1e3

                # Scale variance
                var_scaling = (torch.mean(torch.abs(patch_attn_scaled)) / 
                              (torch.mean(torch.abs(var_sim)) + esp))
                var_sim_scaled = var_sim * var_scaling

                # Calculate token scores
                token_scores = alpha * patch_attn_scaled + beta * var_sim_scaled

                # Compute patch score
                patch_score = token_scores.mean()
                patch_scores.append(patch_score)
                all_patches.append(patch_tokens)
                all_patch_indices.append(patch_indices)

        # Convert to tensor
        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)

        # Allocate new tokens based on scores
        if len(patch_scores) > 0:
            weights = (patch_scores ** power) / ((patch_scores ** power).sum() + esp)
            allocated = (weights * new_image_token_num).floor().long()

            # Distribute remaining tokens
            remaining = new_image_token_num - allocated.sum()
            if remaining > 0 and len(weights) > 0:
                _, indices = torch.topk(weights, k=min(remaining.item(), len(weights)))
                for idx in indices[:remaining]:
                    allocated[idx] += 1

            # Handle token overflow
            new_patches = []
            final_positions = []  

            for i, (patch, alloc, patch_indices) in enumerate(zip(all_patches, allocated, all_patch_indices)):
                patch_size = len(patch)
                if alloc <= 0:
                    continue
                elif alloc >= patch_size:
                    # Keep all tokens in this patch
                    new_patches.append(patch)
                    final_positions.append(patch_indices)
                else:
                    # Sample tokens based on attention scores
                    patch_attn = attention_patches[i]
                    _, top_indices = torch.topk(patch_attn, k=min(alloc.item(), patch_size))
                    new_patches.append(patch[top_indices])
                    final_positions.append(patch_indices[top_indices])

            # Combine all selected tokens
            if new_patches:
                new_image_tokens = torch.cat(new_patches, dim=0)
                final_positions = torch.cat(final_positions, dim=0) 
            else:
                new_image_tokens = torch.zeros((0, D), device=device)
                final_positions = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # No patches to process
            new_image_tokens = torch.zeros((0, D), device=device)
            final_positions = torch.zeros(0, dtype=torch.long, device=device)

        # Pad or truncate to match expected new_image_token_num
        actual_tokens = new_image_tokens.size(0)
        if actual_tokens < new_image_token_num:
            # Pad with zeros if we don't have enough tokens
            padding = torch.zeros((new_image_token_num - actual_tokens, D), device=device)
            new_image_tokens = torch.cat([new_image_tokens, padding], dim=0)
            
            
            padding_positions = torch.full((new_image_token_num - actual_tokens,), -1, dtype=torch.long, device=device)
            final_positions = torch.cat([final_positions, padding_positions], dim=0)
        elif actual_tokens > new_image_token_num:
            # Truncate if we have too many tokens
            new_image_tokens = new_image_tokens[:new_image_token_num]
            final_positions = final_positions[:new_image_token_num]

        pruned_image_tokens_list.append(new_image_tokens)
        final_positions_list.append(final_positions)

    
    return torch.stack(pruned_image_tokens_list, dim=0), torch.stack(final_positions_list, dim=0).squeeze(0)

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
            

