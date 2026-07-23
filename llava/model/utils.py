from transformers import AutoConfig
import torch
import math

def HoloV(image_tokens, attention, num_patches, new_image_token_num, esp=1e-6):
    """Official HoloV (arXiv:2510.02912), faithful port of obananas/HoloV.

    Crops are contiguous slices of the raster-ordered token sequence, i.e.
    horizontal bands -- not 2D blocks.

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
            

