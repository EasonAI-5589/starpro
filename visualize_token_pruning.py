"""
Token pruning visualization for LLaVA-v1.5-7b + PrefixVLM_2.
- Randomly selects 50 images from MME dataset
- Runs single-layer token pruning (layer and target token count configurable)
- Visualizes: original image vs pruned image (pruned regions shown as dark gray)
"""

import os
import sys
import random
import argparse
import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Fix device mismatch: auto-move bool index tensors to match the indexed tensor's device.
# Needed because in encode_images, 'device' is captured before mm_projector moves
# image_features, causing available (bool mask) and vis_sim to land on different GPUs.
_orig_tensor_getitem = torch.Tensor.__getitem__
def _safe_tensor_getitem(self, index):
    if isinstance(index, torch.Tensor) and index.dtype == torch.bool and index.device != self.device:
        index = index.to(self.device)
    elif isinstance(index, tuple):
        index = tuple(
            i.to(self.device) if isinstance(i, torch.Tensor) and i.dtype == torch.bool and i.device != self.device else i
            for i in index
        )
    return _orig_tensor_getitem(self, index)
torch.Tensor.__getitem__ = _safe_tensor_getitem

# ── project root ──────────────────────────────────────────────────────────────
sys.path.insert(0, "/mnt/eason/LLaVA-STAR-Pro2")

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images, tokenizer_image_token, get_model_name_from_path,
)
from llava.utils import disable_torch_init


# ── globals to capture pruning indices ───────────────────────────────────────
_PRUNING_EVENTS = []   # list of dicts per pruning layer

def _patch_prefix2_forward(model_inner):
    """
    Monkey-patch Prefix_2_LlamaModel.forward so we can intercept
    `retained_visual_index` without touching the source file.
    """
    from llava.model.language_model.modeling_llama_prefix_2 import Prefix_2_LlamaModel
    from transformers.models.llama.modeling_llama import (
        _prepare_4d_causal_attention_mask,
    )

    original_forward = Prefix_2_LlamaModel.forward

    def patched_forward(self, *args, **kwargs):
        _PRUNING_EVENTS.clear()

        # We need to intercept inside the layer loop.
        # Strategy: wrap the forward pass and extract from debug prints is messy;
        # instead, override the part that writes `retained_visual_index`.
        # Simplest: call original forward but hook via __class__ attribute.
        return original_forward(self, *args, **kwargs)

    # Better: directly patch the class to record indices.
    # We add a hook that gets called just after `all_selected` is computed.
    # The cleanest way is to override `forward` inline.

    import types
    import einops as ein
    from typing import Dict, List, Optional, Tuple, Union
    from transformers.models.llama import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaModel, Cache, DynamicCache,
        _prepare_4d_causal_attention_mask,
        _prepare_4d_causal_attention_mask_for_sdpa,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast

    R_dict = {
        "7b": {
            8:  {192: 192},
            12: {128: 64,  64: 32, 32: 16},
            16: {192: 144},
            24: {192: 96, 128: 32, 64: 16, 32: 8},
        },
        "13b": {
            15: {192: 128, 128: 64, 64: 32, 32: 16},
            30: {192: 64,  128: 32, 64: 16, 32: 8},
        },
    }

    def new_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if seq_length > 1:
            visual_token_length = self.visual_token_length
            visual_token_num = 0
            visual_token_list = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, attention_mask, position_ids,
                    past_key_values, output_attentions, use_cache,
                )
            else:
                if seq_length > 1:
                    visual_token_num += visual_token_length
                    visual_token_list.append(visual_token_length)

                    if (decoder_layer.self_attn.layer_idx + 1) in self.pruning_loc:
                        attn_mask = torch.ones(
                            (batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(
                            attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        hidden_states = layer_outputs[0]

                        visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length + visual_token_length]
                        text_tokens   = hidden_states[:, self.system_prompt_length + visual_token_length:]
                        matrix = text_tokens @ visual_tokens.transpose(1, 2)
                        matrix = matrix.squeeze(0).softmax(0).mean(1)
                        text_rater_index = torch.where(matrix > matrix.mean())[0]

                        self_attn_weights = layer_outputs[1].mean(1)
                        cross_attn_weights = self_attn_weights[
                            :,
                            text_rater_index + self.system_prompt_length + visual_token_length,
                            self.system_prompt_length:self.system_prompt_length + visual_token_length,
                        ].mean(1)

                        alpha_2 = float(os.environ.get("ALPHA_2", "0.5"))
                        lambda_2 = float(os.environ.get("LAMBDA_2", "0.1"))
                        coverage_method = os.environ.get("COVERAGE", "MEAN")

                        total_visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length + visual_token_length]
                        B, V, D = total_visual_tokens.shape
                        device = hidden_states.device
                        dynamic_res = 5 if self.anyres else 1
                        retained_num = int(R_dict[self.scale][decoder_layer.self_attn.layer_idx + 1][self.retained_num] * dynamic_res)

                        attn = cross_attn_weights.detach()
                        if attn.dim() == 1:
                            attn = attn.unsqueeze(0)
                        attn = (attn - attn.min(dim=1, keepdim=True).values) / (
                            attn.max(dim=1, keepdim=True).values - attn.min(dim=1, keepdim=True).values + 1e-8)

                        vis_norm = torch.nn.functional.normalize(total_visual_tokens, p=2, dim=-1)

                        all_selected = []
                        for b in range(B):
                            a    = attn[b]
                            feat = vis_norm[b]
                            available = torch.ones(V, dtype=torch.bool, device=device)
                            selected = []

                            i0 = torch.argmax(a).item()
                            selected.append(i0)
                            available[i0] = False
                            max_sim_to_S = feat @ feat[i0]
                            max_sim_to_S[i0] = 1.0
                            full_sim = feat @ feat.t()

                            for step in range(1, retained_num):
                                if not available.any():
                                    break
                                diversity = 1.0 - max_sim_to_S
                                if coverage_method == "SCOPE":
                                    coverage = torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                                    coverage[~available] = 0.0
                                else:
                                    sim_to_avail = full_sim[:, available].sum(dim=1)
                                    n_avail = available.sum().item()
                                    coverage = (sim_to_avail - 1.0) / (n_avail - 1) if n_avail > 1 else torch.zeros(V, device=device)

                                scores = (a / a.mean()) + alpha_2 * (diversity / diversity.mean()) + lambda_2 * (coverage / coverage.mean())
                                scores[~available] = -float("inf")
                                idx = torch.argmax(scores).item()
                                selected.append(idx)
                                available[idx] = False
                                sim_vec = full_sim[idx]
                                max_sim_to_S = torch.maximum(max_sim_to_S, sim_vec)
                                max_sim_to_S[idx] = 1.0

                            selected_t = torch.sort(torch.tensor(selected, device=device, dtype=torch.long)).values
                            all_selected.append(selected_t)

                        retained_visual_index = all_selected[0].unsqueeze(0)  # (1, R)

                        # ── RECORD for visualization ──────────────────────
                        _PRUNING_EVENTS.append({
                            "layer_idx": decoder_layer.self_attn.layer_idx + 1,
                            "total_tokens": V,
                            "retained_num": retained_num,
                            "retained_indices": retained_visual_index[0].cpu(),  # (R,)
                        })
                        # ─────────────────────────────────────────────────

                        select_visual_tokens = total_visual_tokens.gather(
                            1, retained_visual_index.unsqueeze(-1).expand(-1, -1, D))

                        hidden_states = torch.cat((
                            hidden_states[:, :self.system_prompt_length],
                            select_visual_tokens,
                            hidden_states[:, self.system_prompt_length + visual_token_length:],
                        ), dim=1)

                        position_ids = position_ids[:, :hidden_states.shape[1]]
                        if not output_attentions:
                            layer_outputs = (hidden_states, layer_outputs[2]) if use_cache else (hidden_states,)
                        visual_token_length = select_visual_tokens.shape[1]

                    else:
                        attn_mask = torch.ones(
                            (batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(
                            attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=torch.tensor(
                            [[past_key_values.get_usable_length(1, decoder_layer.self_attn.layer_idx)]],
                            dtype=torch.int64).cuda(),
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if seq_length > 1:
            self.visual_token_num = visual_token_num / len(self.layers)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    from llava.model.language_model.modeling_llama_prefix_2 import Prefix_2_LlamaModel
    Prefix_2_LlamaModel.forward = new_forward


# ── helpers ───────────────────────────────────────────────────────────────────

def get_mme_images(mme_root, n=50, seed=42):
    """Collect n random jpg images from MME benchmark."""
    all_images = glob.glob(os.path.join(mme_root, "**", "*.jpg"), recursive=True)
    all_images += glob.glob(os.path.join(mme_root, "**", "*.png"), recursive=True)
    random.seed(seed)
    random.shuffle(all_images)
    return all_images[:n]


def visualize_pruning(orig_image: Image.Image, grid_h: int, grid_w: int,
                      total_tokens: int, retained_indices, patch_size_px: int,
                      layer_idx: int, out_path: str):
    """
    Draw two side-by-side panels:
      left  – original image
      right – pruned image (pruned patches → faint white wash, original faintly visible)
    """
    WHITE_BLEND = 0.78
    W, H = orig_image.size

    # Build mask: True = retained, False = pruned
    mask = np.zeros(total_tokens, dtype=bool)
    mask[retained_indices.numpy()] = True
    mask_2d = mask.reshape(grid_h, grid_w)

    orig = orig_image.convert("RGB")
    washed = orig.point(lambda p: int(p * (1 - WHITE_BLEND) + 255 * WHITE_BLEND))
    pruned = washed.copy()

    ph = H / grid_h  # patch height in px
    pw = W / grid_w  # patch width  in px

    for row in range(grid_h):
        for col in range(grid_w):
            if mask_2d[row, col]:
                x0, y0 = int(col * pw), int(row * ph)
                x1, y1 = int((col + 1) * pw), int((row + 1) * ph)
                patch = orig.crop((x0, y0, x1, y1))
                pruned.paste(patch, (x0, y0))

    n_kept   = int(mask.sum())
    n_pruned = total_tokens - n_kept

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_image)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(pruned)
    axes[1].set_title(
        f"After pruning (layer {layer_idx})\n"
        f"Kept {n_kept}/{total_tokens} tokens  |  Pruned {n_pruned}",
        fontsize=11,
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",   default="/mnt/eason_ckp/models/llava-v1.5-7b")
    p.add_argument("--mme-root",     default="/mnt/eason_ckp/LLaVA-Eval/MME/MME_Benchmark_release_version")
    p.add_argument("--output-dir",   default="/mnt/eason/LLaVA-STAR-Pro2/pruning_visualization_results")
    p.add_argument("--num-images",   type=int, default=50)
    p.add_argument("--seed",         type=int, default=42)
    # pruning config
    p.add_argument("--prune-layer",  type=int, default=12,
                   help="Which layer to prune at (1-indexed, e.g. 12 or 24)")
    p.add_argument("--token-budget", type=int, default=128,
                   help="T parameter: final target token count (e.g. 32/64/128/192)")
    p.add_argument("--alpha2",       type=float, default=0.5)
    p.add_argument("--lambda2",      type=float, default=0.5)
    p.add_argument("--coverage",     default="COVERAGE")
    p.add_argument("--conv-mode",    default="vicuna_v1")
    return p.parse_args()


def main():
    args = parse_args()

    # ── env vars for the pruning method ──
    os.environ["ALPHA_2"]    = str(args.alpha2)
    os.environ["LAMBDA_2"]   = str(args.lambda2)
    os.environ["COVERAGE"]   = args.coverage
    # STAGE1_KEEP = 2 * token_budget (mirrors run_prefixvlm2_llava_7b.sh)
    os.environ["STAGE1_KEEP"] = str(2 * args.token_budget)

    # ── patch forward to record pruning indices ──
    _patch_prefix2_forward(None)

    # ── load model ──
    disable_torch_init()
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(args.model_path)
    print(f"Loading model: {model_name}")

    prefixvlm_2_config = {"T": args.token_budget}

    # Override pruning_loc to single layer
    import llava.model.language_model.modeling_llama_prefix_2 as _m2
    original_init = _m2.Prefix_2_LlamaModel.__init__

    prune_layer = args.prune_layer

    def patched_init(self, config, prefixvlm_2_config):
        original_init(self, config, prefixvlm_2_config)
        # Override to single pruning layer
        self.pruning_loc = [prune_layer]
        print(f"[VIS] pruning_loc overridden → {self.pruning_loc}")

    _m2.Prefix_2_LlamaModel.__init__ = patched_init

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name,
        pruning_method="prefixvlm_2",
        use_prefixvlm_2=True,
        prefixvlm_2_config=prefixvlm_2_config,
        device_map="cuda:0",
    )
    model.eval()
    print("Model loaded.")


    # ── figure out visual token grid dimensions ──
    # CLIP ViT-L/14 @ 336 → 24×24 = 576 tokens
    # STAGE1_KEEP tokens are expected at start of stage-2
    stage1_keep = int(os.environ.get("STAGE1_KEEP", "288"))
    # Assuming square grid closest to stage1_keep
    grid_side = int(stage1_keep ** 0.5)
    # If not a perfect square, fall back to a rough estimate
    if grid_side * grid_side != stage1_keep:
        # LLaVA-1.5 ViT patch grid is always 24×24; stage1 keeps a subset
        # We still display relative to what stage-2 receives (not original 576)
        grid_w = grid_side
        grid_h = (stage1_keep + grid_side - 1) // grid_side
    else:
        grid_h = grid_w = grid_side

    print(f"Visual token grid for stage-2 input: {grid_h}×{grid_w} = {grid_h*grid_w} tokens")
    print(f"  (STAGE1_KEEP={stage1_keep}, prune_layer={prune_layer}, T={args.token_budget})")

    # ── get images ──
    images = get_mme_images(args.mme_root, n=args.num_images, seed=args.seed)
    print(f"Selected {len(images)} images.")

    os.makedirs(args.output_dir, exist_ok=True)

    question = "Describe what you see in the image."

    for img_idx, img_path in enumerate(images):
        print(f"\n[{img_idx+1}/{len(images)}] {os.path.basename(img_path)}")

        try:
            raw_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skip (load error): {e}")
            continue

        # ── build input ──
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([raw_image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        # ── forward pass (capture pruning events) ──
        _PRUNING_EVENTS.clear()
        with torch.inference_mode():
            _ = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[raw_image.size],
                do_sample=False,
                max_new_tokens=32,
                use_cache=True,
            )

        if not _PRUNING_EVENTS:
            print("  No pruning events captured (check config).")
            continue

        # ── visualize each pruning layer ──
        for ev in _PRUNING_EVENTS:
            layer_idx      = ev["layer_idx"]
            total_tokens   = ev["total_tokens"]
            retained_idx   = ev["retained_indices"]
            retained_num   = ev["retained_num"]

            # Recompute grid for actual token count received
            side = int(total_tokens ** 0.5)
            gh = side
            gw = (total_tokens + side - 1) // side

            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(
                args.output_dir,
                f"{img_idx:03d}_{base}_layer{layer_idx}_keep{retained_num}.png",
            )

            # Resize raw image to match assumed grid if needed
            display_image = raw_image.resize((gw * 14, gh * 14), Image.LANCZOS)

            visualize_pruning(
                display_image, gh, gw,
                total_tokens, retained_idx,
                patch_size_px=14,
                layer_idx=layer_idx,
                out_path=out_path,
            )
            print(f"  Saved: {out_path}")
            print(f"  Layer {layer_idx}: {retained_num}/{total_tokens} tokens retained  ({100*retained_num/total_tokens:.1f}%)")

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# export http_proxy=http://192.168.32.28:18000  
# export https_proxy=http://192.168.32.28:18000 
# python visualize_token_pruning.py