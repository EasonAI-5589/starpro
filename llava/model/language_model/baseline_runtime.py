# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Haotian Liu
# Copyright 2026 STAR-Pro contributors
# Licensed under the Apache License, Version 2.0.

"""Inference plumbing shared by the released FastV and SparseVLM decoders.

Each decoder retains its own ranking, schedule and merging rules. A separate
mask is kept for each layer because pruning changes the length of its KV cache.
"""

from types import MethodType

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    Cache, DynamicCache, _prepare_4d_causal_attention_mask,
)


def _full_rotary_table(self, x, seq_len=None):
    # FastV retains original positions after pruning. Later layers have fewer
    # keys, but still need the rotary table at those original positions.
    if seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return self.cos_cached.to(x.dtype), self.sin_cached.to(x.dtype)


def initialize_baseline(model, config):
    if config.num_hidden_layers not in (32, 40):
        raise ValueError("Baseline schedules require a 32-layer or 40-layer Llama")
    if getattr(config, "image_aspect_ratio", None) not in ("pad", "anyres"):
        raise ValueError("Baseline configurations require image_aspect_ratio=pad or anyres")
    if model._use_flash_attention_2:
        raise ValueError("FastV/SparseVLM require eager or SDPA attention, not FlashAttention-2")
    model.layer_visual_tokens = []
    model._layer_padding_masks = []
    for layer in model.layers:
        # Scope this compatibility repair to this model's rotary modules.
        layer.self_attn.rotary_emb.forward = MethodType(
            _full_rotary_table, layer.self_attn.rotary_emb
        )


def baseline_forward(model, input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, use_cache=None,
                     output_attentions=None, output_hidden_states=None, return_dict=None):
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Supply exactly one of input_ids and inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)
    batch, seq_length = inputs_embeds.shape[:2]
    if batch != 1:
        raise ValueError("Released baseline inference requires batch size 1")
    if model.training:
        raise ValueError("Released baseline decoders support inference only; call eval()")
    output_attentions = model.config.output_attentions if output_attentions is None else output_attentions
    output_hidden_states = model.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    use_cache = model.config.use_cache if use_cache is None else use_cache
    return_dict = model.config.use_return_dict if return_dict is None else return_dict
    legacy = not isinstance(past_key_values, Cache)
    cache = DynamicCache.from_legacy_cache(past_key_values) if legacy else past_key_values
    is_prefill = cache.get_seq_length() == 0
    if not is_prefill and seq_length != 1:
        raise ValueError("Cached baseline decoding requires one new token per step")
    if not use_cache and not is_prefill:
        raise ValueError("Cached baseline decoding requires use_cache=True")

    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
            raise ValueError("Baseline attention_mask must have shape [1, sequence length]")
        if attention_mask.shape[-1] < seq_length:
            raise ValueError("attention_mask is shorter than the input sequence")
    device = inputs_embeds.device
    current_mask = (
        torch.ones((1, seq_length), device=device, dtype=torch.bool)
        if attention_mask is None else attention_mask[:, -seq_length:].to(device).bool()
    )
    if position_ids is None:
        offset = cache.get_seq_length()
        position_ids = torch.arange(offset, offset + seq_length, device=device).unsqueeze(0)
    if position_ids.shape != (1, seq_length):
        raise ValueError("position_ids must have shape [1, sequence length]")
    input_position_limit = int(position_ids.max()) + 1

    hidden_states = inputs_embeds
    visual_length = model.visual_token_length
    if is_prefill:
        start = model.system_prompt_length
        if start is None or start < 0 or start + visual_length >= seq_length:
            raise ValueError("Set the measured visual span and provide text after the image")
        if not current_mask[:, start:start + visual_length].all():
            raise ValueError("The visual sequence cannot contain masked padding")
        model.layer_visual_tokens = []
        model._layer_padding_masks = []
        state = model.prepare_pruning(hidden_states, current_mask)
    elif len(model._layer_padding_masks) != len(model.layers):
        raise ValueError("The KV cache must originate from this model's multimodal prefill")

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for index, layer in enumerate(model.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if is_prefill:
            model.layer_visual_tokens.append(visual_length)
            layer_padding_mask = current_mask
        else:
            past_mask = model._layer_padding_masks[index].to(hidden_states.device)
            if past_mask.shape[-1] != cache.get_seq_length(index):
                raise ValueError("Stored attention mask and per-layer KV cache disagree")
            layer_padding_mask = torch.cat((past_mask, current_mask), dim=-1)
        past_length = cache.get_seq_length(index) if use_cache else 0
        layer_mask = _prepare_4d_causal_attention_mask(
            layer_padding_mask, hidden_states.shape[:2], hidden_states, past_length
        )
        layer_positions = position_ids
        if not is_prefill and model.compact_decode_positions:
            layer_positions = torch.full_like(position_ids, past_length)
        rotary = layer.self_attn.rotary_emb
        if rotary.max_seq_len_cached < input_position_limit:
            # A short pruned layer can retain positions beyond its initial
            # rotary cache even when its own key sequence is very short.
            required = int(layer_positions.max()) + 1
            if rotary.max_seq_len_cached < required:
                rotary._set_cos_sin_cache(
                    seq_len=required, device=hidden_states.device, dtype=hidden_states.dtype
                )
        # The saved mask describes tokens entering this layer, before its
        # possible pruning operation; those are exactly the cached keys.
        if is_prefill:
            model._layer_padding_masks.append(layer_padding_mask)
        else:
            model._layer_padding_masks[index] = layer_padding_mask
        should_prune = is_prefill and index + 1 in model.pruning_locations
        need_attentions = output_attentions or should_prune
        outputs = layer(
            hidden_states, attention_mask=layer_mask, position_ids=layer_positions,
            past_key_value=cache if use_cache else None,
            output_attentions=need_attentions, use_cache=use_cache,
        )
        hidden_states = outputs[0]
        if output_attentions:
            all_attentions += (outputs[1],)
        if should_prune:
            hidden_states, position_ids, current_mask, visual_length = model.prune_layer(
                index + 1, hidden_states, outputs[1], position_ids,
                current_mask, visual_length, state,
            )

    hidden_states = model.norm(hidden_states)
    if is_prefill:
        model.visual_token_num = sum(model.layer_visual_tokens) / len(model.layers)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = (cache.to_legacy_cache() if legacy else cache) if use_cache else None
    if not return_dict:
        return tuple(x for x in (hidden_states, next_cache, all_hidden_states, all_attentions) if x is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=next_cache,
        hidden_states=all_hidden_states, attentions=all_attentions,
    )
