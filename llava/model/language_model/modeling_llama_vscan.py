"""
VScan Stage 2: Middle Layer Pruning for LLaVA

Reference: https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan

VScan is a training-free visual token reduction framework that uses:
- Stage 1 (in llava_arch.py): Complementary global and local scans
- Stage 2 (this file): Middle layer pruning based on attention scores

Default configuration for LLaVA-1.5-7B:
- Stage 1: 576 -> 96 tokens (complementary scanning)
- Stage 2: 96 -> 32 tokens at layer 16 (attention-based pruning)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel, LlamaForCausalLM, Cache, DynamicCache,
    apply_rotary_pos_emb, LlamaRMSNorm, LlamaMLP, LlamaDecoderLayer, LlamaPreTrainedModel
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from llava.constants import IGNORE_INDEX
import os


# VScan default pruning configuration
# Format: {model_scale: {stage1_tokens: [(prune_layer, target_tokens), ...]}}
VSCAN_SCHEDULE = {
    "7b": {
        # Stage 1 output -> Stage 2 target
        96: [(16, 32)],   # Default: 96 -> 32 at layer 16
        128: [(16, 48)],  # 128 -> 48 at layer 16
        192: [(16, 64)],  # 192 -> 64 at layer 16
    },
    "13b": {
        96: [(20, 32)],   # 96 -> 32 at layer 20
        128: [(20, 48)],  # 128 -> 48 at layer 20
        192: [(20, 64)],  # 192 -> 64 at layer 20
    }
}


class VScanLlamaModel(LlamaModel):
    """
    VScan-enhanced LlamaModel with middle layer token pruning.

    Key features:
    - Prunes visual tokens at a specific middle layer (default: layer 16 for 7B)
    - Uses attention scores from prompt tokens to visual tokens for importance ranking
    - Query: last token of the prompt (text-guided selection)
    - Maintains KV-cache compatibility for efficient decoding
    """

    def __init__(self, config: LlamaConfig, vscan_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0

        # Model scale detection
        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"
        else:
            raise ValueError(f"Unsupported model scale with {config.num_hidden_layers} layers")

        # Visual token configuration
        if getattr(config, 'image_aspect_ratio', 'pad') == "pad":
            self.base_visual_tokens = 576
            self.anyres = False
        else:
            self.base_visual_tokens = 2880
            self.anyres = True

        # VScan configuration
        self.stage1_tokens = vscan_config.get("stage1_tokens", 96)  # Output from Stage 1
        self.stage2_tokens = vscan_config.get("stage2_tokens", 32)  # Final target
        self.prune_layer = vscan_config.get("prune_layer", 16 if self.scale == "7b" else 20)

        # Visual token length expected after Stage 1
        self.visual_token_length = self.stage1_tokens

        # Get pruning schedule
        if self.stage1_tokens in VSCAN_SCHEDULE[self.scale]:
            self.pruning_schedule = VSCAN_SCHEDULE[self.scale][self.stage1_tokens]
        else:
            # Default: single layer pruning at specified layer
            self.pruning_schedule = [(self.prune_layer, self.stage2_tokens)]

        enable_debug = os.environ.get('ENABLE_DEBUG', '0') == '1'
        if enable_debug:
            print(f"\n{'='*80}")
            print(f"[VScan Stage 2] Initialized")
            print(f"  Scale: {self.scale} ({config.num_hidden_layers} layers)")
            print(f"  Stage 1 output: {self.stage1_tokens} tokens")
            print(f"  Stage 2 target: {self.stage2_tokens} tokens")
            print(f"  Prune layer: {self.prune_layer}")
            print(f"  Pruning schedule: {self.pruning_schedule}")
            print(f"{'='*80}\n")

        self.reset_state()

    def reset_state(self):
        """Reset internal state for new generation"""
        self.current_visual_length = None
        self.visual_start_idx = None
        self.visual_end_idx = None
        self.prefill_done = False
        self.visual_token_num = 0

    def _compute_attention_scores(self, hidden_states, layer_idx, position_ids):
        """
        Compute attention scores for visual token importance ranking.

        Args:
            hidden_states: (B, L, D) current hidden states
            layer_idx: which layer to use for attention computation
            position_ids: position ids for RoPE

        Returns:
            attention_scores: (B, N_visual) importance scores for visual tokens
        """
        layer = self.layers[layer_idx]
        B, L, D = hidden_states.shape

        # Get Q, K projections
        hidden_norm = layer.input_layernorm(hidden_states)

        # Compute Q and K
        query_states = layer.self_attn.q_proj(hidden_norm)
        key_states = layer.self_attn.k_proj(hidden_norm)

        # Reshape for multi-head attention
        num_heads = self.config.num_attention_heads
        head_dim = D // num_heads

        query_states = query_states.view(B, L, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(B, L, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = layer.self_attn.rotary_emb(key_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute attention scores
        # Use last text token as query (text-guided selection)
        query_last = query_states[:, :, -1:, :]  # (B, num_heads, 1, head_dim)

        # Keys for visual tokens only
        visual_keys = key_states[:, :, self.visual_start_idx:self.visual_end_idx, :]

        # Scaled dot-product attention scores
        attn_scores = torch.matmul(query_last, visual_keys.transpose(-1, -2))
        attn_scores = attn_scores / (head_dim ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # Average across heads
        attn_scores = attn_scores.mean(dim=1).squeeze(1)  # (B, N_visual)

        return attn_scores

    def _prune_visual_tokens(self, hidden_states, attention_mask, position_ids,
                             past_key_values, target_tokens, layer_idx):
        """
        Prune visual tokens based on attention scores.

        Args:
            hidden_states: (B, L, D)
            attention_mask: (B, 1, L, L) or similar
            position_ids: (B, L)
            past_key_values: KV cache
            target_tokens: number of visual tokens to keep
            layer_idx: current layer index

        Returns:
            Pruned hidden_states, attention_mask, position_ids, past_key_values
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device

        # Compute attention-based importance scores
        attn_scores = self._compute_attention_scores(hidden_states, layer_idx, position_ids)

        # Select top-k visual tokens
        current_visual_tokens = self.visual_end_idx - self.visual_start_idx
        keep_num = min(target_tokens, current_visual_tokens)

        # Get indices of tokens to keep (sorted by position to maintain order)
        _, top_indices = attn_scores.topk(keep_num, dim=-1)
        top_indices = top_indices.sort(dim=-1).values  # (B, keep_num)

        # Build new sequence
        # [system_prompt | selected_visual | text_tokens]
        new_seq_len = self.visual_start_idx + keep_num + (L - self.visual_end_idx)

        new_hidden = torch.zeros(B, new_seq_len, D, device=device, dtype=hidden_states.dtype)
        new_position_ids = torch.zeros(B, new_seq_len, device=device, dtype=position_ids.dtype)

        for b in range(B):
            # System prompt
            new_hidden[b, :self.visual_start_idx] = hidden_states[b, :self.visual_start_idx]
            new_position_ids[b, :self.visual_start_idx] = position_ids[b, :self.visual_start_idx]

            # Selected visual tokens
            visual_indices = top_indices[b] + self.visual_start_idx
            new_hidden[b, self.visual_start_idx:self.visual_start_idx + keep_num] = hidden_states[b, visual_indices]
            new_position_ids[b, self.visual_start_idx:self.visual_start_idx + keep_num] = position_ids[b, visual_indices]

            # Text tokens
            text_start_new = self.visual_start_idx + keep_num
            new_hidden[b, text_start_new:] = hidden_states[b, self.visual_end_idx:]
            new_position_ids[b, text_start_new:] = position_ids[b, self.visual_end_idx:]

        # Update attention mask
        new_attention_mask = torch.ones(B, 1, new_seq_len, new_seq_len, device=device, dtype=hidden_states.dtype)
        new_attention_mask = torch.tril(new_attention_mask)
        # Convert to causal mask format (0 for attend, -inf for mask)
        new_attention_mask = (1.0 - new_attention_mask) * torch.finfo(hidden_states.dtype).min

        # Update KV cache if present
        if past_key_values is not None:
            new_past_key_values = []
            for layer_kv in past_key_values:
                if layer_kv is not None and len(layer_kv) == 2:
                    k, v = layer_kv
                    # Build new KV with selected visual tokens
                    new_k = torch.zeros(B, k.shape[1], new_seq_len, k.shape[3], device=device, dtype=k.dtype)
                    new_v = torch.zeros(B, v.shape[1], new_seq_len, v.shape[3], device=device, dtype=v.dtype)

                    for b in range(B):
                        # System prompt
                        new_k[b, :, :self.visual_start_idx] = k[b, :, :self.visual_start_idx]
                        new_v[b, :, :self.visual_start_idx] = v[b, :, :self.visual_start_idx]

                        # Selected visual tokens
                        visual_indices = top_indices[b] + self.visual_start_idx
                        new_k[b, :, self.visual_start_idx:self.visual_start_idx + keep_num] = k[b, :, visual_indices]
                        new_v[b, :, self.visual_start_idx:self.visual_start_idx + keep_num] = v[b, :, visual_indices]

                        # Text tokens
                        text_start_new = self.visual_start_idx + keep_num
                        new_k[b, :, text_start_new:] = k[b, :, self.visual_end_idx:]
                        new_v[b, :, text_start_new:] = v[b, :, self.visual_end_idx:]

                    new_past_key_values.append((new_k, new_v))
                else:
                    new_past_key_values.append(layer_kv)
            past_key_values = tuple(new_past_key_values)

        # Update visual token tracking
        self.visual_end_idx = self.visual_start_idx + keep_num
        # Note: visual_token_num is set to AVERAGE in builder.py, don't override

        enable_debug = os.environ.get('ENABLE_DEBUG', '0') == '1'
        if enable_debug:
            print(f"[VScan] Layer {layer_idx}: Pruned {current_visual_tokens} -> {keep_num} visual tokens")

        return new_hidden, new_attention_mask, new_position_ids, past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        enable_debug = os.environ.get('ENABLE_DEBUG', '0') == '1'

        # Reset state for new prefill
        if past_key_values is None:
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # Initialize visual token positions on first prefill
        if past_key_values_length == 0 and self.visual_start_idx is None:
            self.visual_start_idx = self.system_prompt_length
            self.visual_end_idx = self.visual_start_idx + self.visual_token_length
            self.current_visual_length = self.visual_token_length
            # Note: visual_token_num is set to AVERAGE in builder.py, don't override

            if enable_debug:
                print(f"[VScan] Initialized: visual tokens at [{self.visual_start_idx}, {self.visual_end_idx})")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Check if we should prune at this layer (only during prefill)
            is_prefill = past_key_values_length == 0 and seq_length > 1
            should_prune = False
            target_tokens = None

            if is_prefill and not self.prefill_done:
                for prune_layer, target in self.pruning_schedule:
                    if idx == prune_layer:
                        should_prune = True
                        target_tokens = target
                        break

            # Prune visual tokens if needed
            if should_prune:
                hidden_states, attention_mask, position_ids, layer_past = self._prune_visual_tokens(
                    hidden_states, attention_mask, position_ids,
                    past_key_values.to_legacy_cache() if past_key_values else None,
                    target_tokens, idx
                )
                # Update past_key_values
                if layer_past is not None:
                    past_key_values = DynamicCache.from_legacy_cache(layer_past)

            # Forward through layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Mark prefill as done
        if past_key_values_length == 0:
            self.prefill_done = True

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare 4D causal attention mask."""
        # Create causal mask
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @staticmethod
    def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
        """Make causal mask for self-attention."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask, dtype, tgt_len=None):
        """Expand attention mask from (bsz, src_len) to (bsz, 1, tgt_len, src_len)."""
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def forward_x(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        selected_indices: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass with middle-layer token pruning (VScan Stage 2).

        Reference: Official VScan modeling_llama_x.py:forward_x

        This method performs layer-by-layer forward pass and prunes visual tokens
        at specified layers (self.layer_list) to target counts (self.image_token_list).
        """
        enable_debug = os.environ.get('ENABLE_DEBUG', '0') == '1'

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True  # Need hidden states for pruning
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Cache handling - match official VScan implementation
        # Reference: Official VScan modeling_llama_x.py:1152-1156
        past_key_values_length = 0
        use_legacy_cache = False
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            # CRITICAL: Always convert to DynamicCache, even if None
            # This ensures decoder_layer returns proper cache format
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare attention mask
        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Forward through layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Check if we should prune at this layer (VScan Stage 2)
            rank_layer = layer_idx + 1
            if hasattr(self, 'layer_list') and rank_layer in self.layer_list:
                if hidden_states.shape[1] != 1:  # Only during prefill
                    stage = self.layer_list.index(rank_layer)
                    (
                        position_ids,
                        attention_mask,
                        hidden_states,
                        labels
                    ) = self.layer_prune(
                        cur_num=stage,
                        rank_layer=rank_layer,
                        features=hidden_states,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        selected_indices=selected_indices,
                    )

                    # Recompute attention mask after pruning
                    if self._use_flash_attention_2:
                        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                    elif self._use_sdpa and not output_attentions:
                        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                            attention_mask,
                            (batch_size, hidden_states.shape[1]),
                            hidden_states,
                            past_key_values_length,
                        )
                    else:
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask, (batch_size, hidden_states.shape[1]), hidden_states, past_key_values_length
                        )

                    if enable_debug:
                        vtn_before = self.image_token_list[stage]
                        vtn_after = self.image_token_list[stage + 1]
                        print(f"[VScan] Layer {rank_layer}: vtn {vtn_before} -> {vtn_after}")
                else:
                    # Update position_ids in decoding stage
                    # Reference: Official VScan modeling_llama_x.py:1284-1293
                    stage = self.layer_list.index(rank_layer)
                    cur_visual_length = [min(int(self.image_token_list[stage]), cur_image_token) for cur_image_token in self.image_tokens]
                    next_visual_length = [min(int(self.image_token_list[stage + 1]), cur_image_token) for cur_image_token in self.image_tokens]

                    # Adjust position_ids for pruned visual tokens (keep as tensor)
                    offset = cur_visual_length[0] - next_visual_length[0]
                    position_ids = position_ids - offset

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None), labels

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), labels

    def layer_prune(
        self, cur_num, rank_layer, features,
        position_ids, attention_mask, labels, selected_indices
    ):
        """
        Prune visual tokens at a specific layer based on attention scores.

        Reference: Official VScan modeling_llama_x.py:layer_prune

        Args:
            cur_num: Current pruning stage index
            rank_layer: Layer index for pruning
            features: Hidden states (B, L, D)
            position_ids: Position IDs
            attention_mask: Attention mask
            labels: Labels for training
            selected_indices: Pre-selected indices (optional)

        Returns:
            Updated position_ids, attention_mask, features, labels
        """
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)

        batch_size = features.shape[0]

        # Get current and target token counts
        image_tokens = [min(int(self.image_token_list[cur_num]), cur_image_token) for cur_image_token in self.image_tokens]
        keep_length = [min(int(self.image_token_list[cur_num + 1]), cur_image_token) for cur_image_token in self.image_tokens]

        features_list = []
        attention_mask_list = []
        labels_list = []

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, features.shape[1]), dtype=torch.bool, device=features.device)
        else:
            attention_mask = attention_mask.bool()

        if labels is None:
            labels = torch.full((batch_size, features.shape[1]), IGNORE_INDEX, device=features.device)

        # Get Q and K for attention computation
        hidden_states = features.clone().detach()
        self_attn = self.layers[rank_layer].self_attn
        hidden_states = self.layers[rank_layer].input_layernorm(hidden_states)

        num_heads = self_attn.num_heads
        num_key_value_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        bsz, q_len, _ = hidden_states.size()

        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self_attn.rotary_emb(key_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Prepare attention mask for scoring
        eager_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, q_len), hidden_states, past_key_values_length=0
        ).to(device=query_states.device)

        # Take valid features
        features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        attention_mask = [cur_attention_mask[cur_attention_mask] for cur_attention_mask in attention_mask]

        # Rank and drop for each sample
        for i in range(batch_size):
            image_index = self.image_token_posi[i]

            if image_index == -1:
                # No image tokens
                features_list.append(features[i])
                attention_mask_list.append(attention_mask[i])
                labels_list.append(labels[i])
                continue

            cur_key_states = key_states[i]
            cur_query_states = query_states[i]
            cur_eager_attention_mask = eager_attention_mask[i]

            # Use last instruction token as query
            if self.training:
                answer_index = torch.where(labels[i] != IGNORE_INDEX)[0].tolist()
                index_before_answer = []
                for index in answer_index:
                    if labels[i][index - 1] == IGNORE_INDEX:
                        index_before_answer.append(index - 1)
                if not index_before_answer:
                    features_list.append(features[i])
                    attention_mask_list.append(attention_mask[i])
                    labels_list.append(labels[i])
                    continue

                index_before_answer = torch.tensor(index_before_answer, device=labels[0].device)
                text_query_states = cur_query_states[:, index_before_answer, :]
                text_eager_attention_mask = cur_eager_attention_mask[:, index_before_answer, :]
            else:
                prompt_total_len = self.prompt_len[i] + image_tokens[i]
                text_query_states = cur_query_states[:, prompt_total_len - 1, :].unsqueeze(1)
                text_eager_attention_mask = cur_eager_attention_mask[:, prompt_total_len - 1, :].unsqueeze(1)

            # Calculate attention scores
            attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + text_eager_attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Average across heads and select visual token attention
            attention_avg_head = torch.mean(attn_weights, dim=0)
            attention_avg_head = attention_avg_head[:, image_index:image_index + image_tokens[i]]
            attention_avg_text = torch.mean(attention_avg_head, dim=0)

            # Rank and drop by attention score
            top_rank_index = attention_avg_text.topk(keep_length[i]).indices
            top_rank_index = top_rank_index + image_index
            top_rank_index = top_rank_index.sort().values

            start_index = image_index + image_tokens[i]
            new_input_embeds = torch.cat([
                features[i][:image_index, :],
                features[i][top_rank_index, :],
                features[i][start_index:, :]
            ], dim=0)
            new_labels = torch.cat([
                labels[i][:image_index],
                labels[i][top_rank_index],
                labels[i][start_index:]
            ], dim=0)
            new_attention_mask = torch.cat([
                attention_mask[i][:image_index],
                attention_mask[i][top_rank_index],
                attention_mask[i][start_index:]
            ], dim=0)

            features_list.append(new_input_embeds)
            attention_mask_list.append(new_attention_mask)
            labels_list.append(new_labels)

        # Note: visual_token_num is set to AVERAGE in _setup_vscan_model_params
        # Don't update it here to keep the average value for display

        # Truncate to max length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]
            new_attention_mask = [x[:tokenizer_model_max_length] for x in attention_mask_list]
            new_labels = [x[:tokenizer_model_max_length] for x in labels_list]

        max_len = max(x.shape[0] for x in new_input_embeds)

        # Pad sequences to form batch
        embeds_padded = []
        labels_padded = []
        attention_mask_padded = []
        position_ids = torch.zeros((batch_size, max_len), dtype=_position_ids.dtype if _position_ids is not None else torch.long, device=features_list[0].device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len_emb = cur_new_embed.shape[0]
            dif = max_len - cur_len_emb

            cur_new_embed = torch.cat([
                cur_new_embed,
                torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ], dim=0)
            cur_new_labels = torch.cat([
                cur_new_labels,
                torch.full((dif,), IGNORE_INDEX, dtype=cur_new_labels.dtype, device=cur_new_labels.device)
            ], dim=0)
            cur_attention_mask = new_attention_mask[i]
            cur_attention_mask = torch.cat([
                cur_attention_mask,
                torch.full((dif,), False, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)
            ], dim=0)

            embeds_padded.append(cur_new_embed)
            labels_padded.append(cur_new_labels)
            attention_mask_padded.append(cur_attention_mask)

            cur_len = new_attention_mask[i].sum().item()
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(embeds_padded, dim=0)
        new_input_embeds = new_input_embeds.to(features_list[0].dtype)
        new_attention_mask = torch.stack(attention_mask_padded, dim=0)
        new_labels = torch.stack(labels_padded, dim=0)

        if _position_ids is None:
            position_ids = None
        if _labels is None:
            new_labels = None
        if _attention_mask is None:
            new_attention_mask = None
        else:
            new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)

        return position_ids, new_attention_mask, new_input_embeds, new_labels


class VScanLlamaForCausalLM(LlamaForCausalLM):
    """
    VScan-enhanced LlamaForCausalLM for visual token pruning.

    This class wraps VScanLlamaModel and provides the standard
    generate() interface for inference.
    """

    def __init__(self, config: LlamaConfig, vscan_config: Dict = None):
        # Initialize parent without model first
        super(LlamaForCausalLM, self).__init__(config)

        # Default VScan config
        if vscan_config is None:
            vscan_config = {
                "stage1_tokens": 96,
                "stage2_tokens": 32,
                "prune_layer": 16,
            }

        # Replace model with VScan-enhanced version
        self.model = VScanLlamaModel(config, vscan_config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
