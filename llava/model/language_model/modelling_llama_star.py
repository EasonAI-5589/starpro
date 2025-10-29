import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast


# STAR-FastV多层剪枝schedule (与STAR相同)
STAR_FASTV_SCHEDULE = {
    "7b": {
        192: [(2, 300), (8, 200), (16, 150), (24, 192)],
        128: [(2, 224), (8, 144), (16, 72), (24, 56)],
        64: [(2, 112), (8, 56), (16, 28), (24, 14)],
    },
    "13b": {
        192: [(2, 300), (10, 200), (20, 150), (30, 192)],
        128: [(2, 250), (10, 170), (20, 135), (30, 128)],
        64: [(2, 288), (10, 192), (20, 96), (30, 64)],
    }
}


class STARVLMModel(LlamaModel):
    """
    STAR-FastV: Multi-Layer Progressive Pruning with FastV-style Importance Scoring

    Core mechanism:
    1. Uses FastV's attention-based importance scoring (simpler than STAR's semantic blocks)
    2. Applies progressive pruning across multiple layers (using STAR's schedule)
    3. Simpler than full STAR: no semantic block discovery, just top-k selection

    Key differences:
    - vs STAR: No semantic block clustering, uses direct attention-based importance
    - vs FastV: Prunes progressively at multiple scheduled layers (not just once at layer K)
    """

    def __init__(self, config: LlamaConfig, starvlm_config: Dict):
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
        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880
            self.anyres = True
        else:
            self.visual_token_length = 576
            self.anyres = False

        # STAR-FastV config
        self.target_visual_tokens = starvlm_config["T"]

        # Load pruning schedule (same as STAR)
        self.pruning_schedule = STAR_FASTV_SCHEDULE[self.scale][self.target_visual_tokens]
        self.pruning_layers = {layer_idx: target for layer_idx, target in self.pruning_schedule}

        print(f"[STAR-FastV] Initialized with scale={self.scale}, target={self.target_visual_tokens}")
        print(f"[STAR-FastV] Pruning schedule: {self.pruning_schedule}")

        self.reset_state()

    def reset_state(self):
        """Reset internal state for new generation"""
        self.current_visual_length = None
        self.visual_token_indices = None
        self.prefill_done = False
        self.visual_token_num = 0

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

        if past_key_values is None:
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
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
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        # Initialize visual token tracking during prefill
        if seq_length > 1 and not self.prefill_done:
            visual_start = self.system_prompt_length
            visual_end = visual_start + self.visual_token_length

            if visual_end > seq_length:
                actual_visual_length = seq_length - visual_start
                visual_end = seq_length
            else:
                actual_visual_length = self.visual_token_length

            self.current_visual_length = actual_visual_length
            self.visual_token_indices = torch.arange(actual_visual_length, device=hidden_states.device)
            self.prefill_done = True
            print(f"[STAR-FastV] Prefill: visual tokens = {self.current_visual_length}")

        # Process layers with progressive pruning
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        visual_token_sum = 0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            # Apply FastV-style pruning at scheduled layers
            if seq_length > 1 and self.prefill_done and layer_idx in self.pruning_layers:
                target_visual_length = self.pruning_layers[layer_idx]

                if self.current_visual_length > target_visual_length:
                    visual_start = self.system_prompt_length
                    visual_end = visual_start + self.current_visual_length

                    print(f"\n[STAR-FastV] Layer {layer_idx}: Pruning {self.current_visual_length} → {target_visual_length}")

                    # Forward pass to get attention scores
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            attention_mask,
                            position_ids,
                            past_key_values,
                            True,
                            use_cache,
                        )
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )

                    hidden_states = layer_outputs[0]
                    layer_attention = layer_outputs[1]  # (B, num_heads, seq_len, seq_len)

                    # FastV's importance scoring
                    # Use the last text token's attention to visual tokens
                    # Average across heads: (B, seq_len, seq_len)
                    device = hidden_states.device
                    attn_avg = layer_attention.mean(dim=1)  # (B, seq_len, seq_len)

                    # Get last token's attention to visual region
                    # last_token_idx = -1 means the newest text token
                    visual_attention = attn_avg[0, -1, visual_start:visual_end]  # (N_vis,)

                    print(f"[STAR-FastV] Visual attention stats: mean={visual_attention.mean():.4f}, "
                          f"std={visual_attention.std():.4f}, max={visual_attention.max():.4f}")

                    # Select top-k important visual tokens (FastV-style)
                    keep_indices = torch.topk(visual_attention, k=target_visual_length).indices
                    keep_indices = keep_indices.sort().values  # Sort to maintain position order

                    print(f"[STAR-FastV] Selected {len(keep_indices)} tokens using last token attention")

                    # Update indices and hidden states
                    self.visual_token_indices = self.visual_token_indices[keep_indices]

                    visual_hidden = hidden_states[:, visual_start:visual_end]
                    new_visual = visual_hidden[:, keep_indices]

                    hidden_states = torch.cat([
                        hidden_states[:, :visual_start],
                        new_visual,
                        hidden_states[:, visual_end:]
                    ], dim=1)

                    if position_ids is not None:
                        new_positions = position_ids[:, visual_start:visual_end][:, keep_indices]
                        position_ids = torch.cat([
                            position_ids[:, :visual_start],
                            new_positions,
                            position_ids[:, visual_end:]
                        ], dim=1)

                    # Update current visual length
                    self.current_visual_length = len(keep_indices)

                    print(f"[STAR-FastV] Layer {layer_idx}: Pruned to {self.current_visual_length} tokens")

                    # Regenerate attention mask
                    new_seq_length = hidden_states.shape[1]
                    if attention_mask is not None:
                        attention_mask = torch.ones(
                            (batch_size, new_seq_length),
                            dtype=torch.bool,
                            device=hidden_states.device
                        )
                        if self._use_flash_attention_2:
                            attention_mask = None
                        elif self._use_sdpa and not output_attentions:
                            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                                attention_mask,
                                (batch_size, new_seq_length),
                                hidden_states,
                                past_key_values_length,
                            )
                        else:
                            attention_mask = _prepare_4d_causal_attention_mask(
                                attention_mask,
                                (batch_size, new_seq_length),
                                hidden_states,
                                past_key_values_length
                            )

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if True else 1]
                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

                    if seq_length > 1 and self.current_visual_length is not None:
                        visual_token_sum += self.current_visual_length

                    continue

            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_sum += self.current_visual_length

            # Normal layer forward
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
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

        hidden_states = self.norm(hidden_states)

        if seq_length > 1 and self.prefill_done:
            self.visual_token_num = visual_token_sum / len(self.layers)
            print(f"[STAR-FastV] Average visual tokens across layers: {self.visual_token_num:.1f}")

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
