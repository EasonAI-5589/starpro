import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast


# STAR-FastV Multi-layer pruning schedule
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

# STAR-V2 Two-Stage Pruning Schedule
# Stage 1 (llava_arch): 576 -> 288 or 2880 -> 1440 (50% via visual self-attention)
# Stage 2 (here): 288 -> target or 1440 -> target (text-guided progressive pruning)
# Goal: Average tokens across all layers ≈ target budget
STAR_V2_SCHEDULE = {
    "7b": {
        # For pad mode: Stage 1 gives 288 tokens (576/2)
        # 32 layers: (288*2 + X*6 + Y*8 + Z*8 + T*8) / 32 = target
        192: [(2, 240), (8, 216), (16, 192), (24, 192)],     # Avg ≈ 192
        128: [(2, 288), (8, 192), (16, 32), (24, 0)],        # Avg = 128.0 (aggressive pruning)
        64: [(2, 128), (8, 72), (16, 16), (24, 0)],          # Avg = 64.0 (very aggressive)
        32: [(2, 64), (8, 40), (16, 24), (24, 16)],          # Avg ≈ 32
    },
    "13b": {
        # For pad mode: Stage 1 gives 288 tokens (576/2)
        # 40 layers: (288*2 + X*8 + Y*10 + Z*10 + T*10) / 40 = target
        192: [(2, 240), (10, 216), (20, 192), (30, 192)],    # Avg ≈ 192
        128: [(2, 180), (10, 144), (20, 128), (30, 128)],    # Avg ≈ 128
        64: [(2, 120), (10, 80), (20, 48), (30, 32)],        # Avg ≈ 64
        32: [(2, 64), (10, 40), (20, 24), (30, 16)],         # Avg ≈ 32
    }
}

# STAR-V3 Two-Stage Pruning Schedule (Adaptive)
# Stage 1 (llava_arch): 576 -> target*2 (THCP pruning, adaptive to target)
# Stage 2 (here): target*2 -> target (text-guided progressive pruning)
# Goal: Average tokens across all layers = target budget (exact)
# Strategy: 3 pruning steps with front-heavy distribution (more tokens in early layers)
STAR_V3_SCHEDULE = {
    "7b": {
        # For pad mode: Stage 1 gives target*2 tokens
        # 32 layers: 8 segments × (t_init + t1 + t2 + t3) / 32 = target
        # Three pruning layers: 8 (25%), 16 (50%), 24 (75%)
        192: [(8, 192), (16, 144), (24, 96)],      # Stage 1: 384 → Avg = 192.0
        128: [(12, 64), (24, 32)],        # Avg = 128.0 (aggressive pruning)
        64: [(12, 32), (24, 16)],         # Stage 1: 128 → Avg = 64.0
        32: [(12, 16), (24, 8)],          # Stage 1: 64 → Avg = 32.0
    },
    "13b": {
        # For pad mode: Stage 1 gives target*2 tokens
        # 40 layers: 10 segments × (t_init + t1 + t2 + t3) / 40 = target
        # Three pruning layers: 10 (25%), 20 (50%), 30 (75%)
        192: [(15, 128), (30, 64)],                # Stage 1: 384 → Avg = 192.0
        128: [(15, 64), (30, 32)],                 # Stage 1: 256 → Avg = 128.0
        64: [(15, 32), (30, 16)],                  # Stage 1: 128 → Avg = 64.0
        32: [(15, 16), (30, 8)],                   # Stage 1: 64  → Avg = 32.0
    }
}

class STARVLMModel(LlamaModel):
    """
    STAR-VLM: Multi-Layer Progressive Pruning with Attention-based Importance Scoring

    Supports two modes:
    1. STAR (original): Single-stage pruning from original visual tokens
    2. STAR-V2 (two-stage):
       - Stage 1 (llava_arch): Visual self-attention pruning (50% reduction)
       - Stage 2 (here): Text-guided progressive pruning to target
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

        # Config
        self.target_visual_tokens = starvlm_config["T"]
        self.mode = starvlm_config.get("mode", "star")  # "star", "star_v2", or "star_v3"

        # Load pruning schedule based on mode
        if self.mode == "star_v3":
            # STAR-V3: Two-stage mode with adaptive schedule (Stage 1 gives target*2)
            self.visual_token_length = self.target_visual_tokens * 2
            self.pruning_schedule = STAR_V3_SCHEDULE[self.scale][self.target_visual_tokens]
            print(f"[STAR-V3 Stage 2] Initialized")
            print(f"  Scale: {self.scale}")
            print(f"  Expected input from Stage 1: {self.visual_token_length} tokens (target × 2)")
            print(f"  Stage 1 method: THCP (adaptive to target)")
            print(f"  Target tokens: {self.target_visual_tokens}")
        elif self.mode == "star_v2":
            # STAR-V2: Two-stage mode (Stage 1 gives 288 tokens, fixed 50%)
            self.visual_token_length = self.visual_token_length // 2
            self.pruning_schedule = STAR_V2_SCHEDULE[self.scale][self.target_visual_tokens]
            print(f"[STAR-V2 Stage 2] Initialized")
            print(f"  Scale: {self.scale}")
            print(f"  Expected input from Stage 1: {self.visual_token_length} tokens")
            print(f"  Stage 1 method: Self-similarity")
            print(f"  Target tokens: {self.target_visual_tokens}")
        else:
            # Original single-stage mode
            self.pruning_schedule = STAR_FASTV_SCHEDULE[self.scale][self.target_visual_tokens]
            print(f"[STAR-FastV] Initialized")
            print(f"  Scale: {self.scale}")
            print(f"  Target tokens: {self.target_visual_tokens}")

        self.pruning_layers = {layer_idx: target for layer_idx, target in self.pruning_schedule}
        print(f"  Pruning schedule: {self.pruning_schedule}")

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

            if self.mode == "star_v3":
                mode_name = "STAR-V3 Stage 2"
            elif self.mode == "star_v2":
                mode_name = "STAR-V2 Stage 2"
            else:
                mode_name = "STAR-FastV"
            print(f"[{mode_name}] Prefill: visual tokens = {self.current_visual_length}")

        # Process layers with progressive pruning
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        visual_token_sum = 0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            # Track visual token count for each layer (following PDrop/SparseVLM approach)
            if seq_length > 1 and self.prefill_done:
                visual_token_sum += self.current_visual_length

            # Apply text-guided pruning at scheduled layers
            if seq_length > 1 and self.prefill_done and layer_idx in self.pruning_layers:
                target_visual_length = self.pruning_layers[layer_idx]

                if self.current_visual_length > target_visual_length:
                    visual_start = self.system_prompt_length
                    visual_end = visual_start + self.current_visual_length

                    if self.mode == "star_v3":
                        mode_name = "STAR-V3 Stage 2"
                    elif self.mode == "star_v2":
                        mode_name = "STAR-V2 Stage 2"
                    else:
                        mode_name = "STAR-FastV"
                    print(f"\n[{mode_name}] Layer {layer_idx}: Pruning {self.current_visual_length} → {target_visual_length}")

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

                    device = hidden_states.device

                    # STAR-V2/V3: Multi-text-token guidance (inspired by SparseVLM)
                    # Unlike PDrop which only uses last token, we identify important text tokens
                    if self.mode in ["star_v2", "star_v3"]:
                        # Extract visual and text hidden states
                        visual_hidden = hidden_states[:, visual_start:visual_end]  # (B, N_vis, D)
                        text_hidden = hidden_states[:, visual_end:]  # (B, N_text, D)

                        # Compute text-visual similarity matrix to find important text tokens
                        # (B, N_text, D) @ (B, D, N_vis) -> (B, N_text, N_vis)
                        text_visual_sim = torch.matmul(text_hidden, visual_hidden.transpose(1, 2))
                        text_visual_sim = text_visual_sim.squeeze(0)  # (N_text, N_vis)

                        # Identify text tokens that attend strongly to visual tokens
                        # Average similarity per text token
                        text_importance = text_visual_sim.softmax(dim=0).mean(dim=1)  # (N_text,)

                        # Select text raters: tokens with above-average importance
                        text_rater_mask = text_importance > text_importance.mean()
                        text_rater_indices = torch.where(text_rater_mask)[0]

                        if len(text_rater_indices) == 0:
                            # Fallback: use top 50% if no tokens above mean
                            num_raters = max(1, len(text_importance) // 2)
                            text_rater_indices = text_importance.topk(num_raters).indices

                        print(f"[{mode_name}] Using {len(text_rater_indices)} text rater tokens (out of {len(text_importance)} text tokens)")

                        # Aggregate attention from text raters to visual tokens
                        # Average across heads: (B, seq_len, seq_len)
                        attn_avg = layer_attention.mean(dim=1)  # (B, seq_len, seq_len)

                        # Get attention from text raters to visual region
                        # Offset text_rater_indices to account for visual tokens
                        text_rater_positions = text_rater_indices + visual_end
                        rater_to_visual_attn = attn_avg[0, text_rater_positions, visual_start:visual_end]  # (N_raters, N_vis)

                        # Aggregate: mean across text raters
                        visual_attention = rater_to_visual_attn.mean(dim=0)  # (N_vis,)

                        print(f"[{mode_name}] Multi-token text guidance - Visual attention stats: "
                              f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                              f"max={visual_attention.max():.4f}")
                    else:
                        # STAR (original): Use last token attention (PDrop-style)
                        attn_avg = layer_attention.mean(dim=1)  # (B, seq_len, seq_len)
                        visual_attention = attn_avg[0, -1, visual_start:visual_end]  # (N_vis,)

                        print(f"[{mode_name}] Single-token text guidance - Visual attention stats: "
                              f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                              f"max={visual_attention.max():.4f}")

                    # Select top-k important visual tokens (text-guided)
                    keep_indices = torch.topk(visual_attention, k=target_visual_length).indices
                    keep_indices = keep_indices.sort().values  # Sort to maintain position order

                    print(f"[{mode_name}] Selected {len(keep_indices)} tokens using text-to-visual attention")

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

                    if attention_mask is not None and attention_mask.dim() == 4:
                        new_mask_visual = attention_mask[:, :, :, visual_start:visual_end][:, :, :, keep_indices]
                        attention_mask = torch.cat([
                            attention_mask[:, :, :, :visual_start],
                            new_mask_visual,
                            attention_mask[:, :, :, visual_end:]
                        ], dim=3)

                        new_mask_query = attention_mask[:, :, visual_start:visual_end, :][:, :, keep_indices, :]
                        attention_mask = torch.cat([
                            attention_mask[:, :, :visual_start, :],
                            new_mask_query,
                            attention_mask[:, :, visual_end:, :]
                        ], dim=2)

                    self.current_visual_length = target_visual_length

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    continue

            # Regular layer forward pass (no pruning)
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

        if seq_length > 1:
            # Calculate average visual tokens across all layers (following PDrop/SparseVLM)
            self.visual_token_num = visual_token_sum / len(self.layers) if self.layers else 0

        # Add hidden states from the last decoder layer
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
