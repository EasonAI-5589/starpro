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
# Strategy: 2-3 pruning steps with front-heavy distribution (more tokens in early layers)
# Note: For anyres mode, target is automatically multiplied by 5 in __init__
STAR_V3_SCHEDULE = {
    "7b": {
        # Pad mode targets (user passes T, actual target = T)
        192: [(8, 192), (16, 144), (24, 96)],      # Stage 1: 384 → Avg = 192.0
        128: [(12, 64), (24, 32)],                 # Stage 1: 256 → Avg = 128.0 (aggressive pruning)
        64: [(12, 32), (24, 16)],                  # Stage 1: 128 → Avg = 64.0
        32: [(12, 16), (24, 8)],                   # Stage 1: 64 → Avg = 32.0
        # Anyres mode targets (user passes T, actual target = T*5)
        960: [(8, 960), (16, 720), (24, 480)],     # Stage 1: 1920 → Avg = 960.0 (user T=192)
        640: [(12, 320), (24, 160)],               # Stage 1: 1280 → Avg = 640.0 (user T=128)
        320: [(12, 160), (24, 80)],                # Stage 1: 640 → Avg = 320.0 (user T=64)
        160: [(12, 80), (24, 40)],                 # Stage 1: 320 → Avg = 160.0 (user T=32)
    },
    "13b": {
        # Pad mode targets (user passes T, actual target = T)
        192: [(15, 128), (30, 64)],                # Stage 1: 384 → Avg = 192.0
        128: [(15, 64), (30, 32)],                 # Stage 1: 256 → Avg = 128.0
        64: [(15, 32), (30, 16)],                  # Stage 1: 128 → Avg = 64.0
        32: [(15, 16), (30, 8)],                   # Stage 1: 64  → Avg = 32.0
        # Anyres mode targets (user passes T, actual target = T*5)
        960: [(15, 640), (30, 320)],               # Stage 1: 1920 → Avg = 960.0 (user T=192)
        640: [(15, 320), (30, 160)],               # Stage 1: 1280 → Avg = 640.0 (user T=128)
        320: [(15, 160), (30, 80)],                # Stage 1: 640 → Avg = 320.0 (user T=64)
        160: [(15, 80), (30, 40)],                 # Stage 1: 320 → Avg = 160.0 (user T=32)
    }
}

# STAR-V5 Schedule (S2 Only - Progressive Pruning Only)
# Stage 1 (llava_arch): Skip THCP, keep all 576 tokens
# Stage 2 (here): 576 -> target (text-guided progressive pruning)
# Goal: Isolate Stage 2 contribution for ablation study
# Note: Aligned with SparseVLM's pruning layers (2, 6, 15) for fair comparison
STAR_V5_SCHEDULE = {
    "7b": {
        # Pad mode targets (user passes T, actual target = T)
        # Initial: 576 tokens (no Stage 1 THCP)
        # Pruning at layers 2, 6, 15 (same as SparseVLM)
        192: [(2, 300), (6, 200), (15, 110)],      # Initial: 576 → Avg = 188.19
        # Verify: (576*2 + 300*4 + 200*9 + 110*17) / 32 = (1152 + 1200 + 1800 + 1870) / 32 = 188.19
        128: [(2, 300), (6, 110), (15, 44)],       # Initial: 576 → Avg = 128.00
        # Verify: (576*2 + 300*4 + 110*9 + 44*17) / 32 = (1152 + 1200 + 990 + 748) / 32 = 128.00
        64: [(2, 65), (6, 30), (15, 15)],          # Initial: 576 → Avg = 60.53
        # Verify: (576*2 + 65*4 + 30*9 + 15*17) / 32 = (1152 + 260 + 270 + 255) / 32 = 60.53
        32: [(2, 40), (6, 20), (15, 10)],          # Initial: 576 → Avg = 31.88
        # Verify: (576*2 + 40*4 + 20*9 + 10*17) / 32 = (1152 + 160 + 180 + 170) / 32 = 31.88

        # Anyres mode targets (user passes T, actual target = T*5)
        # Initial: 2880 tokens (no Stage 1 THCP)
        960: [(2, 1500), (6, 1000), (15, 550)],    # Initial: 2880 → Avg = 940.94 (user T=192)
        # Verify: (2880*2 + 1500*4 + 1000*9 + 550*17) / 32 = 940.94
        640: [(2, 1500), (6, 550), (15, 175)],     # Initial: 2880 → Avg = 615.16 (user T=128)
        # Verify: (2880*2 + 1500*4 + 550*9 + 175*17) / 32 = 615.16
        320: [(2, 325), (6, 150), (15, 75)],       # Initial: 2880 → Avg = 302.66 (user T=64)
        # Verify: (2880*2 + 325*4 + 150*9 + 75*17) / 32 = 302.66
        160: [(2, 200), (6, 100), (15, 50)],       # Initial: 2880 → Avg = 159.38 (user T=32)
        # Verify: (2880*2 + 200*4 + 100*9 + 50*17) / 32 = 159.38
    },
    "13b": {
        # Pad mode targets (40 layers for 13B model)
        # Initial: 576 tokens (no Stage 1 THCP)
        # Pruning at layers 2, 8, 20 (same as SparseVLM for 13B)
        192: [(2, 300), (8, 200), (20, 110)],      # Initial: 576 → Avg = 188.50
        # Verify: (576*2 + 300*6 + 200*12 + 110*20) / 40 = (1152 + 1800 + 2400 + 2200) / 40 = 188.50
        128: [(2, 300), (8, 100), (20, 48)],       # Initial: 576 → Avg = 128.00
        # Verify: (576*2 + 300*6 + 100*12 + 48*20) / 40 = (1152 + 1800 + 1200 + 960) / 40 = 128.00
        64: [(2, 70), (8, 35), (20, 20)],          # Initial: 576 → Avg = 64.00
        # Verify: (576*2 + 70*6 + 35*12 + 20*20) / 40 = (1152 + 420 + 420 + 400) / 40 = 64.00
        32: [(2, 40), (8, 20), (20, 10)],          # Initial: 576 → Avg = 33.50
        # Verify: (576*2 + 40*6 + 20*12 + 10*20) / 40 = (1152 + 240 + 240 + 200) / 40 = 33.50

        # Anyres mode targets (user passes T, actual target = T*5)
        # Initial: 2880 tokens (no Stage 1 THCP)
        960: [(2, 1500), (8, 1000), (20, 550)],    # Initial: 2880 → Avg = 942.50 (user T=192)
        # Verify: (2880*2 + 1500*6 + 1000*12 + 550*20) / 40 = 942.50
        640: [(2, 1500), (8, 500), (20, 200)],     # Initial: 2880 → Avg = 632.50 (user T=128)
        # Verify: (2880*2 + 1500*6 + 500*12 + 200*20) / 40 = 632.50
        320: [(2, 350), (8, 175), (20, 100)],      # Initial: 2880 → Avg = 320.00 (user T=64)
        # Verify: (2880*2 + 350*6 + 175*12 + 100*20) / 40 = 320.00
        160: [(2, 200), (8, 100), (20, 50)],       # Initial: 2880 → Avg = 167.50 (user T=32)
        # Verify: (2880*2 + 200*6 + 100*12 + 50*20) / 40 = 167.50
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
            # Note: User should pass T=128 for pad, T=640 for anyres (manually adjusted)
            self.visual_token_length = self.target_visual_tokens * 2
            self.pruning_schedule = STAR_V3_SCHEDULE[self.scale][self.target_visual_tokens]
            print(f"[STAR-V3 Stage 2] Initialized")
            print(f"  Scale: {self.scale}")
            print(f"  Aspect ratio: {'anyres' if self.anyres else 'pad'}")
            print(f"  User passed T: {starvlm_config['T']}")
            print(f"  Expected input from Stage 1: {self.visual_token_length} tokens (T × 2)")
            print(f"  Stage 1 method: THCP (adaptive to target)")
            print(f"  Target tokens: {self.target_visual_tokens}")
        elif self.mode == "star_v5":
            # STAR-V5: S2 Only ablation (Stage 1 skipped, progressive pruning only)
            # Stage 1 (llava_arch) skips THCP and keeps all 576 (or 2880) tokens
            # Stage 2 (here) progressively prunes from 576 -> target
            # Keep visual_token_length at full size (576 or 2880)
            # visual_token_length is already set above based on aspect_ratio
            self.pruning_schedule = STAR_V5_SCHEDULE[self.scale][self.target_visual_tokens]
            print(f"\n{'='*80}")
            print(f"[STAR-V5 Stage 2 Only] Initialized (Ablation: S2 Only)")
            print(f"  Scale: {self.scale} ({config.num_hidden_layers} layers)")
            print(f"  Aspect ratio: {'anyres' if self.anyres else 'pad'}")
            print(f"  User passed T: {starvlm_config['T']}")
            print(f"  Expected input from Stage 1: {self.visual_token_length} tokens (all tokens, no THCP)")
            print(f"  Stage 1: SKIPPED (no THCP)")
            print(f"  Stage 2: Progressive pruning {self.visual_token_length} → {self.target_visual_tokens}")
            print(f"  Pruning schedule: {self.pruning_schedule}")
            print(f"{'='*80}\n")
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
        self.layer_visual_tokens = []  # Track visual tokens per layer for FLOPS calculation

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

        # ============ DEBUG: 追踪序列长度来源 ============
        if past_key_values is None:  # 只在 prefill 阶段打印
            print("\n" + "="*80)
            print("🔍 [STAR-V3 DEBUG] Forward Input Analysis")
            print("="*80)
            
            # 1. 检查输入来源
            if input_ids is not None:
                print(f"✓ Input via: input_ids")
                print(f"  - Shape: {input_ids.shape}")
                print(f"  - Batch size: {input_ids.shape[0]}")
                print(f"  - Sequence length: {input_ids.shape[1]}")
                seq_length = input_ids.shape[1]
            elif inputs_embeds is not None:
                print(f"✓ Input via: inputs_embeds")
                print(f"  - Shape: {inputs_embeds.shape}")
                print(f"  - Batch size: {inputs_embeds.shape[0]}")
                print(f"  - Sequence length: {inputs_embeds.shape[1]}")
                seq_length = inputs_embeds.shape[1]
            else:
                print("✗ No input provided!")
                seq_length = 0
            
            # 2. 分析序列组成
            print(f"\n📊 Sequence Structure Analysis:")
            print(f"  Total length: {seq_length}")
            print(f"  ├─ System prompt: 0-{self.system_prompt_length-1} ({self.system_prompt_length} tokens)")
            print(f"  ├─ Visual tokens: {self.system_prompt_length}-{self.system_prompt_length + self.visual_token_length - 1} ({self.visual_token_length} tokens)")
            print(f"  └─ Text tokens: {self.system_prompt_length + self.visual_token_length}-{seq_length-1} ({seq_length - self.system_prompt_length - self.visual_token_length} tokens)")
            
            # 3. 检查是否合理
            expected_min_length = self.system_prompt_length + self.visual_token_length
            if seq_length < expected_min_length:
                print(f"\n⚠️  WARNING: Sequence too short! Expected at least {expected_min_length}, got {seq_length}")
            elif seq_length - expected_min_length > 500:
                print(f"\n⚠️  WARNING: Text tokens unusually long ({seq_length - expected_min_length} tokens)!")
                print(f"    This might include conversation history or long context.")
            
            # 4. 如果有 input_ids，尝试查看内容
            if input_ids is not None:
                print(f"\n📝 Token ID Samples:")
                print(f"  System tokens [0:10]: {input_ids[0, :10].tolist()}")
                print(f"  Visual tokens [35:45]: {input_ids[0, 35:45].tolist()}")
                
                text_start = self.system_prompt_length + self.visual_token_length
                if text_start < seq_length:
                    print(f"  Text tokens [{text_start}:{text_start+10}]: {input_ids[0, text_start:text_start+10].tolist()}")
                    print(f"  Last text tokens [{seq_length-10}:{seq_length}]: {input_ids[0, -10:].tolist()}")
            
            print("="*80 + "\n")

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
            elif self.mode == "star_v5":
                mode_name = "STAR-V5 Stage 2"
            else:
                mode_name = "STAR-FastV"
            print(f"[{mode_name}] Prefill: visual tokens = {self.current_visual_length}, target = {self.target_visual_tokens}")

        # Process layers with progressive pruning
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        visual_token_sum = 0

        # Reset layer visual tokens tracking for each forward pass during prefill
        if seq_length > 1 and not self.prefill_done:
            self.layer_visual_tokens = []

        # DEBUG: Print forward pass info
        if self.mode == "star_v5":
            print(f"\n[STAR-V5 DEBUG] Forward pass: seq_length={seq_length}, prefill_done={self.prefill_done}, past_kv={'None' if past_key_values is None else 'exists'}")

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            # Track visual token count for each layer (following PDrop/SparseVLM approach)
            if seq_length > 1 and self.prefill_done:
                visual_token_sum += self.current_visual_length
                # Record token count for this layer for FLOPS calculation
                self.layer_visual_tokens.append(self.current_visual_length)

            # Apply text-guided pruning at scheduled layers
            # DEBUG: Check pruning conditions for STAR-V5
            if self.mode == "star_v5" and layer_idx in [1, 3, 11, 23]:
                print(f"[STAR-V5 DEBUG] Layer {layer_idx}: seq_len={seq_length}, prefill_done={self.prefill_done}, in_schedule={layer_idx in self.pruning_layers}")

            if seq_length > 1 and self.prefill_done and layer_idx in self.pruning_layers:
                # Fix: Ensure visual_token_indices is on the same device as current layer for accelerate compatibility
                if self.visual_token_indices.device != hidden_states.device:
                    self.visual_token_indices = self.visual_token_indices.to(hidden_states.device)

                target_visual_length = self.pruning_layers[layer_idx]

                if self.current_visual_length > target_visual_length:
                    visual_start = self.system_prompt_length
                    visual_end = visual_start + self.current_visual_length

                    if self.mode == "star_v3":
                        mode_name = "STAR-V3 Stage 2"
                    elif self.mode == "star_v2":
                        mode_name = "STAR-V2 Stage 2"
                    elif self.mode == "star_v5":
                        mode_name = "STAR-V5 Stage 2"
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
                        # Fix: Ensure text_rater_positions is on the same device as attn_avg for accelerate compatibility
                        text_rater_positions = text_rater_positions.to(attn_avg.device)
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
                    # Fix: Ensure keep_indices is on the same device as visual_token_indices for accelerate compatibility
                    keep_indices = keep_indices.to(self.visual_token_indices.device)
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
                    print(f"[{mode_name}] Layer {layer_idx}: ✓ Pruning complete. Current tokens: {self.current_visual_length}")

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
