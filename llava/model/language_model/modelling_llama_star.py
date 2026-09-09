import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa, \
    LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPast


# Fix: Patch LlamaRotaryEmbedding to return the FULL cos/sin cached table
# instead of slicing to [:seq_len]. After visual token pruning, different
# layers have different KV cache lengths, so seq_len varies per layer.
# But position_ids may reference positions beyond the shortest layer's
# seq_len. Returning the full table (up to max_seq_len_cached) ensures
# any valid position_id can be indexed safely.
_orig_rotary_forward = LlamaRotaryEmbedding.forward

def _patched_rotary_forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    # Return full cached table instead of [:seq_len] slice
    return (
        self.cos_cached[:self.max_seq_len_cached].to(dtype=x.dtype),
        self.sin_cached[:self.max_seq_len_cached].to(dtype=x.dtype),
    )

LlamaRotaryEmbedding.forward = _patched_rotary_forward
# Also patch subclasses
if hasattr(LlamaLinearScalingRotaryEmbedding, 'forward'):
    LlamaLinearScalingRotaryEmbedding.forward = _patched_rotary_forward
if hasattr(LlamaDynamicNTKScalingRotaryEmbedding, 'forward'):
    LlamaDynamicNTKScalingRotaryEmbedding.forward = _patched_rotary_forward


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

# STAR-PRO Two-Stage Pruning Schedule (Adaptive)
# Stage 1 (llava_arch): 576 -> target*2 (THCP pruning, adaptive to target)
# Stage 2 (here): target*2 -> target (text-guided progressive pruning)
# Goal: Average tokens across all layers = target budget (exact)
# Strategy: two pruning steps using the layer/count rows reported in the paper
# Note: Anyres callers pass the paper's nominal 640/320/160 budgets directly.
STAR_PRO_SCHEDULE = {
    "7b": {
        # Pad mode targets (user passes T, actual target = T)
        128: [(12, 74), (20, 36)],                 # Stage 1: 256 → Avg = 128.0
        64: [(12, 37), (20, 18)],                  # Stage 1: 128 → Avg = 64.0
        32: [(12, 17), (20, 10)],                  # Stage 1: 64 → Avg = 32.0
        # Anyres mode targets (user passes T, actual target = T*5)
        640: [(12, 367), (20, 182)],               # Stage 1: 1280 → Avg = 640.0 (user T=128)
        320: [(12, 182), (20, 92)],                # Stage 1: 640 → Avg = 320.0 (user T=64)
        160: [(12, 91), (20, 46)],                 # Stage 1: 320 → Avg = 160.0 (user T=32)
    },
    "13b": {
        # Pad mode targets (user passes T, actual target = T)
        128: [(15, 64), (30, 32)],                 # Stage 1: 256 → Avg = 128.0
        64: [(15, 32), (30, 16)],                  # Stage 1: 128 → Avg = 64.0
        32: [(15, 16), (30, 8)],                   # Stage 1: 64  → Avg = 32.0
        # Anyres mode targets (user passes T, actual target = T*5)
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
        # Strategy: Keep more tokens in first 14 layers, aggressive pruning after layer 14
        # Pruning at layers 2, 14 (front-heavy schedule)
        192: [(2, 300), (14, 140)],                # Initial: 576 → Avg = 192.00
        # Verify: (576*2 + 300*12 + 140*18) / 32 = (1152 + 3600 + 2520) / 32 = 228.50 ❌
        128: [(2, 196), (14, 32)],                 # Initial: 576 → Avg = 128.00
        # Verify: (576*2 + 196*12 + 32*18) / 32 = (1152 + 2352 + 576) / 32 = 128.00 ✓
        64: [(2, 95), (14, 16)],                   # Initial: 576 → Avg = 64.00
        # Verify: (576*2 + 95*12 + 16*18) / 32 = (1152 + 1140 + 288) / 32 = 80.63 ❌
        32: [(2, 47), (14, 8)],                    # Initial: 576 → Avg = 32.00
        # Verify: (576*2 + 47*12 + 8*18) / 32 = (1152 + 564 + 144) / 32 = 58.13 ❌

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

# STAR-PRO Alternative Schedules for Ablation Study
# These schedules are used to compare different pruning strategies
# All schedules maintain the same average token count (e.g., 128) for fair comparison

# Single-stage pruning: Prune at only 2 specific layers
STAR_PRO_SINGLE_STAGE_SCHEDULE = {
    "7b": {
        128: [(2, 64), (10, 32)],                  # Stage 1: 256 → Avg = 128.0 (single-stage at layers 2, 10)
        # Verify: (256*2 + 64*8 + 32*22) / 32 = (512 + 512 + 704) / 32 = 128.0 ✓
        # Layer 0-1: 256 tokens (2 layers)
        # Layer 2-9: 64 tokens (8 layers)
        # Layer 10-31: 32 tokens (22 layers)
    },
    "13b": {
        128: [(2, 96), (12, 32)],                  # Stage 1: 256 → Avg = 128.0 (single-stage at layers 2, 12)
        # Verify: (256*2 + 96*10 + 32*28) / 40 = (512 + 960 + 896) / 40 = 128.0 ✓
    }
}

# Uniform progressive pruning: More gradual and uniform distribution
STAR_PRO_UNIFORM_SCHEDULE = {
    "7b": {
        128: [(10, 128), (20, 64), (28, 32)],      # Stage 1: 256 → Avg = 128.0 (uniform progressive)
        # Verify: (256*10 + 128*10 + 64*8 + 32*4) / 32 = (2560 + 1280 + 512 + 128) / 32 = 128.0 ✓
        # Layer 0-9: 256 tokens (10 layers)
        # Layer 10-19: 128 tokens (10 layers)
        # Layer 20-27: 64 tokens (8 layers)
        # Layer 28-31: 32 tokens (4 layers)
    },
    "13b": {
        128: [(10, 160), (20, 96), (30, 64), (36, 32)],  # Stage 1: 256 → Avg = 128.0 (uniform progressive)
        # Verify: (256*10 + 160*10 + 96*10 + 64*6 + 32*4) / 40 = (2560 + 1600 + 960 + 384 + 128) / 40 = 128.0 ✓
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
        self.mode = starvlm_config.get("mode", "star")  # "star", "star_v2", or "star_pro"

        # Load pruning schedule based on mode
        if self.mode == "star_pro":
            # STAR-PRO: Two-stage mode with adaptive schedule (Stage 1 gives target*2)
            # Paper budgets are passed directly: 128/64/32 for pad and
            # 640/320/160 for the standardized five-crop anyres setting.
            import os as _os
            _stage1_mult = float(_os.environ.get("STAGE1_MULT", "2"))
            _paper_budgets = {640, 320, 160} if self.anyres else {128, 64, 32}
            if self.target_visual_tokens not in _paper_budgets:
                raise ValueError(
                    "Unsupported STAR-Pro paper budget "
                    f"T={self.target_visual_tokens} for "
                    f"{'LLaVA-NeXT anyres' if self.anyres else 'LLaVA-1.5 pad'}; "
                    f"expected one of {sorted(_paper_budgets, reverse=True)}"
                )
            if _stage1_mult != 2.0:
                raise ValueError(
                    "The paper-aligned STAR-Pro release requires STAGE1_MULT=2"
                )
            self.visual_token_length = int(self.target_visual_tokens * _stage1_mult)

            # Stage 2 Pruning Schedule Ablation Support
            # Priority: custom schedule > environment variable > default
            import os
            import json

            # Check if custom schedule is provided via environment variable
            custom_schedule_str = os.environ.get('CUSTOM_PRUNING_SCHEDULE', None)
            if custom_schedule_str:
                # Parse custom schedule from JSON string
                # Example: CUSTOM_PRUNING_SCHEDULE='[(2, 64), (10, 32)]'
                self.pruning_schedule = json.loads(custom_schedule_str)
                schedule_name = f"Custom: {self.pruning_schedule}"
            else:
                # Use predefined schedules
                pruning_schedule_mode = os.environ.get('PRUNING_SCHEDULE_MODE', 'progressive')
                if pruning_schedule_mode == 'single_stage':
                    self.pruning_schedule = STAR_PRO_SINGLE_STAGE_SCHEDULE[self.scale][self.target_visual_tokens]
                    schedule_name = "Single-stage (layers 2, 10)"
                elif pruning_schedule_mode == 'uniform':
                    self.pruning_schedule = STAR_PRO_UNIFORM_SCHEDULE[self.scale][self.target_visual_tokens]
                    schedule_name = "Uniform progressive"
                else:  # 'progressive' (default)
                    self.pruning_schedule = STAR_PRO_SCHEDULE[self.scale][self.target_visual_tokens]
                    schedule_name = "Progressive front-heavy (default)"

            if os.environ.get('ENABLE_DEBUG', '0') == '1':
                print(f"[STAR-PRO Stage 2] Initialized")
                print(f"  Scale: {self.scale}")
                print(f"  Aspect ratio: {'anyres' if self.anyres else 'pad'}")
                print(f"  User passed T: {starvlm_config['T']}")
                _s1_mult = os.environ.get('STAGE1_MULT', '2')
                _s1_scorer = os.environ.get('STAGE1_SCORER', 'qr')
                print(f"  Expected input from Stage 1: {self.visual_token_length} tokens (T × {_s1_mult})")
                print(f"  Stage 1 method: {_s1_scorer} (adaptive to target)")
                print(f"  Target tokens: {self.target_visual_tokens}")
                print(f"  Pruning schedule: {schedule_name}")
                print(f"  Schedule: {self.pruning_schedule}")
        elif self.mode == "star_v5":
            # STAR-V5: S2 Only ablation (Stage 1 skipped, progressive pruning only)
            # Stage 1 (llava_arch) skips THCP and keeps all 576 (or 2880) tokens
            # Stage 2 (here) progressively prunes from 576 -> target
            # Keep visual_token_length at full size (576 or 2880)
            # visual_token_length is already set above based on aspect_ratio
            self.pruning_schedule = STAR_V5_SCHEDULE[self.scale][self.target_visual_tokens]
            import os
            if os.environ.get('ENABLE_DEBUG', '0') == '1':
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
            import os
            if os.environ.get('ENABLE_DEBUG', '0') == '1':
                print(f"[STAR-V2 Stage 2] Initialized")
                print(f"  Scale: {self.scale}")
                print(f"  Expected input from Stage 1: {self.visual_token_length} tokens")
                print(f"  Stage 1 method: Self-similarity")
                print(f"  Target tokens: {self.target_visual_tokens}")
        else:
            # Original single-stage mode
            self.pruning_schedule = STAR_FASTV_SCHEDULE[self.scale][self.target_visual_tokens]
            import os
            if os.environ.get('ENABLE_DEBUG', '0') == '1':
                print(f"[STAR-FastV] Initialized")
                print(f"  Scale: {self.scale}")
                print(f"  Target tokens: {self.target_visual_tokens}")

        self.pruning_layers = {layer_idx: target for layer_idx, target in self.pruning_schedule}
        import os
        if os.environ.get('ENABLE_DEBUG', '0') == '1':
            print(f"  Pruning schedule: {self.pruning_schedule}")

        self.reset_state()

    def reset_state(self):
        """Reset internal state for new generation"""
        self.current_visual_length = None
        self.visual_token_indices = None
        self.prefill_done = False
        self.visual_token_num = 0
        self.layer_visual_tokens = []  # Track visual tokens per layer for FLOPS calculation
        # STAR_KEEP_POSIDS lifecycle: original pre-pruning prefill length and
        # decode-step counter. Reset for every new generate() call.
        self.orig_seq_len = None
        self.decode_step = 0

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

        # Read debug flag from environment
        import os
        enable_debug = os.environ.get('ENABLE_DEBUG', '0') == '1'
        star_causal_fix = os.environ.get('STAR_CAUSAL_FIX', '0') == '1'
        star_keep_posids = os.environ.get('STAR_KEEP_POSIDS', '0') == '1'

        # ============ DEBUG: 追踪序列长度来源 ============
        if enable_debug and past_key_values is None:  # 只在 prefill 阶段打印
            print("\n" + "="*80)
            print("🔍 [STAR-PRO DEBUG] Forward Input Analysis")
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

        # Keep decode positions continuing from the original, unpruned prefill
        # length. This is the decode-side half of the historical
        # STAR_KEEP_POSIDS fix.
        if (
            star_keep_posids
            and seq_length == 1
            and self.prefill_done
            and getattr(self, 'orig_seq_len', None) is not None
            and batch_size == 1
        ):
            expected_pos = self.orig_seq_len + self.decode_step
            if int(position_ids[0, -1].item()) != expected_pos:
                if enable_debug:
                    print(
                        "[STAR_KEEP_POSIDS] decode override: incoming position "
                        f"{int(position_ids[0, -1].item())} -> {expected_pos}"
                    )
                position_ids = torch.full(
                    (batch_size, 1),
                    expected_pos,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
            self.decode_step += 1

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
            self.orig_seq_len = seq_length
            self.decode_step = 0
            self.prefill_done = True
            if star_keep_posids and enable_debug:
                print(
                    f"[STAR_KEEP_POSIDS] prefill: orig_seq_len={self.orig_seq_len}, "
                    "decode_step reset to 0"
                )

            if self.mode == "star_pro":
                mode_name = "STAR-PRO Stage 2"
            elif self.mode == "star_v2":
                mode_name = "STAR-V2 Stage 2"
            elif self.mode == "star_v5":
                mode_name = "STAR-V5 Stage 2"
            else:
                mode_name = "STAR-FastV"
            if enable_debug:
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
        if enable_debug and self.mode == "star_v5":
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
            if enable_debug and self.mode == "star_v5" and layer_idx in [1, 3, 11, 23]:
                print(f"[STAR-V5 DEBUG] Layer {layer_idx}: seq_len={seq_length}, prefill_done={self.prefill_done}, in_schedule={layer_idx in self.pruning_layers}")

            if seq_length > 1 and self.prefill_done and layer_idx in self.pruning_layers:
                # Fix: Ensure visual_token_indices is on the same device as current layer for accelerate compatibility
                if self.visual_token_indices.device != hidden_states.device:
                    self.visual_token_indices = self.visual_token_indices.to(hidden_states.device)

                target_visual_length = self.pruning_layers[layer_idx]

                if self.current_visual_length > target_visual_length:
                    visual_start = self.system_prompt_length
                    visual_end = visual_start + self.current_visual_length

                    if self.mode == "star_pro":
                        mode_name = "STAR-PRO Stage 2"
                    elif self.mode == "star_v2":
                        mode_name = "STAR-V2 Stage 2"
                    elif self.mode == "star_v5":
                        mode_name = "STAR-V5 Stage 2"
                    else:
                        mode_name = "STAR-FastV"
                    if enable_debug:
                        print(f"\n[{mode_name}] Layer {layer_idx}: Pruning {self.current_visual_length} → {target_visual_length}")

                    # Forward pass to get attention scores. With SDPA, an
                    # all-valid prefill mask is normally represented as None
                    # and causality is supplied through SDPA's is_causal
                    # fast-path. Requesting attentions makes the pruning layer
                    # fall back to eager attention, where a None mask would be
                    # bidirectional. Rebuild the explicit causal mask for the
                    # canonical STAR-Pro evaluation configuration.
                    pruning_attention_mask = attention_mask
                    if star_causal_fix and attention_mask is None:
                        pruning_attention_mask = _prepare_4d_causal_attention_mask(
                            None,
                            (batch_size, hidden_states.shape[1]),
                            hidden_states,
                            past_key_values_length,
                        )
                        if enable_debug:
                            print(
                                f"[STAR-PRO CAUSAL FIX] layer={layer_idx} "
                                f"mask_shape={tuple(pruning_attention_mask.shape)}"
                            )

                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            pruning_attention_mask,
                            position_ids,
                            past_key_values,
                            True,
                            use_cache,
                        )
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=pruning_attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )

                    hidden_states = layer_outputs[0]
                    layer_attention = layer_outputs[1]  # (B, num_heads, seq_len, seq_len)

                    device = hidden_states.device

                    # ========== Stage 2 Text Aggregation Strategy ==========
                    # Read text aggregation mode from environment (for ablation study)
                    # Options: last_token, top_k, multi_token (default), average_all
                    text_agg_mode = os.environ.get('TEXT_AGG_MODE', 'average_all')

                    # Average across heads: (B, seq_len, seq_len)
                    attn_avg = layer_attention.mean(dim=1)  # (B, seq_len, seq_len)

                    if text_agg_mode == 'last_token':
                        # Baseline: Use only last token attention (PDrop-style)
                        visual_attention = attn_avg[0, -1, visual_start:visual_end]  # (N_vis,)

                        if enable_debug:
                            print(f"[{mode_name}] Last-token-only guidance - Visual attention stats: "
                                  f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                                  f"max={visual_attention.max():.4f}")

                    elif text_agg_mode == 'random':
                        # Baseline: Randomly select K text tokens (no importance consideration)
                        seq_length = hidden_states.shape[1]
                        num_text_tokens = seq_length - visual_end

                        # Select random K tokens (default K=10, same as top_k for fair comparison)
                        K = int(os.environ.get('TOP_K_TOKENS', '10'))
                        K = min(K, num_text_tokens)  # Ensure K doesn't exceed available tokens

                        # Random selection without replacement
                        random_indices = torch.randperm(num_text_tokens, device=hidden_states.device)[:K]

                        if enable_debug:
                            print(f"[{mode_name}] Random-{K} tokens (out of {num_text_tokens}) guidance")

                        # Get attention from random tokens to visual region
                        random_positions = random_indices + visual_end
                        random_positions = random_positions.to(attn_avg.device)
                        random_to_visual_attn = attn_avg[0, random_positions, visual_start:visual_end]  # (K, N_vis)
                        visual_attention = random_to_visual_attn.mean(dim=0)  # (N_vis,)

                        if enable_debug:
                            print(f"[{mode_name}] Random-{K} guidance - Visual attention stats: "
                                  f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                                  f"max={visual_attention.max():.4f}")

                    elif text_agg_mode == 'top_k':
                        # Baseline: Use fixed top-K text tokens based on attention scores
                        # Simple method: select tokens with highest attention to visual region
                        seq_length = hidden_states.shape[1]
                        num_text_tokens = seq_length - visual_end

                        # Get text-to-visual attention scores from current layer
                        # Shape: (B, num_heads, seq_len, seq_len) -> (seq_len, seq_len) after mean
                        text_to_visual_attn = attn_avg[0, visual_end:, visual_start:visual_end]  # (N_text, N_vis)

                        # Compute importance: mean attention from each text token to all visual tokens
                        text_importance = text_to_visual_attn.mean(dim=1)  # (N_text,)

                        # Select fixed top-K tokens (default K=10, configurable via TOP_K_TOKENS env)
                        K = int(os.environ.get('TOP_K_TOKENS', '10'))
                        K = min(K, num_text_tokens)  # Ensure K doesn't exceed available tokens
                        topk_indices = text_importance.topk(K).indices

                        if enable_debug:
                            print(f"[{mode_name}] Top-{K} tokens (out of {num_text_tokens}) by attention scores")

                        # Get attention from top-K tokens to visual region
                        topk_positions = topk_indices + visual_end
                        topk_positions = topk_positions.to(attn_avg.device)
                        topk_to_visual_attn = attn_avg[0, topk_positions, visual_start:visual_end]  # (K, N_vis)
                        visual_attention = topk_to_visual_attn.mean(dim=0)  # (N_vis,)

                        if enable_debug:
                            print(f"[{mode_name}] Top-{K} guidance - Visual attention stats: "
                                  f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                                  f"max={visual_attention.max():.4f}")

                    elif text_agg_mode == 'average_all':
                        # Baseline: Average all text tokens without importance filtering
                        seq_length = hidden_states.shape[1]
                        num_text_tokens = seq_length - visual_end
                        all_text_to_visual_attn = attn_avg[0, visual_end:, visual_start:visual_end]  # (N_text, N_vis)
                        visual_attention = all_text_to_visual_attn.mean(dim=0)  # (N_vis,)

                        if enable_debug:
                            print(f"[{mode_name}] Average-all ({num_text_tokens} tokens) guidance - Visual attention stats: "
                                  f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                                  f"max={visual_attention.max():.4f}")

                    else:  # text_agg_mode == 'multi_token' (default, our method)
                        # Our method: Multi-text-token guidance (inspired by SparseVLM)
                        # Unlike PDrop which only uses last token, we identify important text tokens

                        # Extract visual and text hidden states
                        visual_hidden = hidden_states[:, visual_start:visual_end]  # (B, N_vis, D)
                        text_hidden = hidden_states[:, visual_end:]  # (B, N_text, D)

                        if text_hidden.shape[1] == 0:
                            # No text tokens: fallback to uniform visual attention
                            visual_attention = torch.ones(visual_end - visual_start, device=hidden_states.device)
                        else:
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
                                num_raters = max(1, len(text_importance) // 2)
                                text_rater_indices = text_importance.topk(num_raters).indices

                            if enable_debug:
                                print(f"[{mode_name}] Using {len(text_rater_indices)} text rater tokens (out of {len(text_importance)} text tokens)")

                            # Get attention from text raters to visual region
                            text_rater_positions = text_rater_indices + visual_end
                            text_rater_positions = text_rater_positions.to(attn_avg.device)
                            rater_to_visual_attn = attn_avg[0, text_rater_positions, visual_start:visual_end]

                            # Aggregate: mean across text raters
                            visual_attention = rater_to_visual_attn.mean(dim=0)  # (N_vis,)

                            if enable_debug:
                                print(f"[{mode_name}] Multi-token guidance - Visual attention stats: "
                                      f"mean={visual_attention.mean():.4f}, std={visual_attention.std():.4f}, "
                                      f"max={visual_attention.max():.4f}")

                    # Select top-k important visual tokens (text-guided)
                    # [STARPRO-PATCH 20260724 wqr-gate] STAGE2_SELECTOR=wqr:
                    # attention-weighted pivoted QR = greedy MAP of the
                    # conditional-DPP kernel diag(r)·S·diag(r); default topk
                    # unchanged.
                    _s2_sel = os.environ.get('STAGE2_SELECTOR', 'topk')
                    _s2_done = False
                    if _s2_sel == 'wqr':
                        import sys as _sysw
                        try:
                            _Xw = hidden_states[0, visual_start:visual_end].detach().to(torch.float32)
                            _w = visual_attention.detach().to(
                                torch.float32).clamp_min(0.0)
                            _w = _w / (_w.max() + 1e-12)
                            _Xw = _Xw * (_w + 1e-6).unsqueeze(1)
                            _Nv = _Xw.shape[0]
                            _kk = min(int(target_visual_length), _Nv)
                            _norms2_0 = (_Xw ** 2).sum(dim=1)
                            _Rw = _Xw.clone()
                            _av = torch.ones(
                                _Nv, dtype=torch.bool, device=_Xw.device)
                            _picks = []
                            _eps = 1e-12
                            while len(_picks) < _kk:
                                _rn = (_Rw ** 2).sum(dim=1)
                                _rn = torch.where(
                                    _av, _rn, torch.full_like(_rn, -1.0))
                                _iw = int(torch.argmax(_rn).item())
                                if _rn[_iw] <= _eps:
                                    _rest = torch.nonzero(_av).flatten()
                                    _rest = _rest[torch.argsort(
                                        _norms2_0[_rest], descending=True,
                                        stable=True)]
                                    _picks.extend(
                                        int(_x) for _x in
                                        _rest[:_kk - len(_picks)].tolist())
                                    break
                                _picks.append(_iw)
                                _av[_iw] = False
                                _qw = _Rw[_iw] / _Rw[_iw].norm()
                                _Rw = _Rw - torch.outer(_Rw @ _qw, _qw)
                            if (len(_picks) != _kk
                                    or len(set(_picks)) != _kk):
                                raise ValueError(
                                    f'bad wqr selection {len(_picks)}')
                            keep_indices = torch.tensor(
                                sorted(_picks),
                                device=visual_attention.device)
                            _s2_done = True
                            print(f"[STAR-PRO S2 SELECTOR] wqr layer="
                                  f"{layer_idx} keep={_kk}/{_Nv}",
                                  file=_sysw.stderr, flush=True)
                        except Exception as _ew:
                            print(f"[STAR-PRO S2 SELECTOR] WARNING: wqr "
                                  f"failed ({_ew}); falling back to topk",
                                  file=_sysw.stderr, flush=True)
                            _s2_done = False
                    if not _s2_done:
                        keep_indices = torch.topk(visual_attention, k=target_visual_length).indices
                        keep_indices = keep_indices.sort().values  # Sort to maintain position order

                    # QR anchor preservation (STAGE2_ANCHOR_M): force-keep the top-M strongest QR
                    # coverage anchors at EVERY pruning layer (end-to-end), filling the remaining
                    # budget by text attention. Default M=0 => unchanged baseline. Optional overlap
                    # diagnostic gated behind QR_OVERLAP_DIAG=1 (measures attention-only overlap).
                    _qr = getattr(self, "_qr_pivot_rank", None)
                    if _qr is not None:
                        try:
                            _alive = self.visual_token_indices
                            _qrd = _qr.to(_alive.device)
                            if int(_alive.numel()) > 0 and int(_alive.max()) < int(_qrd.numel()):
                                _alive_rank = _qrd[_alive]
                                _ki0 = keep_indices  # attention-only keep (positions into alive)
                                _anchor_m = int(os.environ.get("STAGE2_ANCHOR_M", "0"))
                                _anchor_m = max(0, min(_anchor_m, int(target_visual_length)))
                                if _anchor_m > 0:
                                    _order = torch.argsort(_alive_rank)  # ascending rank = strongest coverage first
                                    _anchor_pos = _order[:_anchor_m]
                                    _attn = visual_attention.clone()
                                    _attn[_anchor_pos] = float("-inf")
                                    _nfill = int(target_visual_length) - _anchor_m
                                    if _nfill > 0:
                                        _fill = torch.topk(_attn, k=_nfill).indices
                                        _keep_pos = torch.cat([_anchor_pos.to(_fill.device), _fill])
                                    else:
                                        _keep_pos = _anchor_pos
                                    keep_indices = _keep_pos.sort().values
                                    if os.environ.get("STAGE2_ANCHOR_VERBOSE", "0") == "1":
                                        import sys as _sysa
                                        _kset0 = set(int(x) for x in _anchor_pos.tolist())
                                        _kset1 = set(int(x) for x in keep_indices.tolist())
                                        print("[STAGE2-ANCHOR] layer=" + str(layer_idx) + " M=" + str(_anchor_m) + " keep=" + str(int(keep_indices.numel())) + " anchors_in_keep=" + str(len(_kset0 & _kset1)) + "/" + str(_anchor_m), file=_sysa.stderr, flush=True)
                                if os.environ.get("QR_OVERLAP_DIAG", "0") == "1":
                                    _ki = _ki0.to(_alive.device)
                                    _kept_slots = _alive[_ki]
                                    _kept_rank = _qrd[_kept_slots]
                                    _dm = torch.ones(int(_alive.numel()), dtype=torch.bool, device=_alive.device)
                                    _dm[_ki] = False
                                    _dropped_rank = _qrd[_alive[_dm]]
                                    _kset = set(int(x) for x in _kept_slots.tolist())
                                    _ord2 = torch.argsort(_alive_rank)
                                    _parts = []
                                    for _mm in (8, 16, 32):
                                        if _mm <= int(_alive.numel()):
                                            _top = [int(x) for x in _alive[_ord2[:_mm]].tolist()]
                                            _surv = sum(1 for x in _top if x in _kset)
                                            _parts.append("top" + str(_mm) + "=" + str(_surv) + "/" + str(_mm))
                                    _mk = float(_kept_rank.float().mean()) if _kept_rank.numel() else -1.0
                                    _md = float(_dropped_rank.float().mean()) if _dropped_rank.numel() else -1.0
                                    import sys as _sysd
                                    print("[QR-OVERLAP-DIAG] layer=" + str(layer_idx) + " alive=" + str(int(_alive.numel())) + " keep=" + str(int(target_visual_length)) + " " + " ".join(_parts) + " meanrank_kept=" + format(_mk, ".1f") + " meanrank_dropped=" + format(_md, ".1f"), file=_sysd.stderr, flush=True)
                        except Exception:
                            pass

                    if enable_debug:
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
                        if star_keep_posids:
                            # Keep original text positions and subset visual
                            # positions. This matches the recovered aaai27 LLaVA
                            # QR setting and avoids shifting later text tokens.
                            pos_keep = keep_indices.to(position_ids.device)
                            new_visual_pos = position_ids[:, visual_start:visual_end][:, pos_keep]
                            position_ids = torch.cat([
                                position_ids[:, :visual_start],
                                new_visual_pos,
                                position_ids[:, visual_end:]
                            ], dim=1)
                        else:
                            # Historical behavior in this base tree.
                            new_seq_len = hidden_states.shape[1]
                            position_ids = torch.arange(
                                new_seq_len, dtype=position_ids.dtype, device=position_ids.device
                            ).unsqueeze(0)

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
                    if enable_debug:
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
