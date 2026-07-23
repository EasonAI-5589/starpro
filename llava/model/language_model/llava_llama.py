#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .modeling_llama_fastv import FastVLlamaModel
from .modeling_llama_sparsevlm import SparseLlamaModel
from .modeling_llama_divdriven import D2PLlamaModel
from .modeling_llama_svd import SVDLlamaModel
from .modeling_llama_prefix import PrefixLlamaModel
from .modeling_llama_idea import IdeaLlamaModel
from .modeling_llama_prefix_2 import Prefix_2_LlamaModel
from .modeling_llama_pdrop import PDropLlamaModel
from .modelling_llama_star import STARVLMModel  # ⭐ 从 modelling_llama_star 导入
from .modelling_llama_mustdrop import MustDropLlamaModel  # MustDrop baseline
from .modeling_llama_vscan import VScanLlamaModel  # 🔬 VScan baseline
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class FastVLlavaLlamaModel(LlavaMetaModel, FastVLlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, fastv_config: dict):
        super(FastVLlavaLlamaModel, self).__init__(config, fastv_config=fastv_config)


class SparseLlavaLlamaModel(LlavaMetaModel, SparseLlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, sparsevlm_config: dict):
        super(SparseLlavaLlamaModel, self).__init__(config, sparsevlm_config=sparsevlm_config)
        
#----------------------------------------------------------------------------------------------
#D2P: Diversity Driven Pruning Algorithm
class D2PLlavaLlamaModel(LlavaMetaModel, D2PLlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, d2p_config: dict):
        super(D2PLlavaLlamaModel, self).__init__(config, d2p_config=d2p_config)
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#SVDVLM: SVD-based Visual Language Model
class SVDLlavaLlamaModel(LlavaMetaModel, SVDLlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, svdvlm_config: dict):
        super(SVDLlavaLlamaModel, self).__init__(config, svdvlm_config=svdvlm_config)
#----------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------
# PrefixVLM: Prefix-based Visual Language Model
class PrefixLlavaLlamaModel(LlavaMetaModel, PrefixLlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, prefixvlm_config: dict):
        super(PrefixLlavaLlamaModel, self).__init__(config, prefixvlm_config=prefixvlm_config)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Idea: Idea baseline
class Idea_LlavaLlamaModel(LlavaMetaModel, IdeaLlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, idea_config: dict):
        super(Idea_LlavaLlamaModel, self).__init__(config, idea_config=idea_config)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Prefix_2: Prefix-2 baseline
class Prefix_2_LlavaLlamaModel(LlavaMetaModel, Prefix_2_LlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, prefixvlm_2_config: dict):
        super(Prefix_2_LlavaLlamaModel, self).__init__(config, prefixvlm_2_config=prefixvlm_2_config)
# -------------------------------------------------------------------------
class PDropLlavaLlamaModel(LlavaMetaModel, PDropLlamaModel):
    config_class = LlavaLlamaConfig
    
    def __init__(self, config: LlamaConfig, pdrop_config: dict):
        super(PDropLlavaLlamaModel, self).__init__(config, pdrop_config=pdrop_config)


# ⭐ STAR: Semantic-aware Token Allocation with Regional Visual Latent Model
class STARLlavaLlamaModel(LlavaMetaModel, STARVLMModel):
    """
    STAR integrated with LLaVA
    """
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, star_config: dict):
        super(STARLlavaLlamaModel, self).__init__(config, starvlm_config=star_config)


# MustDrop: Dual Attention Filter baseline for comparison
class MustDropLlavaLlamaModel(LlavaMetaModel, MustDropLlamaModel):
    """
    MustDrop baseline integrated with LLaVA.

    Reference: MustDrop paper - "MustDrop: Training-free Token Dropping for Efficient VLMs"

    Key features:
    1. Vision Encoder: Layer 0 token merging + Layer 23 key set extraction
    2. LLM: Dual Attention Filter at layers [2, 6, 10, 14]
    3. KV Cache: Progressive sparsification
    """
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, mustdrop_config: dict = None):
        super(MustDropLlavaLlamaModel, self).__init__(config, mustdrop_config=mustdrop_config)


# 🔬 VScan: Training-Free Visual Token Reduction for comparison
class VScanLlavaLlamaModel(LlavaMetaModel, VScanLlamaModel):
    """
    VScan baseline integrated with LLaVA.

    Reference: https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan

    Key features:
    1. Stage 1 (Vision Encoder): Complementary global and local scans
    2. Stage 2 (LLM): Middle layer attention-based pruning
    """
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, vscan_config: dict = None):
        super(VScanLlavaLlamaModel, self).__init__(config, vscan_config=vscan_config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config, pruning_method=None, visual_token_num=None,
                 use_fastv=False, fastv_config=None,
                 use_sparsevlm=False, sparsevlm_config=None,
                 use_svdvlm=False, svdvlm_config=None,
                 use_prefixvlm=False, prefixvlm_config=None,
                 use_d2p=False, d2p_config=None,
                 use_idea=False, idea_config=None,
                 use_prefixvlm_2=False, prefixvlm_2_config=None,
                 use_pdrop=False, pdrop_config=None,
                 use_star=False, star_config=None,  # ⭐ 参数名: use_star, star_config
                 use_mustdrop=False, mustdrop_config=None,  # MustDrop baseline
                 **kwargs):
        # 🔥 MustDrop: Set config flag BEFORE model creation
        # This ensures build_vision_tower() uses CLIPVisionTowerMustDrop
        if use_mustdrop:
            config.use_mustdrop = True
            # Store mustdrop_config in config for later access
            config.mustdrop_config = mustdrop_config if mustdrop_config else {}

        super(LlamaForCausalLM, self).__init__(config)

        # ⭐ Model selection with STAR and MustDrop support
        if use_mustdrop:
            print(f"Use MustDrop: {mustdrop_config}")
            self.model = MustDropLlavaLlamaModel(config, mustdrop_config=mustdrop_config)
        elif use_star:
            self.model = STARLlavaLlamaModel(config, star_config=star_config)
        elif use_fastv:
            print(f"Use FastV: {fastv_config}")
            self.model = FastVLlavaLlamaModel(config, fastv_config)
        elif use_sparsevlm:
            print(f"Use SparseVLM: {sparsevlm_config}")
            self.model = SparseLlavaLlamaModel(config, sparsevlm_config=sparsevlm_config)
        elif use_d2p:
            print(f"Use D2P: {d2p_config}")
            self.model = D2PLlavaLlamaModel(config, d2p_config=d2p_config)
        elif use_svdvlm:
            print(f"Use SVDVLM: {svdvlm_config}")
            self.model = SVDLlavaLlamaModel(config, svdvlm_config=svdvlm_config)
        elif use_prefixvlm:
            print(f"Use PrefixVLM: {prefixvlm_config}")
            self.model = PrefixLlavaLlamaModel(config, prefixvlm_config=prefixvlm_config)
        elif use_idea:
            print(f"Use Idea: {idea_config}")
            self.model = Idea_LlavaLlamaModel(config, idea_config=idea_config)
        elif use_prefixvlm_2:
            print(f"Use PrefixVLM_2: {prefixvlm_2_config}")
            self.model = Prefix_2_LlavaLlamaModel(config, prefixvlm_2_config=prefixvlm_2_config)
        elif use_pdrop:
            print(f"Use PDrop: {pdrop_config}")
            self.model = PDropLlavaLlamaModel(config, pdrop_config=pdrop_config)
        else:
            self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Visual Token Pruning config
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num

        # 🔥 MustDrop: Store flag and config for access in forward/generate
        self.use_mustdrop = use_mustdrop
        self.mustdrop_config = mustdrop_config

        # Latency
        self.start_latency = False
        self.phase = "prefill"
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.prefill_latency = 0.0
        self.decode_latency = 0.0

        # FLOPS tracking
        self.track_flops = False
        self.layer_visual_tokens = []  # Record visual tokens per layer
        self.total_flops = 0.0  # Total FLOPs in TFLOPs
        self.flops_count = 0  # Number of samples for averaging

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_pruning_method(self):
        return self.pruning_method

    def get_visual_token_num(self):
        return self.visual_token_num

    def get_model(self):
        return self.model

    def calculate_flops(self):
        """
        Calculate FLOPs based on recorded visual token numbers per layer.

        Formula per layer (from cal_flops.py):
        FLOPs = 8 * n * d^2 + 4 * n^2 * d + 6 * n * d * m
        where:
        - n: number of visual tokens in this layer
        - d: hidden dimension (config.hidden_size)
        - m: FFN intermediate dimension (config.intermediate_size)

        Returns:
            float: Total FLOPs in TFLOPs (10^12 FLOPs)
        """
        if not self.layer_visual_tokens:
            return 0.0

        d = self.config.hidden_size
        m = self.config.intermediate_size

        flops = 0.0
        for n in self.layer_visual_tokens:
            flops += (8 * n * d * d + 4 * n * n * d + 6 * n * d * m)

        # Convert to TFLOPs
        flops_tflops = flops / 1e12
        return flops_tflops

    def record_layer_tokens(self, layer_tokens):
        """
        Record visual token numbers for each layer.

        Args:
            layer_tokens: List of token numbers per layer
        """
        self.layer_visual_tokens = layer_tokens
        if self.track_flops:
            flops = self.calculate_flops()
            self.total_flops += flops
            self.flops_count += 1

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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # MustDrop: Set parameters on model before forward
        # Reference: MustDrop sparse_llava_llama.py parameter passing
        if hasattr(self.model, 'img_seq') and images is not None and inputs_embeds is not None:
            # 🔥 MustDrop Full Integration: Get actual values from Vision Encoder
            # Reference: MustDrop clip_encoder.py + modelling_sparse_llama.py
            if self.use_mustdrop and hasattr(self, '_mustdrop_img_seq'):
                # Use actual values from MustDrop Vision Encoder
                # img_seq: Number of visual tokens AFTER merging (< 576)
                self.model.img_seq = self._mustdrop_img_seq

                # key_set: Format expected by MustDropLlamaModel is [num_keys, key_indices_tensor]
                # Reference: modelling_llama_mustdrop.py:256-257
                if self._mustdrop_key_set is not None:
                    key_set = self._mustdrop_key_set
                    # key_set shape: [B, K] where K is number of key tokens
                    num_keys = key_set.shape[1] if key_set.dim() > 1 else key_set.shape[0]
                    key_indices = key_set[0] if key_set.dim() > 1 else key_set  # Take first batch
                    self.model.key_set = [num_keys, key_indices]
                else:
                    self.model.key_set = None

                print(f"[MustDrop] img_seq={self.model.img_seq} (after merging), "
                      f"key_set_size={self.model.key_set[0] if self.model.key_set else 0}")
            else:
                # Fallback: Standard LLaVA without Vision Encoder modifications
                # Default LLaVA: 576 tokens for 336x336 image with 14x14 patch
                self.model.img_seq = 576
                self.model.key_set = None

            # System prompt length (before image tokens)
            # Reference: MustDrop modelling_sparse_llama.py:237
            self.model.pre_prompt_length_list = [35]  # Default LLaVA system prompt length

            # Total token length
            self.model.token_length_list = [inputs_embeds.shape[1]]

        if self.start_latency:
            if self.phase == "prefill" and input_ids is None:
                self.start_event.record()
            elif self.phase == "decode" and input_ids is not None and input_ids.shape[1] == 1:
                self.start_event.record()

        # Pass token library to inner model for dynamic cluster expand/collapse (idea method)
        if hasattr(self, '_idea_token_library'):
            self.model._idea_token_library = self._idea_token_library

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if self.start_latency:
            if self.phase == "prefill" and input_ids is None:
                self.end_event.record()
                torch.cuda.synchronize()
                self.prefill_latency += self.start_event.elapsed_time(self.end_event)
                self.phase = "decode"
            elif self.phase == "decode" and input_ids is not None and input_ids.shape[1] == 1:
                self.end_event.record()
                torch.cuda.synchronize()
                self.decode_latency += self.start_event.elapsed_time(self.end_event)
                self.phase = "prefill"

        # Clean up dynamic cluster state after forward (idea method)
        if hasattr(self.model, '_idea_expanded_clusters'):
            del self.model._idea_expanded_clusters
        if hasattr(self.model, '_idea_current_token_cluster_ids'):
            del self.model._idea_current_token_cluster_ids

        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        texts: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                visual_token_num
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                texts=texts
            )

            # 🔥 MustDrop: Set parameters on model before generate
            # This is needed because generate() internally calls forward()
            if self.use_mustdrop and hasattr(self.model, 'img_seq'):
                if hasattr(self, '_mustdrop_img_seq'):
                    self.model.img_seq = self._mustdrop_img_seq

                    if self._mustdrop_key_set is not None:
                        key_set = self._mustdrop_key_set
                        num_keys = key_set.shape[1] if key_set.dim() > 1 else key_set.shape[0]
                        key_indices = key_set[0] if key_set.dim() > 1 else key_set
                        self.model.key_set = [num_keys, key_indices]
                    else:
                        self.model.key_set = None
                else:
                    self.model.img_seq = 576
                    self.model.key_set = None

                self.model.pre_prompt_length_list = [35]
                self.model.token_length_list = [inputs_embeds.shape[1]]
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0

        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        # Record layer visual tokens for FLOPS calculation (for STAR models)
        if hasattr(self.model, 'layer_visual_tokens') and self.model.layer_visual_tokens:
            self.record_layer_tokens(self.model.layer_visual_tokens)

        return output, visual_token_num

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        texts = kwargs.pop("texts", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if texts is not None:
            inputs["texts"] = texts
        return inputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)