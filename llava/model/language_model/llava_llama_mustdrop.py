"""
MustDrop LLaVA-Llama Integration Layer

Source: MustDrop official implementation
Reference: /tmp/MustDrop/llava/model/language_model/sparse_llava_llama.py

This file integrates MustDrop pruning with LLaVA architecture.
Key design: Must inherit from LlamaDynamicvitForCausalLM to use its custom generate()/greedy_search().
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .modelling_llama_mustdrop import LlamaDynamicvitModel, LlamaDynamicvitForCausalLM


class MustDropLlavaConfig(LlamaConfig):
    """Configuration class for MustDrop LLaVA model."""
    model_type = "mustdrop_llava_llama"


class MustDropLlavaLlamaModel(LlavaMetaModel, LlamaDynamicvitModel):
    """
    MustDrop integrated with LLaVA model.

    Combines:
    - LlavaMetaModel: Vision encoder and multimodal projection
    - LlamaDynamicvitModel: LLM with Dual Attention Filter pruning

    Reference: MustDrop sparse_llava_llama.py:23-27
    """
    config_class = MustDropLlavaConfig

    def __init__(self, config: LlamaConfig):
        super(MustDropLlavaLlamaModel, self).__init__(config)


class MustDropLlavaLlamaForCausalLM(LlamaDynamicvitForCausalLM, LlavaMetaForCausalLM):
    """
    MustDrop LLaVA for Causal Language Modeling.

    CRITICAL: Must inherit from LlamaDynamicvitForCausalLM FIRST to use its custom
    generate() and greedy_search() methods that pass MustDrop parameters through
    the inference pipeline.

    This class integrates:
    1. MustDrop Vision Encoder modifications (via multimodal_encoder)
    2. MustDrop LLM pruning (via LlamaDynamicvitForCausalLM)
    3. LLaVA multimodal capabilities (via LlavaMetaForCausalLM)

    Key features:
    - Token merging in Vision Encoder Layer 0
    - Key Token Set extraction in Vision Encoder Layer 23
    - Dual Attention Filter pruning in LLM layers [2, 6, 10, 14]
    - Key Token Set protection during LLM pruning
    - Progressive KV Cache sparsification

    Reference: MustDrop sparse_llava_llama.py:30-175
    """
    config_class = MustDropLlavaConfig

    def __init__(self, config, **kwargs):
        # Initialize parent class (LlamaDynamicvitForCausalLM)
        # Note: kwargs may contain extra parameters like pruning_method, visual_token_num, etc.
        # These are safely ignored here
        LlamaDynamicvitForCausalLM.__init__(self, config)

        # Replace model with MustDrop LLaVA model
        self.model = MustDropLlavaLlamaModel(config)

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MustDrop specific state
        # Reference: sparse_llava_llama.py:39-42
        self.img_seq = 576
        self.key_set = None
        self.token_length_list = []
        self.pre_prompt_length_list = []

        # Latency tracking (for compatibility with eval metrics)
        self.start_latency = False
        self.phase = "prefill"
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.prefill_latency = 0.0
        self.decode_latency = 0.0

        # FLOPS tracking (for compatibility with eval metrics)
        self.track_flops = False
        self.layer_visual_tokens = []
        self.total_flops = 0.0
        self.flops_count = 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        img_seq=576,
        token_length_list=[],
        pre_prompt_length_list=[],
        global_thr=1.0,
        individual_thr=0.0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with MustDrop multimodal support.

        Reference: MustDrop sparse_llava_llama.py:49-108
        """

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                img_seq,
                token_length_list,
                pre_prompt_length_list,
            ) = self.prepare_sparse_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            img_seq=self.img_seq,
            key_set=self.key_set,
            token_length_list=token_length_list,
            pre_prompt_length_list=pre_prompt_length_list,
            global_thr=global_thr,
            individual_thr=individual_thr
        )

    @torch.no_grad()
    def generate(
        self,
        global_thr,
        individual_thr,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        img_seq=576,
        token_length_list=[],
        pre_prompt_length_list=[],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text with MustDrop pruning.

        CRITICAL: This method must call super().generate() which goes to
        LlamaDynamicvitForCausalLM.generate() - NOT the standard Transformers generate().

        Reference: MustDrop sparse_llava_llama.py:110-161
        """
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
                img_seq,
                token_length_list,
                pre_prompt_length_list,
            ) = self.prepare_sparse_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # Call LlamaDynamicvitForCausalLM.generate() with MustDrop parameters
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            img_seq=img_seq,
            token_length_list=token_length_list,
            pre_prompt_length_list=pre_prompt_length_list,
            global_thr=global_thr,
            individual_thr=individual_thr,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        Prepare inputs for generation, preserving image information.

        Reference: MustDrop sparse_llava_llama.py:163-174
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


# Register model with AutoConfig and AutoModelForCausalLM
AutoConfig.register("mustdrop_llava_llama", MustDropLlavaConfig)
AutoModelForCausalLM.register(MustDropLlavaConfig, MustDropLlavaLlamaForCausalLM)


# Backward compatibility aliases
LlavaLlamaDynamicModel = MustDropLlavaLlamaModel
LlavaLlamaDynamicForCausalLM = MustDropLlavaLlamaForCausalLM
