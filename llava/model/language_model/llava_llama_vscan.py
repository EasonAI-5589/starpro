"""
VScan LLaVA-Llama Integration Layer

Source: Official VScan implementation
Reference: https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan

This file integrates VScan with LLaVA architecture for training-free visual token reduction.

Key features:
- Stage 1 (in llava_arch.py): Complementary global and local scans
- Stage 2 (in modeling_llama_vscan.py): Middle layer pruning based on attention scores

The LlavaLlamaForCausalLM_VScan class provides the entry point for VScan inference.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig, VScanLlavaLlamaModel


class VScanLlavaConfig(LlavaLlamaConfig):
    """Configuration class for VScan LLaVA model."""
    model_type = "vscan_llava_llama"


class LlavaLlamaForCausalLM_VScan(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    VScan LLaVA for Causal Language Modeling.

    This class integrates VScan with LLaVA:
    1. VScan Stage 1 (via prepare_inputs_labels_for_multimodal_x)
    2. VScan Stage 2 (via forward_x with layer pruning)

    Reference: VScan llava_llama_x.py:LlavaLlamaForCausalLM_X
    """
    config_class = VScanLlavaConfig

    def __init__(self, config, vscan_config: dict = None, **kwargs):
        # Default VScan config
        if vscan_config is None:
            vscan_config = {
                "stage1_tokens": 96,
                "stage2_tokens": 32,
                "prune_layer": 16,
            }

        # Initialize LlamaForCausalLM parent (skip its model creation)
        super(LlamaForCausalLM, self).__init__(config)

        # Create VScan-enabled LLaVA model
        self.model = VScanLlavaLlamaModel(config, vscan_config)

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # VScan configuration
        self.vscan_config = vscan_config
        self.visual_token_num = vscan_config.get("stage1_tokens", 96)

        # Set up Stage 2 parameters on model
        self._setup_vscan_model_params(vscan_config)

        # Latency tracking
        self.start_latency = False
        self.phase = "prefill"
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.prefill_latency = 0.0
        self.decode_latency = 0.0

        # FLOPS tracking (for compatibility with eval scripts)
        self.track_flops = False
        self.layer_visual_tokens = []
        self.total_flops = 0.0
        self.flops_count = 0

        # Initialize weights
        self.post_init()

    def get_model(self):
        return self.model

    def _setup_vscan_model_params(self, vscan_config):
        """Set up VScan parameters on model for Stage 2 pruning."""
        # Model scale detection
        num_layers = self.config.num_hidden_layers
        if num_layers == 32:
            scale = "7b"
        elif num_layers == 40:
            scale = "13b"
        else:
            scale = "unknown"

        stage1_tokens = vscan_config.get("stage1_tokens", 96)
        stage2_tokens = vscan_config.get("stage2_tokens", 32)
        prune_layer = vscan_config.get("prune_layer", 16 if scale == "7b" else 20)

        # For compatibility with official VScan forward_x/layer_prune
        self.model.layer_list = [prune_layer]  # Layers to prune at
        self.model.image_token_list = [stage1_tokens, stage2_tokens]  # Token counts at each stage
        self.model.image_token_posi = [-1]  # Image token positions (set during forward)
        self.model.prompt_len = None  # Prompt length (set during forward)
        self.model.image_tokens = [0]  # Number of image tokens per sample

        # Visual token tracking - use AVERAGE for display
        # For 7B (32 layers), prune at layer 16: avg = (stage1 * 16 + stage2 * 16) / 32
        visual_token_avg = (stage1_tokens + stage2_tokens) // 2
        self.model.visual_token_num = visual_token_avg
        self.model.visual_token_length = stage1_tokens

    def get_visual_token_num(self):
        return self.visual_token_num

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
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with VScan multimodal support.

        Uses prepare_inputs_labels_for_multimodal_x for Stage 1 token selection,
        then forward_x for Stage 2 middle-layer pruning.
        """
        if inputs_embeds is None:
            # Use VScan-specific multimodal preparation (Stage 1)
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal_x(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # Use forward_x for middle-layer pruning (Stage 2)
        outputs, labels = self.model.forward_x(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            selected_indices=None
        )

        # Compute logits
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
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

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text with VScan token reduction.

        Reference: VScan llava_llama_x.py:275-317
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        # Remove texts parameter (used by TRIM/CDP methods, not needed for VScan)
        kwargs.pop("texts", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal_x(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes
            )
        else:
            # No images - reset model state
            self.model.image_token_posi = [-1]
            self.model.prompt_len = None
            self.model.image_tokens = [0]
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # Use LlamaForCausalLM's generate with VScan-processed embeddings
        return LlamaForCausalLM.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), self.model.visual_token_num

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        Prepare inputs for generation, preserving image information.
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
AutoConfig.register("vscan_llava_llama", VScanLlavaConfig)
AutoModelForCausalLM.register(VScanLlavaConfig, LlavaLlamaForCausalLM_VScan)

# Backward compatibility aliases
VScanLlavaLlamaForCausalLM = LlavaLlamaForCausalLM_VScan
