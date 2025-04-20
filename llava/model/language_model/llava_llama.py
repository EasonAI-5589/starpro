#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .modeling_fastv_llama import FastVLlamaModel
from .modeling_sparse_llama import SparseLlamaModel
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class FastVLlavaLlamaModel(LlavaMetaModel, FastVLlamaModel):
    # Alter LlavaLlamaModel to FastVLlavaLlamaModel
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, fastv_config: dict):
        super(FastVLlavaLlamaModel, self).__init__(config, fastv_config=fastv_config)


class SparseLlavaLlamaModel(LlavaMetaModel, SparseLlamaModel):
    # Alter LlavaLlamaModel to SparseLlavaLlamaModel
    config_class = LlavaLlamaConfig
    def __init__(self, config: LlamaConfig, sparsevlm_config: dict):
        super(SparseLlavaLlamaModel, self).__init__(config, sparsevlm_config=sparsevlm_config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config, pruning_method=None, visual_token_num=None, 
                 use_fastv=False, fastv_config=None,
                 use_sparsevlm=False, sparsevlm_config=None):
        super(LlamaForCausalLM, self).__init__(config)
        if use_fastv:
            print(f"Use FastV: {fastv_config}")
            self.model = FastVLlavaLlamaModel(config, fastv_config)
        elif use_sparsevlm:
            print(f"Use SparseVLM: {sparsevlm_config}")
            self.model = SparseLlavaLlamaModel(config, sparsevlm_config=sparsevlm_config)
        else:
            self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Visual Token Pruning config
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num

        # Initialize weights and apply final processing
        self.post_init()
    
    # Visual Token Pruning
    def get_pruning_method(self):
        return self.pruning_method

    # Visual Token Pruning
    def get_visual_token_num(self):
        return self.visual_token_num

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
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
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
                image_sizes=image_sizes
            )
            if self.pruning_method == "fastv":
                visual_token_num = self.visual_token_num
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), visual_token_num

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
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

AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
