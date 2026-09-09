# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Haotian Liu
# Copyright 2026 STAR-Pro contributors
# Licensed under the Apache License, Version 2.0.
# Baseline decoder adapted from the STAR-Pro evaluation implementation.

import torch
from typing import Dict
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from .baseline_runtime import baseline_forward, initialize_baseline


R_dict = {
    "7b": {
        2: {
            256: 235,
            192: 166,
            128: 98,
            64: 29,
        },
        3: {
            256: 223,
            192: 152,
            128: 81,
            64: 11,
        },
    },
    "13b": {
        2: {
            192: 171,
            128: 104,
            64: 37,
        },
        3: {
            192: 160,
            128: 91,
            64: 22,
        }
    }
}


class FastVLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, fastv_config: Dict):
        super().__init__(config)
        self.system_prompt_length = None
        self.visual_token_num = 0

        initialize_baseline(self, config)

        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880
            self.anyres = True

        # FastV config
        self.K = fastv_config["K"]
        if self.K != 2 or fastv_config.get("T") not in (64, 128) or "R" in fastv_config:
            raise ValueError("Released FastV uses K=2 and per-crop nominal T=64 or 128")
        self.R = R_dict[self.scale][self.K][fastv_config["T"]]
        if self.anyres:
            self.R *= 5
        self.pruning_locations = (self.K,)
        self.compact_decode_positions = False

    def prepare_pruning(self, hidden_states, padding_mask):
        return None

    def prune_layer(self, layer, hidden_states, attention, position_ids,
                    padding_mask, visual_length, state):
        start = self.system_prompt_length
        end = start + visual_length
        # FastV ranks visual keys using the final query token at layer K.
        image_attention = attention.mean(dim=1)[0, -1, start:end]
        visual_index = torch.sort(torch.topk(image_attention, k=self.R).indices).values
        indices = torch.cat((
            torch.arange(start, device=hidden_states.device), visual_index + start,
            torch.arange(end, hidden_states.shape[1], device=hidden_states.device),
        ))
        return (hidden_states[:, indices], position_ids[:, indices],
                padding_mask[:, indices], self.R)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return baseline_forward(
            self, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
