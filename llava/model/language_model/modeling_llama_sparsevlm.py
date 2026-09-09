# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Haotian Liu
# Copyright 2026 STAR-Pro contributors
# Licensed under the Apache License, Version 2.0.
# Baseline decoder adapted from the STAR-Pro evaluation implementation.

import torch
import einops as ein
from typing import Dict
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from .baseline_runtime import baseline_forward, initialize_baseline


R_dict = {
    "7b": {
        2: {
            192: 300,
            128: 300,
            64: 65,
        },
        6: {
            192: 200,
            128: 110,
            64: 30,
        },
        15: {
            192: 110,
            128: 35,
            64: 15,
        },
    },
    "13b": {
        2: {
            192: 300,
            128: 300,
            64: 70,
        },
        8: {
            192: 200,
            128: 100,
            64: 35,
        },
        20: {
            192: 110,
            128: 40,
            64: 20,
        },
    }
}


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_and_merge(x, cluster_num):
    B, N, C = x.shape

    x1 = ein.rearrange(x, "b l r -> b l () r")
    x2 = ein.rearrange(x, "b l r -> b () l r")
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)

    # get local density
    dist_nearest = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False).values
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

    # get distance indicator
    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1).values

    # select clustering center according to score
    score = dist * density
    index_down = torch.topk(score, k=cluster_num, dim=-1).indices

    # assign tokens to the nearest center
    dist_matrix = index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix.argmin(dim=1)

    # make sure cluster center merge to itself
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # merge tokens
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    token_weight = torch.ones(B, N, 1)
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N).cpu(), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6

    # average token features
    norm_weight = token_weight / all_weight[idx.cpu()]
    x_source = x.cpu() * norm_weight

    x_merged = torch.zeros(B * cluster_num, C)
    x_merged.index_add_(dim=0, index=idx.reshape(B * N).cpu(), source=x_source.reshape(B * N, C))
    x_merged = x_merged.reshape(B, cluster_num, C)

    return x_merged.to(device=x.device, dtype=x.dtype)


class SparseLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, sparsevlm_config: Dict):
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

        # SparseVLM config
        if config.num_hidden_layers == 32:
            self.pruning_loc = [2, 6, 15]
        elif config.num_hidden_layers == 40:
            self.pruning_loc = [2, 8, 20]
        self.retained_num = sparsevlm_config["T"]
        if self.retained_num not in (64, 128):
            raise ValueError("Released SparseVLM uses per-crop nominal T=64 or 128")
        self.pruning_locations = tuple(self.pruning_loc)
        self.compact_decode_positions = True

    def prepare_pruning(self, hidden_states, padding_mask):
        start = self.system_prompt_length
        end = start + self.visual_token_length
        visual_tokens = hidden_states[:, start:end]
        text_tokens = hidden_states[:, end:]
        valid_text = torch.where(padding_mask[0, end:])[0]
        if not valid_text.numel():
            raise ValueError("SparseVLM requires unmasked text after the image")
        matrix = text_tokens[:, valid_text] @ visual_tokens.transpose(1, 2)
        matrix = matrix.squeeze(0).softmax(0).mean(1)
        raters = torch.where(matrix > matrix.mean())[0]
        if not raters.numel():
            # The original strict threshold has no rater for a one-token or
            # exactly tied query. Use its highest-scoring text token then.
            raters = matrix.argmax().reshape(1)
        return valid_text[raters]

    def prune_layer(self, layer, hidden_states, attention, position_ids,
                    padding_mask, visual_length, text_raters):
        start = self.system_prompt_length
        end = start + visual_length
        self_attention = attention.mean(1)
        cross_attention = self_attention[:, text_raters + end, start:end].mean(1)
        retained_mask = torch.zeros_like(cross_attention, dtype=torch.bool)
        multiplier = 5 if self.anyres else 1
        retained_index = torch.topk(
            cross_attention,
            min(R_dict[self.scale][layer][self.retained_num] * multiplier, visual_length - 1),
            dim=1,
        ).indices
        retained_mask.scatter_(1, retained_index, True)

        visual_tokens = hidden_states[:, start:end]
        sparse_index = torch.where(~retained_mask[0])[0]
        sparse_tokens = visual_tokens[:, sparse_index]
        sparse_attention = cross_attention[:, sparse_index]
        merge_num = max(int(sparse_attention.shape[1] * 0.3), 1)
        merge_index = sparse_attention.topk(merge_num).indices
        merge_tokens = sparse_tokens[:, merge_index.squeeze(0)]
        cluster_num = max(int(merge_tokens.shape[1] * 0.1), 1)
        merged = cluster_and_merge(merge_tokens, cluster_num)

        retained_index = torch.sort(retained_index).values
        selected = visual_tokens[:, retained_index.squeeze(0)]
        new_visual_length = selected.shape[1] + merged.shape[1]
        hidden_states = torch.cat((
            hidden_states[:, :start], selected, merged, hidden_states[:, end:],
        ), dim=1)
        padding_mask = torch.cat((
            padding_mask[:, :start],
            padding_mask.new_ones((1, new_visual_length)), padding_mask[:, end:],
        ), dim=1)
        # Preserve the original SparseVLM position-compaction convention.
        position_ids = position_ids[:, :hidden_states.shape[1]]
        return hidden_states, position_ids, padding_mask, new_visual_length

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
