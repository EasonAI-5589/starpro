import torch
import einops as ein
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast

#暂时使用SparseVLM的R_dict，后续需要修改
#注意，这里的R_dict只有前面一些是对的，其他后面需要改正
# R_dict = { #这个是使用5%mergetoken的一些配置
#     "7b": {
#         2: {
#             192: 240,
#             128: 240,
#             64: 116,
#             32: 64,
#         },
#         8: {
#             192: 216,
#             128: 192,
#             64: 68,
#             32: 40,
#         },
#         16: {
#             192: 192,
#             128: 56, #标准的128配置
#             64: 16,
#             32: 24,
#         },
#         24: {
#             192: 108,
#             128: 0,
#             64: 0,
#             32: 16,
#         },
#     },
#     "13b": {
#         2: {
#             192: 300,
#             128: 300,
#             64: 70,
#         },
#         8: {
#             192: 200,
#             128: 100,
#             64: 35,
#         },
#         20: {
#             192: 110,
#             128: 40,
#             64: 20,
#         },
#     }
# }

R_dict = { #这个是STAR_V2的配置
    "7b": {
        2: {
            192: 240,
            128: 288,
            64: 128,
            32: 64,  # patched 20260724: STAR_V2 table had no 32-token entry (KeyError on MMMU/SEED @32);
        },           # value taken from the disabled 5%-merge-token table above, NOT STAR_V2-calibrated.
        8: {
            192: 216,
            128: 192,
            64: 72,
            32: 40,  # patched 20260724, see note on layer-2 entry above.
        },
        16: {
            192: 192,
            128: 32,
            64: 16,
            32: 24,  # patched 20260724, see note on layer-2 entry above.
        },
        24: {
            192: 192,
            128: 0,
            64: 0,
            32: 16,  # patched 20260724, see note on layer-2 entry above.
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

    # # add a little noise to ensure no tokens have the same density.
    # density += torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

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

    # token_weight = x.new_ones(B, N, 1).float()
    token_weight = torch.ones(B, N, 1)
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N).cpu(), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6

    # average token features
    norm_weight = token_weight / all_weight[idx.cpu()]
    x_source = x.cpu() * norm_weight

    # x_merged = x.new_zeros(B * cluster_num, C).float()
    x_merged = torch.zeros(B * cluster_num, C)
    x_merged.index_add_(dim=0, index=idx.reshape(B * N).cpu(), source=x_source.reshape(B * N, C))
    x_merged = x_merged.reshape(B, cluster_num, C)
    
    return x_merged.to(device=x.device, dtype=x.dtype)


class HoloV_2_LlamaModel(LlamaModel):
    """
    HoloV_2 Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    """

    def __init__(self, config: LlamaConfig, holov2_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0
        
        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        self.retained_num = holov2_config["T"]
        #使用retained_num*2作为visual_token_length，因为S1阶段要保留的token数量就是retained_num*2
        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576 // 2
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880 // 2
            self.anyres = True

        # PrefixVLM config
        if config.num_hidden_layers == 32:
            self.pruning_loc = [2, 8, 16, 24]
        elif config.num_hidden_layers == 40:
            self.pruning_loc = [2, 8, 20]
    
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
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
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
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
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # SparseVLM
        if seq_length > 1:
            visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
            text_tokens = hidden_states[:, self.system_prompt_length+self.visual_token_length:]
            matrix = text_tokens @ visual_tokens.transpose(1, 2)
            matrix = matrix.squeeze(0).softmax(0).mean(1)
            text_rater_index = torch.where(matrix > matrix.mean())[0]

            visual_token_length = self.visual_token_length
            visual_token_num = 0
            visual_token_list = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

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
                # # PrefixVLM
                # if seq_length > 1:
                #     visual_token_num += visual_token_length
                #     visual_token_list.append(visual_token_length)
                    
                #     if (decoder_layer.self_attn.layer_idx + 1) in self.pruning_loc:
                #         attn_mask = torch.ones((batch_size, hidden_states.shape[1]), device=hidden_states.device)
                #         attn_mask = _prepare_4d_causal_attention_mask(attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                #         layer_outputs = decoder_layer(
                #             hidden_states,
                #             attention_mask=attn_mask,
                #             position_ids=position_ids,
                #             past_key_value=past_key_values,
                #             output_attentions=True,
                #             use_cache=use_cache,
                #         )
                #         hidden_states = layer_outputs[0]

                #         self_attn_weights = layer_outputs[1].mean(1)
                #         cross_attn_weights = self_attn_weights[:, text_rater_index+self.system_prompt_length+visual_token_length, 
                #                                                self.system_prompt_length:self.system_prompt_length+visual_token_length]
                #         cross_attn_weights = cross_attn_weights.mean(1)

                #         retained_visual_tokens = torch.zeros_like(cross_attn_weights, dtype=bool)
                #         dynamic_res = 5 if self.anyres else 1
                #         retained_num = min(
                #             R_dict[self.scale][decoder_layer.self_attn.layer_idx+1][self.retained_num] * dynamic_res,
                #             visual_token_length - 1
                #         )
                #         #直接保留后面的token，剪枝前面的若干token
                #         retained_visual_index = torch.arange(
                #             visual_token_length - retained_num,
                #             visual_token_length,
                #             device=hidden_states.device
                #         ).unsqueeze(0)
                #         retained_visual_tokens[0][retained_visual_index] = 1
                        
                #         total_visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length+visual_token_length]
                #         sparse_visual_tokens = total_visual_tokens[:, torch.where(retained_visual_tokens == 0)[1]]
                #         sparse_attn_weights = cross_attn_weights[:, torch.where(retained_visual_tokens == 0)[1]]
                #         merge_num = max(int(sparse_attn_weights.shape[1] * 0.3), 1)
                #         merge_token_index = sparse_attn_weights.topk(merge_num).indices

                #         merge_visual_tokens = sparse_visual_tokens[:, merge_token_index.squeeze(0)]
                #         cluster_num = max(int(merge_visual_tokens.shape[1] * 0.1), 1)
                #         if cluster_num == 0:
                #             cluster_num = merge_visual_tokens.shape[1]
                #         merge_sparse_tokens = cluster_and_merge(merge_visual_tokens, cluster_num)

                #         retained_visual_index = torch.sort(retained_visual_index).values
                #         select_visual_tokens = total_visual_tokens[:, retained_visual_index.squeeze(0)]
                #         #拼接的时候，注意顺序：sys-(prune-merge)-retain-text
                #         hidden_states = torch.cat((
                #             hidden_states[:, :self.system_prompt_length],
                #             merge_sparse_tokens,
                #             select_visual_tokens,
                #             hidden_states[:, self.system_prompt_length+visual_token_length:]
                #         ), dim=1)
                #         position_ids = position_ids[:,:hidden_states.shape[1]]

                #         layer_outputs = (hidden_states, layer_outputs[2])
                #         visual_token_length = select_visual_tokens.shape[1] + merge_sparse_tokens.shape[1]
                #         print(f"layer_idx: {decoder_layer.self_attn.layer_idx+1}, visual_token_length: {visual_token_length}, retained_num: {retained_num}, merge_num: {merge_num}, cluster_num: {cluster_num}")
                
                # PrefixVLM+
                #不使用cross-attention，而是使用sim来挑选merge token
                if seq_length > 1:
                    visual_token_num += visual_token_length
                    visual_token_list.append(visual_token_length)
                    
                    if (decoder_layer.self_attn.layer_idx + 1) in self.pruning_loc:
                        attn_mask = torch.ones((batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        hidden_states = layer_outputs[0]

                        self_attn_weights = layer_outputs[1].mean(1)
                        cross_attn_weights = self_attn_weights[:, text_rater_index+self.system_prompt_length+visual_token_length, 
                                                               self.system_prompt_length:self.system_prompt_length+visual_token_length]
                        cross_attn_weights = cross_attn_weights.mean(1)

                        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        #0从前往后剪枝，retained_token+merge_token
                        #1439
                        # retained_visual_tokens = torch.zeros_like(cross_attn_weights, dtype=bool)
                        # dynamic_res = 5 if self.anyres else 1
                        # retained_num = min(
                        #     R_dict[self.scale][decoder_layer.self_attn.layer_idx+1][self.retained_num] * dynamic_res,
                        #     visual_token_length - 1
                        # )
                        # #直接保留后面的token，剪枝前面的若干token
                        # retained_visual_index = torch.arange(
                        #     visual_token_length - retained_num,
                        #     visual_token_length,
                        #     device=hidden_states.device
                        # ).unsqueeze(0)
                        # retained_visual_tokens[0][retained_visual_index] = 1
                        
                        # total_visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length+visual_token_length]
                        # sparse_visual_tokens = total_visual_tokens[:, torch.where(retained_visual_tokens == 0)[1]]

                        # retained_visual_index = torch.sort(retained_visual_index).values
                        # select_visual_tokens = total_visual_tokens[:, retained_visual_index.squeeze(0)]

                        # sparse_visual_tokens_norm = torch.nn.functional.normalize(sparse_visual_tokens, p=2, dim=-1)
                        # if select_visual_tokens.shape[1] > 0:
                        #     select_visual_tokens_norm = torch.nn.functional.normalize(select_visual_tokens, p=2, dim=-1)
                        #     sim_matrix = torch.matmul(
                        #         sparse_visual_tokens_norm,
                        #         select_visual_tokens_norm.transpose(1, 2)
                        #     )
                        #     max_sim = sim_matrix.max(dim=-1).values
                        #     merge_scores = 1 - max_sim #使用和后面的视觉token的相似度的最大值的负数+1作为score挑选merge token
                        # else:
                        #     text_tokens = hidden_states[:, self.system_prompt_length+visual_token_length:]
                        #     text_tokens_norm = torch.nn.functional.normalize(text_tokens, p=2, dim=-1)
                        #     sim_matrix = torch.matmul(
                        #         sparse_visual_tokens_norm,
                        #         text_tokens_norm.transpose(1, 2)
                        #     )
                        #     max_sim = sim_matrix.max(dim=-1).values
                        #     merge_scores = 1 - max_sim#如果后面没有视觉token了，使用和后面的文本token的相似度计算score
                        
                        # merge_num = max(int(sparse_visual_tokens.shape[1] * 0.05), 1)
                        # merge_token_index = merge_scores.topk(merge_num).indices

                        # merge_visual_tokens = sparse_visual_tokens[:, merge_token_index.squeeze(0)]
                        # # cluster_num = max(int(merge_visual_tokens.shape[1] * 0.1), 1)
                        # # if cluster_num == 0:
                        # #     cluster_num = merge_visual_tokens.shape[1]
                        # # merge_sparse_tokens = cluster_and_merge(merge_visual_tokens, cluster_num)
                        # merge_sparse_tokens = merge_visual_tokens
                        # cluster_num = merge_num
                        # #拼接的时候，注意顺序：sys-(prune-merge)-retain-text
                        # hidden_states = torch.cat((
                        #     hidden_states[:, :self.system_prompt_length],
                        #     merge_sparse_tokens,
                        #     select_visual_tokens,
                        #     hidden_states[:, self.system_prompt_length+visual_token_length:]
                        # ), dim=1)
                        # position_ids = position_ids[:,:hidden_states.shape[1]]

                        # layer_outputs = (hidden_states, layer_outputs[2])
                        # visual_token_length = select_visual_tokens.shape[1] + merge_sparse_tokens.shape[1]
                        # print(f"layer_idx: {decoder_layer.self_attn.layer_idx+1}, visual_token_length: {visual_token_length}, retained_num: {retained_num}, merge_num: {merge_num}, cluster_num: {cluster_num}")
                        
                        
                        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        #1) 用“平均相似度最低”来选 retained_token（更分散/更不相似的token）
                        #不从前往后剪枝
                        #使用retained_token和merge_token
                        #1449很work
                        # total_visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length + visual_token_length]  # [B, V, D]
                        # B, V, D = total_visual_tokens.shape

                        # retained_visual_tokens = torch.zeros((B, V), device=hidden_states.device, dtype=torch.bool)

                        # dynamic_res = 5 if self.anyres else 1
                        # retained_num = min(
                        #     R_dict[self.scale][decoder_layer.self_attn.layer_idx + 1][self.retained_num] * dynamic_res,
                        #     V - 1
                        # )

                        # # cosine 相似度矩阵： [B, V, V]
                        # vis_norm = torch.nn.functional.normalize(total_visual_tokens, p=2, dim=-1)
                        # sim_vv = torch.matmul(vis_norm, vis_norm.transpose(1, 2))  # cosine

                        # # 去掉对角线（自己和自己相似度=1），不参与平均
                        # eye = torch.eye(V, device=sim_vv.device, dtype=torch.bool).unsqueeze(0)  # [1, V, V]
                        # sim_vv = sim_vv.masked_fill(eye, 0.0)

                        # # 每个 token 对其它 token 的平均相似度：越低越“独特”
                        # avg_sim = sim_vv.sum(dim=-1) / max(V - 1, 1)  # [B, V]

                        # # 选平均相似度最低的 retained_num 个
                        # retained_visual_index = avg_sim.topk(k=retained_num, largest=False, dim=-1).indices  # [B, retained_num]
                        # retained_visual_tokens.scatter_(1, retained_visual_index, True)

                        # # 为了保持 token 原始相对顺序：排序 retained 的 index
                        # retained_visual_index = torch.sort(retained_visual_index, dim=-1).values  # [B, retained_num]
                        # select_visual_tokens = total_visual_tokens.gather(
                        #     1, retained_visual_index.unsqueeze(-1).expand(-1, -1, D)
                        # )  # [B, retained_num, D]

                        # # sparse = 非 retained 的视觉 token（候选 merge / prune）
                        # sparse_mask = ~retained_visual_tokens  # [B, V]
                        # sparse_idx = torch.where(sparse_mask[0])[0]  # [Ns] (batch_size=1 假设)
                        # sparse_visual_tokens = total_visual_tokens[:, sparse_idx]  # [B, Ns, D]

                        # # === 2) 计算 merge_scores，再选 merge_token（保持 sys-(merge)-retain-text 的拼接顺序不变） ===
                        # sparse_visual_tokens_norm = torch.nn.functional.normalize(sparse_visual_tokens, p=2, dim=-1)

                        # if select_visual_tokens.shape[1] > 0:
                        #     select_visual_tokens_norm = torch.nn.functional.normalize(select_visual_tokens, p=2, dim=-1)
                        #     sim_matrix = torch.matmul(
                        #         sparse_visual_tokens_norm,
                        #         select_visual_tokens_norm.transpose(1, 2)
                        #     )  # [B, Ns, retained_num]
                        #     max_sim = sim_matrix.max(dim=-1).values  # [B, Ns]
                        #     merge_scores = 1 - max_sim
                        # else:
                        #     # 极端情况：retained_num=0（一般不会发生），退化为跟 text token 比
                        #     text_tokens = hidden_states[:, self.system_prompt_length + visual_token_length:]
                        #     text_tokens_norm = torch.nn.functional.normalize(text_tokens, p=2, dim=-1)
                        #     sim_matrix = torch.matmul(
                        #         sparse_visual_tokens_norm,
                        #         text_tokens_norm.transpose(1, 2)
                        #     )  # [B, Ns, T]
                        #     max_sim = sim_matrix.max(dim=-1).values  # [B, Ns]
                        #     merge_scores = 1 - max_sim

                        # merge_num = max(int(sparse_visual_tokens.shape[1] * 0.05), 1)
                        # merge_num = min(merge_num, sparse_visual_tokens.shape[1])

                        # # topk 得到的是 sparse 内部索引（不一定有序）
                        # merge_rel_idx = merge_scores.topk(merge_num, largest=True, dim=-1).indices  # [B, merge_num]
                        # # 为了保持 merge token 的原始相对顺序：映射回视觉 token 的“绝对位置”再排序
                        # merge_abs_idx = sparse_idx[merge_rel_idx[0]]  # [merge_num]
                        # merge_abs_idx = torch.sort(merge_abs_idx).values
                        # retained_abs_idx = retained_visual_index[0]  # [R]  (batch_size=1)
                        # # 合并并排序（去重，避免 retained/merge 交叉重复）
                        # kept_abs_idx = torch.unique(torch.cat([retained_abs_idx, merge_abs_idx], dim=0))
                        # kept_abs_idx_sorted = torch.sort(kept_abs_idx).values  # [K]
                        # # 新视觉段：按原视觉段位置顺序取 token（这样就“插回原位置顺序”了）
                        # new_visual_tokens = total_visual_tokens[:, kept_abs_idx_sorted]  # [B, K, D]

                        # # =========================
                        # # 拼接：sys - (new_visual in original order) - text
                        # # =========================
                        # hidden_states = torch.cat((
                        #     hidden_states[:, :self.system_prompt_length],
                        #     new_visual_tokens,
                        #     hidden_states[:, self.system_prompt_length + visual_token_length:]
                        # ), dim=1)

                        # position_ids = position_ids[:, :hidden_states.shape[1]]
                        # layer_outputs = (hidden_states, layer_outputs[2])

                        # # 更新 visual_token_length
                        # visual_token_length = new_visual_tokens.shape[1]

                        # print(
                        #     f"layer_idx: {decoder_layer.self_attn.layer_idx+1}, "
                        #     f"visual_token_length: {visual_token_length}, "
                        #     f"retained_num: {retained_abs_idx.numel()}, "
                        #     f"merge_num: {merge_abs_idx.numel()}"
                        # )
            
            
                        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        #2: score = λ * attn + (1-λ) * diversity
                        #只用retained_token
                        # keep top-K as retained tokens (no merge)
                        #lambda=0.5结果是1464很好
                        import os
                        lambda_val = float(os.environ.get("LAMBDA", "0.5"))  # λ越大越偏向attn，越小越偏向多样性

                        total_visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length + visual_token_length]  # [B,V,D]
                        B, V, D = total_visual_tokens.shape
                        device = hidden_states.device

                        dynamic_res = 5 if self.anyres else 1
                        retained_num = min(
                            R_dict[self.scale][decoder_layer.self_attn.layer_idx + 1][self.retained_num] * dynamic_res,
                            V - 1
                        )
                        retained_num = R_dict[self.scale][decoder_layer.self_attn.layer_idx + 1][self.retained_num] * dynamic_res
                        #1) attn importance (higher better)
                        attn = cross_attn_weights
                        if attn.dim() == 1:
                            attn = attn.unsqueeze(0)  # (1,V)
                        attn = (attn - attn.min(dim=1, keepdim=True).values) / (
                            attn.max(dim=1, keepdim=True).values - attn.min(dim=1, keepdim=True).values + 1e-8
                        )  # (B,V) in [0,1]

                        #2) normalized features for cosine
                        vis_norm = torch.nn.functional.normalize(total_visual_tokens, p=2, dim=-1)  # (B,V,D)

                        all_selected = []
                        for b in range(B):
                            a = attn[b]            # (V,)
                            feat = vis_norm[b]     # (V,D)

                            available = torch.ones(V, dtype=torch.bool, device=device)
                            selected = []

                            # if retained_num > 0:
                            # step0: pick highest attention
                            i0 = torch.argmax(a).item()
                            selected.append(i0)     #最少保留了一个image token
                            available[i0] = False

                            max_sim_to_S = feat @ feat[i0]    # (V,) cosine
                            max_sim_to_S[i0] = 1.0            # 自己的sim设为1（反正不可选了）

                            for step in range(1, retained_num):
                                if not available.any():
                                    break

                                diversity = 1.0 - max_sim_to_S             # 越大越好
                                # 对diversity进行minmax归一化
                                # diversity=(diversity-diversity.min())/(diversity.max()-diversity.min()+1e-8)
                                scores = (1-lambda_val) * a + lambda_val * diversity

                                scores[~available] = -float("inf")
                                idx = torch.argmax(scores).item()

                                selected.append(idx)
                                available[idx] = False

                                # 更新 m_i = max(m_i, sim(i, idx))
                                sim_vec = feat @ feat[idx]                 # (V,)
                                max_sim_to_S = torch.maximum(max_sim_to_S, sim_vec)
                                max_sim_to_S[idx] = 1.0

                            selected_t = torch.tensor(selected, device=device, dtype=torch.long)
                            selected_t = torch.sort(selected_t).values     # 保持原视觉顺序
                            all_selected.append(selected_t)

                        # batch=1 常见：直接取第0个
                        retained_visual_index = all_selected[0].unsqueeze(0)  # (B,R)

                        select_visual_tokens = total_visual_tokens.gather(
                            1, retained_visual_index.unsqueeze(-1).expand(-1, -1, D)
                        )  # (B,R,D)

                        # ---- rebuild: sys - retained_visual - text ----
                        hidden_states = torch.cat((
                            hidden_states[:, :self.system_prompt_length],
                            select_visual_tokens,
                            hidden_states[:, self.system_prompt_length + visual_token_length:]
                        ), dim=1)

                        position_ids = position_ids[:, :hidden_states.shape[1]]
                        layer_outputs = (hidden_states, layer_outputs[2])
                        visual_token_length = select_visual_tokens.shape[1]

                        # ---- debug: avg selected attn & avg pairwise sim ----
                        # 选中token对应的attn均值
                        avg_attn = a[selected_t].mean().item()
                        # 选中token两两相似度均值（去掉对角线）
                        if selected_t.numel() > 1:
                            sel_feat = feat[selected_t]                # (R, D)  这里feat是vis_norm[b]
                            sel_sim = sel_feat @ sel_feat.t()          # (R, R)
                            mask = ~torch.eye(sel_sim.size(0), dtype=torch.bool, device=sel_sim.device)
                            avg_sim = sel_sim[mask].mean().item()
                        else:
                            avg_sim = 0.0
                        print(
                            f"layer_idx: {decoder_layer.self_attn.layer_idx+1}, "
                            f"visual_token_length: {visual_token_length}, "
                            f"retained_num: {retained_num}, "
                            f"lambda_val: {lambda_val}, "
                            f"avg_attn: {avg_attn:.4f}, "
                            f"avg_sim: {avg_sim:.4f}"
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

                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=torch.tensor([[
                            past_key_values.get_usable_length(1, decoder_layer.self_attn.layer_idx)]], 
                            dtype=torch.int64).cuda(),
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
            self.visual_token_num = visual_token_num / len(self.layers)

        # add hidden states from the last decoder layer
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
