import torch
import einops as ein
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast
R_dict = {  
    "7b": {
        8: {
            192: 192,
        },
        12:{256:128,
            128:64,
            64:32,
            32:16,
            16:8,
            8:4,},
        16: {
            192: 144,
        },
        24: {
            256: 64,
            192: 96,
            128: 32,
            64: 16,
            32: 8,
            16: 4,
            8: 2,
        },
    },
    "13b": {
        15: {
            192: 128,
            128: 64,
            64: 32,
            32: 16,
        },
        30: {
            192: 64,
            128: 32,
            64: 16,
            32: 8,
        },
    }
}


class Prefix_2_LlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, prefixvlm_2_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0

        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        self.retained_num = prefixvlm_2_config["T"]
        import os
        if config.image_aspect_ratio == "pad":
            print(f"config.image_aspect_ratio == 'pad'")
            self.visual_token_length = int(os.environ.get("STAGE1_KEEP", "288"))
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            print(f"config.image_aspect_ratio == 'anyres'")
            self.visual_token_length = int(os.environ.get("STAGE1_KEEP", "1280"))
            self.anyres = True

        # PrefixVLM config
        if config.num_hidden_layers == 32:
            if self.retained_num == 8:
                self.pruning_loc = [12,24]
            if self.retained_num == 16:
                self.pruning_loc = [12,24]
            if self.retained_num == 32:
                self.pruning_loc = [12,24]
            elif self.retained_num == 64:
                self.pruning_loc = [12, 24]
            elif self.retained_num == 128:
                self.pruning_loc = [12, 24]
            elif self.retained_num == 256:
                self.pruning_loc = [12, 24]
            elif self.retained_num == 192:
                self.pruning_loc = [8,16,24]
        elif config.num_hidden_layers == 40:
            self.pruning_loc = [15,30]

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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

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
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values)
            past_key_values_length = past_key_values.get_usable_length(
                seq_length)

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
            attention_mask = attention_mask if (
                attention_mask is not None and 0 in attention_mask) else None
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
                attention_mask, (batch_size,
                                 seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # SparseVLM
        if seq_length > 1:

            import os
            visual_token_length = self.visual_token_length
            visual_token_num = 0
            visual_token_list = []

            # ————————————————————————————————————————————————————————
            # 渐进式剪枝可视化：记录每次剪枝后保留的原始 token 索引
            self.pruning_history = []
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            _current_indices = torch.arange(visual_token_length, device=device)
            # ————————————————————————————————————————————————————————

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
                if seq_length > 1:
                    visual_token_num += visual_token_length
                    visual_token_list.append(visual_token_length)

                    if (decoder_layer.self_attn.layer_idx + 1) in self.pruning_loc:
                        attn_mask = torch.ones(
                            (batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(
                            attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        hidden_states = layer_outputs[0]
                        
                        # 1.动态计算 text_rater_index（循环内，因为 visual_token_length 每层在变）
                        visual_tokens = hidden_states[:,self.system_prompt_length:self.system_prompt_length + visual_token_length]
                        text_tokens = hidden_states[:,self.system_prompt_length + visual_token_length:]
                        matrix = text_tokens @ visual_tokens.transpose(1, 2)
                        matrix = matrix.squeeze(0).softmax(0).mean(1)
                        text_rater_index = torch.where(
                            matrix > matrix.mean())[0]
                        # print(f"[PrefixVLM_2 S2] text_rater_index: {text_rater_index}")
                        
                        # 2.使用所有文本token作为text_rater
                        # text_len = hidden_states.shape[1] - self.system_prompt_length - visual_token_length
                        # text_rater_index = torch.arange(text_len, device=hidden_states.device)
                        # print(f"[PrefixVLM_2 S2] text_rater_index: {text_rater_index}")
                        
                        # 3.使用最后一个文本token作为text_rater
                        # text_len = hidden_states.shape[1] - self.system_prompt_length - visual_token_length
                        # text_rater_index = torch.tensor([text_len - 1], device=hidden_states.device)
                        # print(f"[PrefixVLM_2 S2] text_rater_index: {text_rater_index}")

                        self_attn_weights = layer_outputs[1].mean(1)
                        cross_attn_weights = self_attn_weights[:, text_rater_index+self.system_prompt_length+visual_token_length,
                                                               self.system_prompt_length:self.system_prompt_length+visual_token_length]
                        cross_attn_weights = cross_attn_weights.mean(1)

                        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        import os
                        alpha_2 = float(os.environ.get("ALPHA_2", "0.5"))
                        lambda_2 = float(os.environ.get("LAMBDA_2", "0.1"))
                        coverage_method = os.environ.get("COVERAGE", "MEAN")
                        # print("S2使用的coverage方法是:",coverage_method)

                        # [B,V,D]
                        total_visual_tokens = hidden_states[:,self.system_prompt_length:self.system_prompt_length + visual_token_length]
                        B, V, D = total_visual_tokens.shape
                        device = hidden_states.device
                        dynamic_res = 5 if self.anyres else 1
                        retained_num = int(R_dict[self.scale][decoder_layer.self_attn.layer_idx + 1][self.retained_num] * dynamic_res)

                        # 1) attn importance (higher better)
                        attn = cross_attn_weights.detach()
                        if attn.dim() == 1:
                            attn = attn.unsqueeze(0)  # (B, V)
                        attn = (attn - attn.min(dim=1, keepdim=True).values) / (
                            attn.max(dim=1, keepdim=True).values -
                            attn.min(dim=1, keepdim=True).values + 1e-8)  # (B,V) in [0,1]

                        # 2) normalized features for cosine
                        vis_norm = torch.nn.functional.normalize(
                            total_visual_tokens, p=2, dim=-1)  # (B,V,D)

                        all_selected = []
                        for b in range(B):
                            a = attn[b]            # (V,)
                            feat = vis_norm[b]     # (V,D)

                            available = torch.ones(
                                V, dtype=torch.bool, device=device)
                            selected = []

                            # if retained_num > 0:
                            # step0: pick highest attention
                            i0 = torch.argmax(a).item()
                            selected.append(i0)  # 最少保留了一个image token
                            available[i0] = False

                            max_sim_to_S = feat @ feat[i0]    # (V,) cosine
                            # 自己的sim设为1（反正不可选了）
                            max_sim_to_S[i0] = 1.0

                            # 预计算全量相似度矩阵 (V, V)，用于覆盖性分数
                            full_sim = feat @ feat.t()  # (V, V)

                            for step in range(1, retained_num):
                                if not available.any():
                                    break

                                diversity = 1.0 - max_sim_to_S             # 越大越好

                                # 覆盖性分数: 候选token与其他未选中token的平均相似度（排除自身）
                                # sum_j(sim(i,j)) for j in available, 然后减去sim(i,i)=1, 再除以(n_avail-1)
                                if coverage_method == "SCOPE":
                                    # SCOPE: marginal coverage gain
                                    coverage = torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                                    coverage[~available] = 0.0
                                else:
                                    # MEAN: avg similarity to remaining unselected tokens
                                    sim_to_avail = full_sim[:, available].sum(dim=1)
                                    n_avail = available.sum().item()
                                    if n_avail > 1:
                                        coverage = (sim_to_avail - 1.0) / (n_avail - 1)
                                    else:
                                        coverage = torch.zeros(V, device=device)

                                scores = (a/a.mean()) + \
                                    alpha_2 * (diversity/diversity.mean()) + \
                                    lambda_2 * (coverage/coverage.mean())

                                scores[~available] = -float("inf")
                                idx = torch.argmax(scores).item()

                                selected.append(idx)
                                available[idx] = False

                                # 更新 m_i = max(m_i, sim(i, idx))
                                # (V,)
                                sim_vec = full_sim[idx]
                                max_sim_to_S = torch.maximum(
                                    max_sim_to_S, sim_vec)
                                max_sim_to_S[idx] = 1.0

                            selected_t = torch.tensor(
                                selected, device=device, dtype=torch.long)
                            selected_t = torch.sort(
                                selected_t).values     # 保持原视觉顺序
                            all_selected.append(selected_t)

                        # ————————————————————————————————————————————————————————
                        # 更新原始索引映射：将当前选中的相对索引映射回原始 token 空间
                        _current_indices = _current_indices[all_selected[0]]
                        self.pruning_history.append(_current_indices.cpu().clone())
                        # ————————————————————————————————————————————————————————

                        # batch=1 常见：直接取第0个
                        retained_visual_index = all_selected[0].unsqueeze(
                            0)  # (B,R)

                        select_visual_tokens = total_visual_tokens.gather(
                            1, retained_visual_index.unsqueeze(
                                -1).expand(-1, -1, D)
                        )  # (B,R,D)

                        # ---- rebuild: sys - retained_visual - text ----
                        hidden_states = torch.cat((
                            hidden_states[:, :self.system_prompt_length],
                            select_visual_tokens,
                            hidden_states[:, self.system_prompt_length +
                                          visual_token_length:]
                        ), dim=1)

                        position_ids = position_ids[:, :hidden_states.shape[1]]
                        if not output_attentions:
                            # 下游期望 (hidden, cache) 或 (hidden,)，需要去掉attn_weights
                            if use_cache:
                                layer_outputs = (hidden_states, layer_outputs[2])
                            else:
                                layer_outputs = (hidden_states,)
                        visual_token_length = select_visual_tokens.shape[1]

                        # ---- debug ----
                        # 选中token对应的attn均值
                        # avg_attn = a[selected_t].mean().item()
                        # # 选中token两两相似度均值（去掉对角线）
                        # if selected_t.numel() > 1:
                        #     sel_feat = feat[selected_t]
                        #     sel_sim = sel_feat @ sel_feat.t()          # (R, R)
                        #     diag_mask = ~torch.eye(sel_sim.size(
                        #         0), dtype=torch.bool, device=sel_sim.device)
                        #     avg_sim = sel_sim[diag_mask].mean().item()
                        # else:
                        #     avg_sim = 0.0
                        # # 选中token对未选中token的平均覆盖度
                        # unselected_mask = torch.ones(V, dtype=torch.bool, device=device)
                        # unselected_mask[selected_t] = False
                        # if unselected_mask.any() and selected_t.numel() > 0:
                        #     # (R, U) 选中token与未选中token的相似度
                        #     cov_sim = full_sim[selected_t][:, unselected_mask]
                        #     # 对每个未选中token取最大覆盖（被哪个选中token最好地代表）
                        #     max_cov_per_unsel = cov_sim.max(dim=0).values  # (U,)
                        #     avg_coverage = max_cov_per_unsel.mean().item()
                        # else:
                        #     avg_coverage = 1.0
                        # print(
                        #     f"[PrefixVLM_2 S2] layer_idx: {decoder_layer.self_attn.layer_idx+1}, "
                        #     f"[PrefixVLM_2 S2] visual_token_length: {visual_token_length}, "
                        #     f"[PrefixVLM_2 S2] retained_num: {retained_num}, "
                        #     f"[PrefixVLM_2 S2] alpha_2: {alpha_2}, "
                        #     f"[PrefixVLM_2 S2] lambda_2: {lambda_2}, "
                        #     f"[PrefixVLM_2 S2] avg_attn_on_selected_tokens: {avg_attn:.4f}, "
                        #     f"[PrefixVLM_2 S2] avg_sim_on_selected_tokens: {avg_sim:.4f}, "
                        #     f"[PrefixVLM_2 S2] avg_coverage_on_selected_tokens: {avg_coverage:.4f}"
                        # )

                    else:
                        # 非剪枝层：正常前向传播
                        attn_mask = torch.ones(
                            (batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(
                            attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
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
            self.layer_visual_tokens = visual_token_list  # expose for FLOPs tracking

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache(
            ) if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )