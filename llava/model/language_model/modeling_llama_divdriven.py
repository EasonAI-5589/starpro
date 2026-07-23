import torch
import einops as ein
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast

R_dict = { 
    "7b": {
        2: {
            192: 240,
            128: 288,
            64: 128,
            32: 72,
        },
        8: {
            192: 216,
            128: 192,
            64: 72,
            32 :36,
        },
        16: {
            192: 192,
            128: 32,
            64: 16,
            32: 0,
        },
        24: {
            192: 192,
            128: 0,
            64: 0,
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

# D2P: Diversity Driven Pruning Algorithm
class D2PLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, d2p_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0

        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        self.retained_num = d2p_config.get("T", None)
        import os
        if config.image_aspect_ratio == "pad":
            self.visual_token_length = int(os.environ.get("STAGE1_KEEP", "288"))
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 5*int(os.environ.get("STAGE1_KEEP", "288"))
            self.anyres = True

        if "stages" in d2p_config:
            # Direct stage config: [(layer, keep_count), ...] — bypasses R_dict
            self.stage_keep = {s[0]: s[1] for s in d2p_config["stages"]}
            self.pruning_loc = sorted(self.stage_keep.keys())
        else:
            # D2P config
            if config.num_hidden_layers == 32:
                if self.retained_num == 32:
                    self.pruning_loc = [2, 8, 16]
                elif self.retained_num == 64:
                    self.pruning_loc = [2, 8, 16, 24]
                elif self.retained_num == 128:
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

                        # ------------------------------------------------------------------
                        # score(i) = (1 - mean_sim_to_text(i)) + (1 - max_sim_to_selected_visual(i))
                        # 第一项“和文本”的相似度
                        # 第二项“和已选视觉集合”的相似度
                        # ------------------------------------------------------------------
                        # [B, V, D]
                        total_visual_tokens = hidden_states[
                            :, self.system_prompt_length:self.system_prompt_length + visual_token_length
                        ]
                        B, V, D = total_visual_tokens.shape
                        device = hidden_states.device

                        import os
                        div_2 = float(os.environ.get("DIV_2", "0.5"))
                        dynamic_res = 5 if self.anyres else 1
                        if hasattr(self, 'stage_keep'):
                            retained_num = self.stage_keep[decoder_layer.self_attn.layer_idx + 1] * (5 if self.anyres else 1)
                        else:
                            retained_num = int(R_dict[self.scale][decoder_layer.self_attn.layer_idx + 1][self.retained_num] * dynamic_res)

                        # 文本 token: [B, T, D]
                        total_text_tokens = hidden_states[:, self.system_prompt_length + visual_token_length:]

                        # 归一化，后面直接点积就是 cosine sim
                        vis_norm = torch.nn.functional.normalize(total_visual_tokens, p=2, dim=-1)   # (B,V,D)

                        if total_text_tokens.shape[1] > 0:
                            text_norm = torch.nn.functional.normalize(total_text_tokens, p=2, dim=-1)  # (B,T,D)
                        else:
                            text_norm = None

                        all_selected = []

                        for b in range(B):
                            feat = vis_norm[b]   # (V,D)
                            available = torch.ones(V, dtype=torch.bool, device=device)
                            selected = []

                            # 第一项: 1 - max_sim_to_text
                            if text_norm is not None and text_norm.shape[1] > 0:
                                txt = text_norm[b]                    # (T,D)
                                sim_to_text = feat @ txt.t()         # (V,T)
                                max_sim_to_text = sim_to_text.max(dim=1).values   # (V,)
                            else:
                                max_sim_to_text = torch.zeros(V, device=device, dtype=feat.dtype)

                            text_score = 1.0 - max_sim_to_text      # (V,)
                            
                            #step0:这时还没有 selected 集合，先只按第一项选
                            init_scores = text_score.clone()
                            init_scores[~available] = -float("inf")

                            i0 = torch.argmax(init_scores).item()
                            selected.append(i0)
                            available[i0] = False

                            # max_sim_to_S[i] = 当前候选 i 与已选集合 S 中最相似那个 token 的相似度
                            max_sim_to_S = feat @ feat[i0]    # (V,)
                            max_sim_to_S[i0] = 1.0
                            for step in range(1, retained_num):
                                if not available.any():
                                    break

                                visual_div_score = 1.0 - max_sim_to_S    # (V,)
                                scores = div_2*text_score/text_score.mean() + visual_div_score/visual_div_score.mean()   # (V,)

                                scores[~available] = -float("inf")
                                idx = torch.argmax(scores).item()

                                selected.append(idx)
                                available[idx] = False

                                sim_vec = feat @ feat[idx]               # (V,)
                                max_sim_to_S = torch.maximum(max_sim_to_S, sim_vec)
                                max_sim_to_S[idx] = 1.0

                            selected_t = torch.tensor(selected, device=device, dtype=torch.long)
                            selected_t = torch.sort(selected_t).values   # 保持原视觉顺序
                            all_selected.append(selected_t)

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
                        layer_outputs = (hidden_states, layer_outputs[2])
                        visual_token_length = select_visual_tokens.shape[1]

                        # ---- debug: avg selected attn & avg pairwise sim ----
                        
                        # 选中token两两相似度均值（去掉对角线）
                        if selected_t.numel() > 1:
                            # (R, D)  这里feat是vis_norm[b]
                            sel_feat = feat[selected_t]
                            sel_sim = sel_feat @ sel_feat.t()          # (R, R)
                            mask = ~torch.eye(sel_sim.size(
                                0), dtype=torch.bool, device=sel_sim.device)
                            avg_sim = sel_sim[mask].mean().item()
                        else:
                            avg_sim = 0.0
                        avg_img_text_sim = max_sim_to_text[selected_t].mean().item()
                        print(
                            f"layer_idx: {decoder_layer.self_attn.layer_idx+1}, "
                            f"visual_token_length: {visual_token_length}, "
                            f"retained_num: {retained_num}, "
                            f"div_2: {div_2}, "
                            f"avg_image_text_similarity: {avg_img_text_sim:.4f}, "
                            f"avg_image_similarity: {avg_sim:.4f}"
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