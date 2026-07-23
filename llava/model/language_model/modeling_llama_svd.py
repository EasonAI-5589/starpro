import torch
import einops as ein
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast

R_dict = { #这个是STAR_V2的配置
    "7b": {
        2: {
            192: 240,
            128: 288,
            64: 128,
        },
        8: {
            192: 216,
            128: 192,
            64: 72,
        },
        16: {
            192: 192,
            128: 32,
            64: 16,
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


class SVDLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, svdvlm_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0
        
        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        self.retained_num = svdvlm_config["T"]
        #使用retained_num*2作为visual_token_length，因为S1阶段要保留的token数量就是retained_num*2
        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576 // 2
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880 // 2
            self.anyres = True

        # SVDVLM config
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
                        # --- Text-guided SVD pruning ---
                        import os
                        tau = float(os.environ.get("SVD_TAU", "0.95"))        # energy threshold for SVD rank
                        lambda_val = float(os.environ.get("LAMBDA", "0.5"))   # diversity weight in greedy selection

                        # Save input hidden states for computing per-head V
                        input_hidden_states = hidden_states

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

                        # Per-head attention weights: [B, H, N, N]
                        attn_weights_full = layer_outputs[1]
                        B_size, H, N_total, _ = attn_weights_full.shape
                        device = hidden_states.device
                        D = hidden_states.shape[-1]

                        Ns = self.system_prompt_length
                        Nv = visual_token_length
                        Nt = N_total - Ns - Nv
                        layer_idx = decoder_layer.self_attn.layer_idx + 1

                        dynamic_res = 5 if self.anyres else 1
                        retained_num = R_dict[self.scale][layer_idx][self.retained_num] * dynamic_res

                        print(f"\n{'='*80}")
                        print(f"[SVD Pruning] Layer {layer_idx}")
                        print(f"  seq layout: Ns={Ns}, Nv={Nv}, Nt={Nt}, total={N_total}")
                        print(f"  target retained_num={retained_num}, H={H}, head_dim={D // H}")
                        print(f"  tau={tau}, lambda={lambda_val}")

                        text_start = Ns + Nv

                        # --- Step 3: Text-guided weights per head ---
                        # A_h(q, i) for q in text, i in visual: [B, H, Nt, Nv]
                        text_to_vis_attn = attn_weights_full[:, :, text_start:, Ns:Ns+Nv]
                        # w_h: uniform average over text tokens -> [B, H, Nv]
                        w_h = text_to_vis_attn.mean(dim=2)

                        print(f"  [Step 3] text_to_vis_attn shape={text_to_vis_attn.shape}")
                        print(f"  [Step 3] w_h: min={w_h.min().item():.6f}, max={w_h.max().item():.6f}, mean={w_h.mean().item():.6f}")

                        # --- Step 8: Head importance beta ---
                        # beta_h = mean text-to-vision attention per head -> [B, H]
                        beta_h = text_to_vis_attn.mean(dim=(2, 3))
                        beta_h_normalized = beta_h / (beta_h.sum(dim=1, keepdim=True) + 1e-8)  # normalized [B, H]

                        print(f"  [Step 8] beta_h (raw): min={beta_h.min().item():.6f}, max={beta_h.max().item():.6f}")
                        print(f"  [Step 8] beta_h (norm): min={beta_h_normalized.min().item():.6f}, max={beta_h_normalized.max().item():.6f}")
                        # Show top-5 and bottom-5 head weights
                        beta_sorted, beta_indices = beta_h_normalized[0].sort(descending=True)
                        top5_str = ", ".join([f"h{beta_indices[i].item()}={beta_sorted[i].item():.4f}" for i in range(min(5, H))])
                        bot5_str = ", ".join([f"h{beta_indices[-(i+1)].item()}={beta_sorted[-(i+1)].item():.4f}" for i in range(min(5, H))])
                        print(f"  [Step 8] top-5 heads: {top5_str}")
                        print(f"  [Step 8] bot-5 heads: {bot5_str}")

                        # --- Extract per-head V from KV cache (already computed by the layer) ---
                        self_attn = decoder_layer.self_attn
                        head_dim = self_attn.head_dim
                        num_heads = self_attn.num_heads
                        num_kv_heads = self_attn.num_key_value_heads
                        cache_layer_idx = self_attn.layer_idx

                        v_states = past_key_values.value_cache[cache_layer_idx]
                        # v_states: [B, num_kv_heads, seq_len, head_dim]

                        print(f"  [V cache] v_states shape={v_states.shape}, num_kv_heads={num_kv_heads}, num_heads={num_heads}")

                        # Handle GQA: repeat KV heads to match attention heads
                        if num_kv_heads < num_heads:
                            n_rep = num_heads // num_kv_heads
                            print(f"  [V cache] GQA detected, repeating KV heads x{n_rep}")
                            v_states = v_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
                            v_states = v_states.reshape(B_size, num_heads, N_total, head_dim)

                        # Extract visual V: [B, H, Nv, d_h]
                        V_visual = v_states[:, :, Ns:Ns+Nv, :]
                        print(f"  [V cache] V_visual shape={V_visual.shape}, norm={V_visual.float().norm().item():.4f}")

                        all_selected = []
                        for b in range(B_size):
                            scores_per_token = torch.zeros(Nv, device=device, dtype=hidden_states.dtype)
                            head_ranks = []       # track SVD rank per head
                            head_skipped = 0      # count heads with zero energy

                            for h in range(H):
                                w = w_h[b, h]            # [Nv]
                                V_v = V_visual[b, h]     # [Nv, d_h]

                                # --- Step 4: Text-guided weighted Value matrix ---
                                V_weighted = torch.sqrt(w.clamp(min=0)).unsqueeze(1) * V_v  # [Nv, d_h]

                                # --- Step 5: SVD ---
                                U, S, Vh = torch.linalg.svd(V_weighted.float(), full_matrices=False)
                                # U: [Nv, r_max], S: [r_max]

                                # --- Step 6: Energy criterion for rank ---
                                energy_total = (S ** 2).sum()
                                if energy_total < 1e-12:
                                    head_skipped += 1
                                    continue
                                cumulative_energy = torch.cumsum(S ** 2, dim=0) / energy_total
                                r_indices = (cumulative_energy >= tau).nonzero(as_tuple=True)[0]
                                r = (r_indices[0].item() + 1) if len(r_indices) > 0 else len(S)
                                head_ranks.append(r)

                                # --- Step 7: Leverage scores ---
                                U_r = U[:, :r]                          # [Nv, r]
                                leverage = (U_r ** 2).sum(dim=1)        # [Nv]

                                # Per-head score: w_i * leverage_i
                                s_h = w * leverage.to(w.dtype)          # [Nv]

                                # Accumulate with head weight
                                scores_per_token += beta_h_normalized[b, h] * s_h

                            # --- SVD debug summary ---
                            if head_ranks:
                                ranks_t = torch.tensor(head_ranks, dtype=torch.float)
                                print(f"  [SVD] b={b}: active_heads={len(head_ranks)}/{H}, skipped={head_skipped}")
                                print(f"  [SVD] rank stats: min={ranks_t.min().item():.0f}, max={ranks_t.max().item():.0f}, "
                                      f"mean={ranks_t.mean().item():.1f}, median={ranks_t.median().item():.0f}")
                            else:
                                print(f"  [SVD] b={b}: WARNING all heads skipped (zero energy)!")

                            print(f"  [Score] scores_per_token: min={scores_per_token.min().item():.6f}, "
                                  f"max={scores_per_token.max().item():.6f}, mean={scores_per_token.mean().item():.6f}, "
                                  f"std={scores_per_token.std().item():.6f}")
                            # Show top-10 and bottom-10 token scores
                            topk_vals, topk_ids = scores_per_token.topk(min(10, Nv))
                            botk_vals, botk_ids = scores_per_token.topk(min(10, Nv), largest=False)
                            print(f"  [Score] top-10 tokens: {list(zip(topk_ids.tolist(), [f'{v:.4f}' for v in topk_vals.tolist()]))}")
                            print(f"  [Score] bot-10 tokens: {list(zip(botk_ids.tolist(), [f'{v:.4f}' for v in botk_vals.tolist()]))}")

                            # --- Steps 9-11: Greedy diverse selection ---
                            # Diversity features: use input hidden states (方案A)
                            vis_hidden = input_hidden_states[b, Ns:Ns+Nv, :]   # [Nv, D]
                            vis_norm = torch.nn.functional.normalize(vis_hidden, p=2, dim=-1)

                            # Normalize scores to [0, 1]
                            s = scores_per_token
                            s_range = s.max() - s.min()
                            if s_range > 1e-8:
                                s = (s - s.min()) / s_range
                               
                            # # 方案B：使用直接使用energy的topk选择retained_num个token
                            # selected = torch.topk(s,retained_num,largest=True).indices

                            #方案A：使用加权diversity的贪心算法选择retained_num个token
                            available = torch.ones(Nv, dtype=torch.bool, device=device)
                            selected = []

                            # First pick: highest SVD-based score
                            i0 = torch.argmax(s).item()
                            selected.append(i0)
                            available[i0] = False
                            max_sim_to_S = vis_norm @ vis_norm[i0]
                            max_sim_to_S[i0] = 1.0
                            print(f"  [Greedy] step 0: picked token {i0}, score={s[i0].item():.4f}")

                            for step in range(1, retained_num):
                                if not available.any():
                                    break
                                diversity = 1.0 - max_sim_to_S
                                combined = ((1 - lambda_val) * (s / (s.mean() + 1e-8))
                                            + lambda_val * (diversity / (diversity.mean() + 1e-8)))
                                combined[~available] = -float("inf")
                                idx = torch.argmax(combined).item()
                                selected.append(idx)
                                available[idx] = False
                                sim_vec = vis_norm @ vis_norm[idx]
                                max_sim_to_S = torch.maximum(max_sim_to_S, sim_vec)
                                max_sim_to_S[idx] = 1.0

                                # Print first 5 and every 50th step
                                if step < 5 or step % 50 == 0:
                                    print(f"  [Greedy] step {step}: picked token {idx}, "
                                          f"score={s[idx].item():.4f}, diversity={diversity[idx].item():.4f}, "
                                          f"combined={combined[idx].item():.4f}")

                            print(f"  [Greedy] done: {len(selected)}/{Nv} tokens retained")

                            selected_t = torch.tensor(selected, device=device, dtype=torch.long)
                            selected_t = torch.sort(selected_t).values
                            all_selected.append(selected_t)

                            # Print spatial distribution of selected tokens
                            quartile_size = Nv // 4
                            if quartile_size > 0:
                                q1 = (selected_t < quartile_size).sum().item()
                                q2 = ((selected_t >= quartile_size) & (selected_t < 2*quartile_size)).sum().item()
                                q3 = ((selected_t >= 2*quartile_size) & (selected_t < 3*quartile_size)).sum().item()
                                q4 = (selected_t >= 3*quartile_size).sum().item()
                                print(f"  [Spatial] token distribution by quartile: Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}")

                        # batch=1 常见：直接取第0个
                        retained_visual_index = all_selected[0].unsqueeze(0)  # (B, R)

                        total_visual_tokens = hidden_states[:, Ns:Ns+Nv, :]
                        select_visual_tokens = total_visual_tokens.gather(
                            1, retained_visual_index.unsqueeze(-1).expand(-1, -1, D)
                        )  # (B, R, D)

                        # ---- rebuild: sys - retained_visual - text ----
                        hidden_states = torch.cat((
                            hidden_states[:, :Ns],
                            select_visual_tokens,
                            hidden_states[:, Ns + Nv:]
                        ), dim=1)

                        position_ids = position_ids[:, :hidden_states.shape[1]]
                        layer_outputs = (hidden_states, layer_outputs[2])
                        visual_token_length = select_visual_tokens.shape[1]

                        print(f"  [Result] hidden_states shape after pruning: {hidden_states.shape}")
                        print(f"  [Result] visual_token_length: {Nv} -> {visual_token_length}")
                        print(f"{'='*80}\n")
                    
                    
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
