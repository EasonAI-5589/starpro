import torch
import einops as ein
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast

class IdeaLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, idea_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0
        
        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        self.retained_num = idea_config["T"]
        import os
        #使用retained_num*2作为visual_token_length，因为S1阶段要保留的token数量就是retained_num*2
        if config.image_aspect_ratio == "pad":
            self.visual_token_length = int(os.environ.get("NUM_CLUSTERS",64))
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 5*int(os.environ.get("NUM_CLUSTERS",64))
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
                # Idea: Dynamic Token Library — expand/collapse clusters based on cross-attention
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

                        # --- Get cross-attention: last text token attending to visual tokens ---
                        self_attn_weights = layer_outputs[1].mean(1)  # (B, seq, seq)
                        last_text_pos = hidden_states.shape[1] - 1
                        cross_attn_weights = self_attn_weights[:, last_text_pos,
                                                               self.system_prompt_length:self.system_prompt_length+visual_token_length]
                        # cross_attn_weights: (B, V)

                        import os
                        enable_debug = os.environ.get('ENABLE_DEBUG', '0') == '1'
                        attn_threshold_ratio = float(os.environ.get("ATTN_THRESHOLD_RATIO", "0.5"))

                        # --- Access token library ---
                        token_lib = self._idea_token_library
                        cluster_assignments = token_lib['cluster_assignments']  # (B, N_orig)
                        all_features = token_lib['all_features_raw']                # (B, N_orig, D_llm)
                        num_clusters = token_lib['num_clusters']
                        representative_indices = token_lib['representative_indices']  # (B, K)

                        # Initialize expanded_clusters tracking on first pruning_loc hit
                        if not hasattr(self, '_idea_expanded_clusters'):
                            self._idea_expanded_clusters = set()  # set of cluster ids currently expanded

                        B_cur = hidden_states.shape[0]
                        D = hidden_states.shape[-1]
                        device = hidden_states.device

                        # --- Current visual tokens in hidden_states ---
                        visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length + visual_token_length]

                        # --- Build per-cluster attention scores ---
                        # cross_attn_weights is over the current visual token sequence
                        # We need to map current visual positions back to cluster ids
                        # _idea_current_token_cluster_ids tracks which cluster each visual position belongs to
                        if not hasattr(self, '_idea_current_token_cluster_ids'):
                            # First time: visual tokens are just the representative tokens (one per cluster)
                            # Position k in visual sequence = cluster k
                            self._idea_current_token_cluster_ids = list(range(num_clusters))

                        current_cluster_ids = self._idea_current_token_cluster_ids
                        if cross_attn_weights.dim() == 1:
                            cross_attn_weights = cross_attn_weights.unsqueeze(0)

                        # Compute average attention per cluster
                        cluster_attn = torch.zeros(num_clusters, device=device)
                        cluster_count = torch.zeros(num_clusters, device=device)
                        for pos, cid in enumerate(current_cluster_ids):
                            if pos < cross_attn_weights.shape[1]:
                                cluster_attn[cid] += cross_attn_weights[0, pos]
                                cluster_count[cid] += 1
                        cluster_count = cluster_count.clamp(min=1)
                        cluster_avg_attn = cluster_attn / cluster_count  # (num_clusters,)

                        # --- Determine which clusters to expand and collapse ---
                        expanded = self._idea_expanded_clusters
                        attn_mean = cluster_avg_attn.mean()

                        # Clusters to EXPAND: all collapsed clusters whose attention > mean * threshold
                        expand_ids = set()
                        for k in range(num_clusters):
                            if k not in expanded and cluster_avg_attn[k] > attn_mean * attn_threshold_ratio:
                                expand_ids.add(k)

                        # Clusters to COLLAPSE: currently expanded but attention dropped below mean
                        collapse_ids = set()
                        for cid in list(expanded):
                            if cluster_avg_attn[cid] < attn_mean * attn_threshold_ratio:
                                collapse_ids.add(cid)

                        # Update expanded set
                        new_expanded = (expanded | expand_ids) - collapse_ids

                        if enable_debug:
                            print(f"\n[Idea Lib] layer={decoder_layer.self_attn.layer_idx+1}")
                            print(f"  cluster_avg_attn: min={cluster_avg_attn.min().item():.4f}, "
                                  f"max={cluster_avg_attn.max().item():.4f}, mean={attn_mean.item():.4f}")
                            print(f"  previously expanded: {sorted(expanded)}")
                            print(f"  expanding: {sorted(expand_ids)}, collapsing: {sorted(collapse_ids)}")
                            print(f"  new expanded: {sorted(new_expanded)}")

                        # --- Rebuild visual token sequence ---
                        # For each cluster (in order):
                        #   - if expanded: insert all cluster member tokens (with delta update)
                        #   - if collapsed: insert only the representative token
                        # The representative token's hidden_state may have changed through LLM layers,
                        # so we compute delta = current_rep_hidden - original_rep_feature
                        # and add it to all cluster members when expanding.

                        # Build a map: cluster_id -> current position in visual sequence
                        cluster_to_positions = {}
                        for pos, cid in enumerate(current_cluster_ids):
                            cluster_to_positions.setdefault(cid, []).append(pos)

                        new_visual_list = []
                        new_cluster_ids = []

                        for k in range(num_clusters):
                            positions = cluster_to_positions.get(k, [])
                            if len(positions) == 0:
                                continue

                            if k in new_expanded:
                                # EXPAND this cluster: use all members with delta update
                                # Get the current representative hidden state (first position of this cluster if collapsed,
                                # or we need to identify the rep among expanded members)
                                # Use the mean of current positions as the "current state" of this cluster
                                current_rep_hidden = visual_tokens[0, positions].mean(dim=0, keepdim=True)  # (1, D)

                                # Original representative feature from token library
                                rep_idx = representative_indices[0, k].item()
                                original_rep_feature = all_features[0, rep_idx:rep_idx+1]  # (1, D)

                                # Delta: how much the representative evolved through LLM layers
                                delta = current_rep_hidden - original_rep_feature  # (1, D)

                                # Get all cluster member features and add delta
                                member_mask = (cluster_assignments[0] == k)
                                member_features = all_features[0, member_mask]  # (M, D)
                                updated_members = member_features + delta  # (M, D) broadcast

                                new_visual_list.append(updated_members)
                                new_cluster_ids.extend([k] * updated_members.shape[0])
                            else:
                                # COLLAPSE: use representative token from token library
                                rep_idx = representative_indices[0, k].item()
                                # Get the current hidden state of the rep token if it's in the sequence,
                                # otherwise use the library feature with accumulated delta
                                if len(positions) == 1:
                                    # Already collapsed — keep current token (it has been updated by LLM layers)
                                    new_visual_list.append(visual_tokens[0, positions[0]:positions[0]+1])  # (1, D)
                                else:
                                    # Was expanded, now collapsing — find the rep token among members
                                    # Use the position that corresponds to the representative index
                                    member_mask = (cluster_assignments[0] == k)
                                    member_indices = member_mask.nonzero(as_tuple=True)[0]
                                    rep_local_pos = (member_indices == rep_idx).nonzero(as_tuple=True)[0]
                                    if rep_local_pos.numel() > 0:
                                        # rep token is at this offset within the expanded positions
                                        rep_pos_in_seq = positions[rep_local_pos[0].item()]
                                        new_visual_list.append(visual_tokens[0, rep_pos_in_seq:rep_pos_in_seq+1])
                                    else:
                                        # Fallback: use first position
                                        new_visual_list.append(visual_tokens[0, positions[0]:positions[0]+1])
                                new_cluster_ids.append(k)

                        # Stack new visual tokens
                        new_visual_tokens = torch.cat(new_visual_list, dim=0).unsqueeze(0)  # (1, V_new, D)

                        # Update state
                        self._idea_expanded_clusters = new_expanded
                        self._idea_current_token_cluster_ids = new_cluster_ids

                        # Rebuild hidden_states: sys - visual - text
                        hidden_states = torch.cat((
                            hidden_states[:, :self.system_prompt_length],
                            new_visual_tokens,
                            hidden_states[:, self.system_prompt_length + visual_token_length:]
                        ), dim=1)

                        # Regenerate position_ids and attention_mask to match new sequence length
                        new_seq_len = hidden_states.shape[1]
                        position_ids = torch.arange(new_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
                        attn_mask_new = torch.ones((batch_size, new_seq_len), device=hidden_states.device)
                        attention_mask = _prepare_4d_causal_attention_mask(attn_mask_new, (batch_size, new_seq_len), hidden_states, 0)
                        layer_outputs = (hidden_states, layer_outputs[2])
                        visual_token_length = new_visual_tokens.shape[1]

                        if enable_debug:
                            print(f"  new visual_token_length: {visual_token_length}")
                            print(f"  expanded clusters have tokens: "
                                  f"{sum(1 for c in new_cluster_ids if new_cluster_ids.count(c) > 1 and c in new_expanded)}")

                        print(
                            f"layer_idx: {decoder_layer.self_attn.layer_idx+1}, "
                            f"visual_token_length: {visual_token_length}, "
                            f"expanded: {len(new_expanded)}, "
                            f"expand_new: {len(expand_ids)}, "
                            f"collapsed: {len(collapse_ids)}"
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
