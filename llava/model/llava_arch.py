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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config, **kwargs):
        super(LlavaMetaModel, self).__init__(config, **kwargs)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, texts=None):
        if 'prumerge' in self.pruning_method or self.pruning_method == 'visionzip' or self.pruning_method == 'fastervlm':
            image_features, image_attentions, image_keys, image_cls = self.get_model().get_vision_tower()(images, output_attentions=True)
        elif self.pruning_method == 'trim' or self.pruning_method == 'cdp3' or 'thcp' in self.pruning_method or self.pruning_method == 'star_v3':
            # 🔥 添加这行调试
            print(f"Pruning method: {self.pruning_method}, texts is None: {texts is None}, texts value: {texts}")
            image_features, image_embeds, text_embeds = self.get_model().get_vision_tower()(images, texts=texts)
        else:
            image_features = self.get_model().get_vision_tower()(images)
        
        B, N, C = image_features.shape
        device = image_features.device
        index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
        merged_features = None
        
        if 'prumerge' in self.pruning_method:
            cls_attn = image_attentions.mean(dim=1)

            if self.pruning_method == 'prumerge':
                selected_idx = cls_attn.topk(k=self.visual_token_num-1, dim=1).indices
            elif self.pruning_method == 'prumerge+':
                step_length = int(N / self.visual_token_num * 2)
                arithmetic_idx = torch.arange(int(step_length/2), 575, int(step_length), device=device)
                cls_attn[:, arithmetic_idx] = 0
                selected_idx = cls_attn.topk(k=self.visual_token_num//2-1, dim=1).indices
                selected_idx = torch.cat([selected_idx, arithmetic_idx.unsqueeze(0).expand(B, -1)], dim=1)
            
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, selected_idx, True)
            pruned_idx = index_masks.float().topk(k=N-self.visual_token_num+1, dim=1, largest=False).indices

            selected_feat = torch.gather(image_features, dim=1, index=selected_idx.unsqueeze(-1).expand(-1, -1, C))
            selected_key = torch.gather(image_keys, dim=1, index=selected_idx.unsqueeze(-1).expand(-1, -1, C))
            selected_attn = torch.gather(cls_attn, dim=1, index=selected_idx)
            pruned_feat = torch.gather(image_features, dim=1, index=pruned_idx.unsqueeze(-1).expand(-1, -1, C))
            pruned_key = torch.gather(image_keys, dim=1, index=pruned_idx.unsqueeze(-1).expand(-1, -1, C))
            pruned_attn = torch.gather(cls_attn, dim=1, index=pruned_idx)

            # selected_norm = F.normalize(selected_feat, p=2, dim=-1)
            # pruned_norm = F.normalize(pruned_feat, p=2, dim=-1)
            selected_norm = F.normalize(selected_key, p=2, dim=-1)
            pruned_norm = F.normalize(pruned_key, p=2, dim=-1)

            updated_feat = torch.zeros_like(selected_feat)
            for b in range(B):
                for i in range(updated_feat.shape[1]):
                    center_norm = selected_norm[b, i, :].unsqueeze(0)
                    others_norm = torch.cat([
                        selected_norm[b, :i, :],
                        selected_norm[b, i+1:, :],
                        pruned_norm[b, :, :],
                    ], dim=0)

                    # calculate cosine similarity and get cluster centers
                    cos_sim_matrix = center_norm @ others_norm.t()
                    cluster_idx = torch.topk(cos_sim_matrix, k=32, dim=1).indices

                    others_feat = torch.cat([
                        selected_feat[b, :i, :],
                        selected_feat[b, i+1:, :],
                        pruned_feat[b, :, :],
                    ], dim=0)
                    others_attn = torch.cat([
                        selected_attn[b, :i],
                        selected_attn[b, i+1:],
                        pruned_attn[b, :],
                    ], dim=0)

                    cluster_tokens = others_feat[cluster_idx.squeeze(), :]
                    cluster_weights = others_attn[cluster_idx.squeeze()].unsqueeze(-1) # (1, 32, 1)

                    # update cluster centers
                    cluster_avg = torch.sum(cluster_tokens * cluster_weights, dim=0)
                    cluster_center = cluster_avg + selected_feat[b, i, :] * (1 - torch.sum(cluster_weights))
                    updated_feat[b, i, :] = cluster_center
            image_features[index_masks] = updated_feat.flatten(0, 1)
            
            extra_weight = pruned_attn / torch.sum(pruned_attn, dim=1, keepdim=True)
            merged_features = torch.sum(pruned_feat * extra_weight.unsqueeze(-1), dim=1, keepdim=True)
        
        elif self.pruning_method == 'visionzip':
            dominant_token_num = int(self.visual_token_num * 27 / 32) - 1
            contextual_token_num = self.visual_token_num - dominant_token_num - 1

            # dominant visual tokens
            cls_attn = image_attentions.mean(dim=1)
            topk_idx = cls_attn.topk(dominant_token_num, dim=1).indices

            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, topk_idx, True)
            
            # filter
            features_filtered = image_features.masked_select(~index_masks.unsqueeze(-1)).view(B, N - dominant_token_num, C)
            metric_filtered = image_keys[~index_masks].view(B, N - dominant_token_num, C)
            # metric_normalized = features_filtered / features_filtered.norm(dim=-1, keepdim=True)
            metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

            # contextual visual tokens
            target_step = max(1, metric_normalized.shape[1] // contextual_token_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], target_step, device=device)[:contextual_token_num]
            target_tokens = metric_normalized[:, target_indices, :]

            tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=device), target_indices), :]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = features_filtered.new_zeros(B, tokens_to_merge.shape[1], contextual_token_num)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = features_filtered[:, ~torch.isin(torch.arange(features_filtered.shape[1], device=device), target_indices), :]
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            target_hidden = features_filtered[:, target_indices, :]
            contextual_tokens = target_hidden + aggregated_hidden

            # merge
            merged_features = torch.cat([image_cls, contextual_tokens], dim=1)
        
        elif self.pruning_method == 'fastervlm':
            cls_attn = image_attentions.mean(dim=1)

            topk_idx = cls_attn.topk(self.visual_token_num, dim=-1).indices
            topk_idx = torch.sort(topk_idx).values

            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, topk_idx, True)
        
        elif self.pruning_method == 'trim':
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarity = -torch.matmul(image_embeds, text_embeds.t())
            similarity = similarity.mean(dim=-1)

            # select topk tokens by similarity scores
            topk_idx = similarity.topk(self.visual_token_num - 1, dim=-1).indices
            topk_idx = torch.sort(topk_idx).values

            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, topk_idx, True)

            # aggregate the remaining tokens
            remaining_tokens = image_features[~index_masks].view(B, -1, C)
            merged_features = remaining_tokens.mean(dim=1, keepdim=True)
        
        elif self.pruning_method == 'dart':
            image_normalized = (image_features / image_features.norm(dim=-1, keepdim=True))
            visual_token_num = (self.visual_token_num - 8) // 8
            step = N // (8 + 1)

            retained_idx = []
            for b in range(B):
                retained_set = set(range(step, N, step))
                valid_set = set(range(N)) - retained_set
                for pivot_idx in list(retained_set):
                    valid_list = list(valid_set)
                    cos_sim = torch.matmul(image_normalized[b, pivot_idx], image_normalized[b, valid_list].t())
                    topk_idx = torch.topk(cos_sim, k=visual_token_num, largest=False).indices

                    topk_real_idx = [valid_list[idx] for idx in topk_idx]
                    retained_set.update(topk_real_idx)
                    valid_set.difference_update(topk_real_idx)
                retained_idx.append(list(retained_set))
            
            retained_idx = torch.tensor(retained_idx, dtype=torch.long, device=device)
            retained_idx = torch.sort(retained_idx).values

            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, retained_idx, True)
        
        image_features = self.get_model().mm_projector(image_features)

        if merged_features is not None:
            merged_features = self.get_model().mm_projector(merged_features)

        # ========== 在这里添加设备同步 ==========
        # 确保 index_masks 与 image_features 在同一设备上
        index_masks = index_masks.to(image_features.device)
        # ========================================
        
        if self.pruning_method == 'divprune':
            image_normalized = (image_features / image_features.norm(dim=-1, keepdim=True))
            cosine_matrix = 1.0 - torch.matmul(image_normalized, image_normalized.transpose(1, 2))
            
            select_idx = torch.empty((B, self.visual_token_num), dtype=torch.long, device=device)
            for i in range(self.visual_token_num):
                m2 = cosine_matrix if i==0 \
                    else cosine_matrix[torch.arange(B).unsqueeze(1).expand(-1, i), select_idx[:, :i]]
                scores = torch.topk(m2, 2, dim=1, largest=False).values[:, 1] \
                    if i==0 else torch.min(m2, dim=1).values
                select_idx[:, i] = torch.argmax(scores, dim=-1)
            
            select_idx = torch.sort(select_idx).values
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, select_idx, True)
        
        elif self.pruning_method == 'dp3':
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
            image_normalized = image_normalized.float() # (B, N, D)
            similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)

            kernel = similarity

            cis = torch.zeros((self.visual_token_num, B, N), device=device) # (T, B, N)
            di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
            select_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device) # (T, B)
            for i in range(self.visual_token_num):
                j = torch.argmax(di2s, dim=-1)
                select_idx[i] = j

                eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                    / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
                cis[i, :, :] = eis
                di2s -= torch.square(eis)
                di2s[torch.arange(B), j] = -float('inf')
            
            select_idx = torch.sort(select_idx.t()).values # (B, T)
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, select_idx, True)
        
        elif 'cdp3' in self.pruning_method:
            # ========== Debug配置 ==========
            enable_debug = True
            debug_info = {
                'text_stats': [],
                'relevance_stats': [],
                'kernel_stats': [],
                'selection_stats': []
            }
            
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
            image_normalized = image_normalized.float() # (B, N, D)
            similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
            
            # Debug: 文本embeddings统计
            if enable_debug:
                M = text_embeds.shape[0]
                print(f"\n{'='*80}")
                print(f"CDP3 Debug Information")
                print(f"{'='*80}")
                print(f"\n[Text Embeddings Shape]")
                print(f"  text_embeds.shape: {text_embeds.shape}")
                print(f"  M (num text tokens): {M}")
                print(f"  image_embeds.shape: {image_embeds.shape}")
                
                # 如果M>1，显示文本tokens之间的相似度
                if M > 1:
                    text_sim = torch.matmul(text_embeds, text_embeds.t())
                    text_sim_triu = text_sim[torch.triu(torch.ones_like(text_sim), diagonal=1).bool()]
                    print(f"\n[Text Token Similarity] (M={M})")
                    print(f"  Avg pairwise similarity: {text_sim_triu.mean().item():.4f}")
                    print(f"  Max pairwise similarity: {text_sim_triu.max().item():.4f}")
                    print(f"  Min pairwise similarity: {text_sim_triu.min().item():.4f}")
                else:
                    print(f"\n[Warning] Only 1 text token! CDP3 may not work optimally.")
            
            relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
            
            # Debug: Relevance矩阵统计（在mean之前）
            if enable_debug:
                print(f"\n[Relevance Matrix] (before mean)")
                print(f"  Shape: {relevance.shape}")  # (B, N, M)
                for b in range(min(B, 1)):  # 只显示第一个batch
                    print(f"  Batch {b}:")
                    print(f"    Mean: {relevance[b].mean().item():.4f}")
                    print(f"    Std: {relevance[b].std().item():.4f}")
                    print(f"    Range: [{relevance[b].min().item():.4f}, {relevance[b].max().item():.4f}]")
                    
                    # 显示每个文本token的平均相关性
                    if M > 1 and M <= 10:  # 只在M不太大时显示
                        per_text_relevance = relevance[b].mean(dim=0)  # (M,)
                        print(f"    Per-text avg relevance: {per_text_relevance.cpu().tolist()}")
            
            relevance = (-relevance).mean(dim=-1) # (B, N)
            relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)
            
            # Debug: Relevance统计（在mean和normalize之后）
            if enable_debug:
                print(f"\n[Relevance Scores] (after mean & normalize)")
                print(f"  Shape: {relevance.shape}")  # (B, N)
                for b in range(min(B, 1)):
                    print(f"  Batch {b}:")
                    print(f"    Mean: {relevance[b].mean().item():.4f}")
                    print(f"    Std: {relevance[b].std().item():.4f}")
                    print(f"    Range: [{relevance[b].min().item():.4f}, {relevance[b].max().item():.4f}]")
                    print(f"    Top-5 values: {relevance[b].topk(5).values.cpu().tolist()}")
                    print(f"    Top-5 indices: {relevance[b].topk(5).indices.cpu().tolist()}")

            kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)
            
            # Debug: Kernel矩阵统计
            if enable_debug:
                print(f"\n[Kernel Matrix]")
                print(f"  Shape: {kernel.shape}")  # (B, N, N)
                for b in range(min(B, 1)):
                    kernel_diag = torch.diagonal(kernel[b])
                    kernel_offdiag = kernel[b][~torch.eye(N, dtype=torch.bool, device=device)]
                    print(f"  Batch {b}:")
                    print(f"    Diagonal mean: {kernel_diag.mean().item():.4f}")
                    print(f"    Diagonal range: [{kernel_diag.min().item():.4f}, {kernel_diag.max().item():.4f}]")
                    print(f"    Off-diagonal mean: {kernel_offdiag.mean().item():.4f}")
                    print(f"    Off-diagonal range: [{kernel_offdiag.min().item():.4f}, {kernel_offdiag.max().item():.4f}]")

            cis = torch.zeros((self.visual_token_num, B, N), device=device) # (T, B, N)
            di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
            select_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device) # (T, B)
            
            # Debug: 记录选择过程
            if enable_debug:
                selection_process = []
            
            for i in range(self.visual_token_num):
                j = torch.argmax(di2s, dim=-1)
                select_idx[i] = j

                eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                    / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
                cis[i, :, :] = eis
                di2s -= torch.square(eis)
                di2s[torch.arange(B), j] = -float('inf')
                
                # Debug: 记录关键步骤的统计（每10步记录一次）
                if enable_debug and i % 10 == 0:
                    for b in range(min(B, 1)):
                        valid_di2s = di2s[b][di2s[b] != -float('inf')]
                        selection_process.append({
                            'step': i,
                            'batch': b,
                            'selected_idx': j[b].item(),
                            'selected_relevance': relevance[b, j[b]].item(),
                            'di2s_mean': valid_di2s.mean().item() if len(valid_di2s) > 0 else 0,
                            'di2s_max': valid_di2s.max().item() if len(valid_di2s) > 0 else 0,
                            'di2s_std': valid_di2s.std().item() if len(valid_di2s) > 0 else 0,
                        })
            
            select_idx = torch.sort(select_idx.t()).values # (B, T)
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, select_idx, True)
            
            # Debug: 最终选择统计
            if enable_debug:
                print(f"\n[DPP Selection Process] (Batch 0, every 10 steps)")
                print(f"  {'Step':<6} {'SelIdx':<8} {'SelRel':<10} {'Di2sMean':<12} {'Di2sMax':<12} {'Di2sStd':<12}")
                print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
                for info in selection_process:
                    if info['batch'] == 0:
                        print(f"  {info['step']:<6} "
                            f"{info['selected_idx']:<8} "
                            f"{info['selected_relevance']:<10.4f} "
                            f"{info['di2s_mean']:<12.4f} "
                            f"{info['di2s_max']:<12.4f} "
                            f"{info['di2s_std']:<12.4f}")
                
                print(f"\n[Final Selection Statistics]")
                for b in range(min(B, 1)):
                    selected_indices = select_idx[b].cpu()
                    selected_relevances = relevance[b, selected_indices]
                    
                    # 计算选中tokens的多样性
                    selected_sim = similarity[b][selected_indices][:, selected_indices]
                    mask = ~torch.eye(self.visual_token_num, dtype=torch.bool, device=device)
                    avg_sim = selected_sim[mask].mean().item()
                    
                    print(f"  Batch {b}:")
                    print(f"    Selected tokens: {self.visual_token_num}")
                    print(f"    Avg relevance of selected: {selected_relevances.mean().item():.4f}")
                    print(f"    Relevance range: [{selected_relevances.min().item():.4f}, {selected_relevances.max().item():.4f}]")
                    print(f"    Avg pairwise similarity: {avg_sim:.4f}")
                    print(f"    Diversity score: {1-avg_sim:.4f}")
                    
                    # 对比：随机选择的baseline
                    random_indices = torch.randperm(N, device=device)[:self.visual_token_num]
                    random_relevances = relevance[b, random_indices]
                    random_sim = similarity[b][random_indices][:, random_indices]
                    random_avg_sim = random_sim[mask].mean().item()
                    
                    print(f"\n  [Comparison with Random Selection]")
                    print(f"    Random avg relevance: {random_relevances.mean().item():.4f}")
                    print(f"    Random diversity: {1-random_avg_sim:.4f}")
                    print(f"    Relevance improvement: {(selected_relevances.mean() - random_relevances.mean()).item():.4f}")
                    print(f"    Diversity improvement: {(1-avg_sim) - (1-random_avg_sim):.4f}")
                
                print(f"\n{'='*80}\n")

        elif 'thcp_orig' in self.pruning_method:
            # Text-Visual Hierarchical Clustering Pruning (Simplified)
            # 核心思想：选择与文本相关且视觉多样的tokens

            # ========== 超参数配置（简化为2个） ==========
            text_keep_ratio = 0.1  # 保留的文本token比例
            diversity_weight = 0.9  # 多样性权重（0-1之间，越大越注重多样性）

            # ========== Step 1: 选择重要文本tokens ==========
            M = text_embeds.shape[0]
            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)

            # 简化的重要性计算：只用范数和唯一性，去掉位置权重
            text_norm = text_embeds.norm(dim=-1)
            text_sim_matrix = torch.matmul(text_normalized, text_normalized.t())
            text_uniqueness = 1 - (text_sim_matrix.sum(dim=-1) - 1) / max(M - 1, 1)

            # 简单加权
            text_importance = 0.9 * text_norm + 0.1 * text_uniqueness
            text_importance = text_importance / (text_importance.sum() + 1e-8)

            num_important_texts = max(1, int(M * text_keep_ratio))
            important_text_indices = torch.topk(text_importance, num_important_texts).indices
            important_texts = text_embeds[important_text_indices]
            important_texts_norm = important_texts / (important_texts.norm(dim=-1, keepdim=True) + 1e-8)

            # ========== Step 2: 为每个batch选择视觉tokens ==========
            all_masks = []

            for b in range(B):
                # 2.1 计算文本相关性得分（核心得分）
                image_normalized = image_embeds[b] / (image_embeds[b].norm(dim=-1, keepdim=True) + 1e-8)
                text_relevance = torch.matmul(image_normalized, important_texts_norm.t()).max(dim=1).values  # (N,)

                # 2.2 贪心选择：既要文本相关，又要视觉多样
                selected_indices = []
                available_mask = torch.ones(N, dtype=torch.bool, device=device)

                # 预计算视觉相似度矩阵（用于多样性判断）
                visual_normalized = image_features[b] / (image_features[b].norm(dim=-1, keepdim=True) + 1e-8)
                visual_similarity = torch.matmul(visual_normalized, visual_normalized.t())  # (N, N)

                for _ in range(self.visual_token_num):
                    if not available_mask.any():
                        break

                    # 计算候选得分 = 文本相关性
                    scores = text_relevance.clone()
                    scores[~available_mask] = -float('inf')

                    # 如果已经选了一些token，减去与已选token的相似度（鼓励多样性）
                    if len(selected_indices) > 0:
                        selected_tensor = torch.tensor(selected_indices, device=device)
                        max_similarity = visual_similarity[:, selected_tensor].max(dim=1).values
                        # 多样性惩罚：与已选token越相似，得分越低
                        scores = scores - diversity_weight * max_similarity

                    # 选择得分最高的token
                    selected_idx = torch.argmax(scores).item()
                    selected_indices.append(selected_idx)
                    available_mask[selected_idx] = False

                # 构建mask
                batch_mask = torch.zeros(N, dtype=torch.bool, device=device)
                batch_mask[torch.tensor(selected_indices, device=device)] = True
                all_masks.append(batch_mask)

            index_masks = torch.stack(all_masks, dim=0)  # (B, N)
        
        elif 'thcp_1' in self.pruning_method:
            # 改进的THCP：借鉴TRIM的文本相关性计算
            
            # ========== 超参数配置 ==========
            text_keep_ratio = 0.4  # 可以适当调高，因为现在不用max了
            diversity_weight = 0.9  # 可以适当降低，让文本相关性也发挥作用
            
            # ========== Step 1: 选择重要文本tokens（保持原有逻辑） ==========
            M = text_embeds.shape[0]
            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
            
            text_norm = text_embeds.norm(dim=-1)
            text_sim_matrix = torch.matmul(text_normalized, text_normalized.t())
            text_uniqueness = 1 - (text_sim_matrix.sum(dim=-1) - 1) / max(M - 1, 1)
            
            text_importance = 0.5 * text_norm + 0.5 * text_uniqueness
            text_importance = text_importance / (text_importance.sum() + 1e-8)
            
            num_important_texts = max(1, int(M * text_keep_ratio))
            important_text_indices = torch.topk(text_importance, num_important_texts).indices
            important_texts = text_embeds[important_text_indices]
            important_texts_norm = important_texts / (important_texts.norm(dim=-1, keepdim=True) + 1e-8)
            
            # ========== Step 2: 为每个batch选择视觉tokens ==========
            all_masks = []
            
            for b in range(B):
                # 关键改进：使用TRIM风格的平均相似度，而不是max
                image_normalized = image_embeds[b] / (image_embeds[b].norm(dim=-1, keepdim=True) + 1e-8)
                
                # 计算每个视觉token与所有重要文本token的相似度矩阵
                text_relevance_matrix = torch.matmul(image_normalized, important_texts_norm.t())  # (N, K)
                
                # 使用平均而不是max，这保留了区分度
                text_relevance = text_relevance_matrix.mean(dim=1)  # (N,)
                
                # 可选：使用加权平均，给相似度更高的文本词汇更大权重
                # weights = F.softmax(text_relevance_matrix * 3, dim=1)
                # text_relevance = (text_relevance_matrix * weights).sum(dim=1)
                
                # 归一化到[0, 1]区间，让数值范围与后面的相似度可比
                text_relevance = (text_relevance - text_relevance.min()) / \
                                (text_relevance.max() - text_relevance.min() + 1e-8)
                
                # 贪心选择：既要文本相关，又要视觉多样
                selected_indices = []
                available_mask = torch.ones(N, dtype=torch.bool, device=device)
                
                # 使用image_embeds而不是image_features计算相似度，保持特征空间统一
                visual_normalized = image_normalized  # 复用已经归一化的image_embeds
                visual_similarity = torch.matmul(visual_normalized, visual_normalized.t())  # (N, N)
                
                for step in range(self.visual_token_num):
                    if not available_mask.any():
                        break
                    
                    # 计算候选得分
                    scores = text_relevance.clone()
                    scores[~available_mask] = -float('inf')
                    
                    # 多样性惩罚
                    if len(selected_indices) > 0:
                        selected_tensor = torch.tensor(selected_indices, device=device)
                        max_similarity = visual_similarity[:, selected_tensor].max(dim=1).values
                        
                        # 现在text_relevance和similarity都在合理的范围内，可以直接相减
                        scores = scores - diversity_weight * max_similarity
                    
                    # 选择得分最高的token
                    selected_idx = torch.argmax(scores).item()
                    selected_indices.append(selected_idx)
                    available_mask[selected_idx] = False
                
                # 构建mask
                batch_mask = torch.zeros(N, dtype=torch.bool, device=device)
                batch_mask[torch.tensor(selected_indices, device=device)] = True
                all_masks.append(batch_mask)
            
            index_masks = torch.stack(all_masks, dim=0)
        
        elif 'thcp_v2' in self.pruning_method:
            # Text-Concept Coverage Maximization (Vectorized)
            # 核心思想：确保选出的visual tokens能够覆盖所有重要的文本概念
            
            # ========== 超参数配置 ==========
            coverage_weight = 0.1  # 文本覆盖的权重
            diversity_weight = 0.6  # 视觉多样性的权重
            
            # ========== Step 1: 计算文本重要性（不预先筛选） ==========
            M = text_embeds.shape[0]
            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
            
            all_masks = []
            
            for b in range(B):
                # 获取当前batch的embeddings
                image_emb_b = image_embeds[b]  # (N, C)
                image_emb_b_norm = image_emb_b / (image_emb_b.norm(dim=-1, keepdim=True) + 1e-8)
                
                # ========== Step 2: 计算文本-视觉响应矩阵 ==========
                response_matrix = torch.matmul(image_emb_b_norm, text_normalized.t())  # (N, M)
                
                # 计算文本重要性：能激活视觉的文本更重要
                text_visual_activation = response_matrix.max(dim=0).values  # (M,)
                text_sim_matrix = torch.matmul(text_normalized, text_normalized.t())
                text_uniqueness = 1 - (text_sim_matrix.sum(dim=-1) - 1) / max(M - 1, 1)
                text_importance = 0.7 * text_visual_activation + 0.3 * text_uniqueness  # (M,)
                text_importance = text_importance / (text_importance.sum() + 1e-8)
                
                # ========== Step 3: 覆盖导向的贪心选择 ==========
                selected_indices = []
                available_mask = torch.ones(N, dtype=torch.bool, device=device)
                
                # 预计算视觉相似度矩阵
                visual_feat_b = image_features[b]  # (N, D)
                visual_feat_b_norm = visual_feat_b / (visual_feat_b.norm(dim=-1, keepdim=True) + 1e-8)
                visual_similarity = torch.matmul(visual_feat_b_norm, visual_feat_b_norm.t())  # (N, N)
                
                # 追踪文本覆盖度
                text_coverage = torch.zeros(M, device=device)  # (M,)
                
                for step in range(self.visual_token_num):
                    if not available_mask.any():
                        break
                    
                    if step == 0:
                        # 第一个token：选择能覆盖最多重要文本的
                        coverage_scores = (response_matrix * text_importance.unsqueeze(0)).sum(dim=-1)  # (N,)
                        scores = coverage_scores
                    else:
                        # === 计算覆盖增益（向量化） ===
                        # 对于每个候选token，计算它能带来的新覆盖
                        new_coverage = torch.maximum(text_coverage.unsqueeze(0), response_matrix)  # (N, M)
                        coverage_gain = (new_coverage - text_coverage.unsqueeze(0)) * text_importance.unsqueeze(0)  # (N, M)
                        total_coverage_gain = coverage_gain.sum(dim=-1)  # (N,)
                        
                        # === 计算多样性得分（向量化） ===
                        selected_tensor = torch.tensor(selected_indices, device=device)
                        max_similarity_to_selected = visual_similarity[:, selected_tensor].max(dim=1).values  # (N,)
                        diversity_scores = 1 - max_similarity_to_selected  # (N,)
                        
                        # === 综合得分 ===
                        scores = coverage_weight * total_coverage_gain + diversity_weight * diversity_scores
                    
                    # 屏蔽已选择的tokens
                    scores[~available_mask] = -float('inf')
                    
                    # 选择得分最高的token
                    selected_idx = torch.argmax(scores).item()
                    selected_indices.append(selected_idx)
                    available_mask[selected_idx] = False
                    
                    # 更新文本覆盖度
                    text_coverage = torch.maximum(text_coverage, response_matrix[selected_idx])
                
                # 构建mask
                batch_mask = torch.zeros(N, dtype=torch.bool, device=device)
                batch_mask[torch.tensor(selected_indices, device=device)] = True
                all_masks.append(batch_mask)
            
            index_masks = torch.stack(all_masks, dim=0)  # (B, N)


        
        elif 'thcp_v3' in self.pruning_method:
            # Text-Concept Coverage Maximization (Vectorized)
            # 核心思想：确保选出的visual tokens能够覆盖所有重要的文本概念

            # ========== 超参数配置 ==========
            coverage_weight = 0.7  # 文本覆盖的权重
            diversity_weight = 0.3  # 视觉多样性的权重

            # ========== Debug配置 ==========
            enable_debug = True  # 设置为True启用debug
            debug_info = {
                'text_importance_stats': [],
                'coverage_progression': [],
                'score_breakdown': [],
                'final_coverage': [],
                'selection_diversity': []
            }

            # ========== Step 1: 计算文本重要性（不预先筛选） ==========
            M = text_embeds.shape[0]
            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)

            all_masks = []

            for b in range(B):
                # 获取当前batch的embeddings
                image_emb_b = image_embeds[b]  # (N, C)
                image_emb_b_norm = image_emb_b / (image_emb_b.norm(dim=-1, keepdim=True) + 1e-8)

                # ========== Step 2: 计算文本-视觉响应矩阵 ==========
                response_matrix = torch.matmul(image_emb_b_norm, text_normalized.t())  # (N, M)

                # 计算文本重要性：能激活视觉的文本更重要
                text_visual_activation = response_matrix.max(dim=0).values  # (M,)
                text_sim_matrix = torch.matmul(text_normalized, text_normalized.t())
                text_uniqueness = 1 - (text_sim_matrix.sum(dim=-1) - 1) / max(M - 1, 1)
                text_importance = 0.7 * text_visual_activation + 0.3 * text_uniqueness  # (M,)
                text_importance = text_importance / (text_importance.sum() + 1e-8)

                # Debug: 记录文本重要性统计
                if enable_debug:
                    debug_info['text_importance_stats'].append({
                        'batch_idx': b,
                        'mean': text_importance.mean().item(),
                        'std': text_importance.std().item(),
                        'max': text_importance.max().item(),
                        'min': text_importance.min().item(),
                        'top5_values': text_importance.topk(min(5, M)).values.cpu().tolist(),
                        'top5_indices': text_importance.topk(min(5, M)).indices.cpu().tolist(),
                        'activation_contribution': 0.7,  # 对应上面的0.7权重
                        'uniqueness_contribution': 0.3
                    })

                # ========== Step 3: 覆盖导向的贪心选择 ==========
                selected_indices = []
                available_mask = torch.ones(N, dtype=torch.bool, device=device)

                # 预计算视觉相似度矩阵
                visual_feat_b = image_features[b]  # (N, D)
                visual_feat_b_norm = visual_feat_b / (visual_feat_b.norm(dim=-1, keepdim=True) + 1e-8)
                visual_similarity = torch.matmul(visual_feat_b_norm, visual_feat_b_norm.t())  # (N, N)

                # 追踪文本覆盖度
                text_coverage = torch.zeros(M, device=device)  # (M,)

                # Debug: 记录每一步的覆盖进展和得分分解
                step_debug = []

                for step in range(self.visual_token_num):
                    if not available_mask.any():
                        break

                    if step == 0:
                        # 第一个token：选择能覆盖最多重要文本的
                        coverage_scores = (response_matrix * text_importance.unsqueeze(0)).sum(dim=-1)  # (N,)
                        scores = coverage_scores

                        # Debug: 第一步的统计
                        if enable_debug:
                            step_info = {
                                'step': step,
                                'coverage_gain': 0.0,
                                'diversity_score': 0.0,
                                'total_score': scores.max().item(),
                                'weighted_coverage': coverage_weight * scores.max().item(),
                                'weighted_diversity': 0.0,
                                'current_coverage_rate': 0.0
                            }
                    else:
                        # === 计算覆盖增益（向量化） ===
                        new_coverage = torch.maximum(text_coverage.unsqueeze(0), response_matrix)  # (N, M)
                        coverage_gain = (new_coverage - text_coverage.unsqueeze(0)) * text_importance.unsqueeze(0)  # (N, M)
                        total_coverage_gain = coverage_gain.sum(dim=-1)  # (N,)

                        # === 计算多样性得分（向量化） ===
                        selected_tensor = torch.tensor(selected_indices, device=device)
                        max_similarity_to_selected = visual_similarity[:, selected_tensor].max(dim=1).values  # (N,)
                        diversity_scores = 1 - max_similarity_to_selected  # (N,)

                        # === 综合得分 ===
                        scores = coverage_weight * total_coverage_gain + diversity_weight * diversity_scores

                        # Debug: 记录当前步骤的详细信息
                        if enable_debug:
                            valid_scores = scores[available_mask]
                            best_idx = torch.argmax(scores)
                            current_coverage_rate = (text_coverage > 0.5).float().mean().item()  # 覆盖率（阈值0.5）

                            step_info = {
                                'step': step,
                                'coverage_gain': total_coverage_gain[best_idx].item(),
                                'diversity_score': diversity_scores[best_idx].item(),
                                'total_score': scores[best_idx].item(),
                                'weighted_coverage': (coverage_weight * total_coverage_gain[best_idx]).item(),
                                'weighted_diversity': (diversity_weight * diversity_scores[best_idx]).item(),
                                'current_coverage_rate': current_coverage_rate,
                                'avg_coverage_gain': total_coverage_gain[available_mask].mean().item(),
                                'avg_diversity': diversity_scores[available_mask].mean().item()
                            }

                    # 屏蔽已选择的tokens
                    scores[~available_mask] = -float('inf')

                    # 选择得分最高的token
                    selected_idx = torch.argmax(scores).item()
                    selected_indices.append(selected_idx)
                    available_mask[selected_idx] = False

                    # 更新文本覆盖度
                    text_coverage = torch.maximum(text_coverage, response_matrix[selected_idx])

                    # Debug: 记录这一步的信息
                    if enable_debug:
                        step_debug.append(step_info)

                # Debug: 记录最终覆盖统计
                if enable_debug:
                    final_coverage_rate_05 = (text_coverage > 0.5).float().mean().item()
                    final_coverage_rate_03 = (text_coverage > 0.3).float().mean().item()
                    final_coverage_rate_07 = (text_coverage > 0.7).float().mean().item()
                    weighted_coverage = (text_coverage * text_importance).sum().item()

                    debug_info['coverage_progression'].append(step_debug)
                    debug_info['final_coverage'].append({
                        'batch_idx': b,
                        'coverage_rate_0.3': final_coverage_rate_03,
                        'coverage_rate_0.5': final_coverage_rate_05,
                        'coverage_rate_0.7': final_coverage_rate_07,
                        'weighted_coverage': weighted_coverage,
                        'mean_coverage': text_coverage.mean().item(),
                        'max_coverage': text_coverage.max().item(),
                        'min_coverage': text_coverage.min().item(),
                        'uncovered_texts': (text_coverage < 0.3).sum().item(),
                        'well_covered_texts': (text_coverage > 0.7).sum().item()
                    })

                    # 计算选择的tokens的多样性
                    if len(selected_indices) > 1:
                        selected_tensor = torch.tensor(selected_indices, device=device)
                        selected_similarity = visual_similarity[selected_tensor][:, selected_tensor]
                        # 去掉对角线（自己和自己的相似度）
                        mask = ~torch.eye(len(selected_indices), dtype=torch.bool, device=device)
                        avg_similarity = selected_similarity[mask].mean().item()
                        max_similarity = selected_similarity[mask].max().item()
                        min_similarity = selected_similarity[mask].min().item()
                    else:
                        avg_similarity = 0.0
                        max_similarity = 0.0
                        min_similarity = 0.0

                    debug_info['selection_diversity'].append({
                        'batch_idx': b,
                        'avg_pairwise_similarity': avg_similarity,
                        'max_pairwise_similarity': max_similarity,
                        'min_pairwise_similarity': min_similarity,
                        'diversity_score': 1 - avg_similarity
                    })

                # 构建mask
                batch_mask = torch.zeros(N, dtype=torch.bool, device=device)
                batch_mask[torch.tensor(selected_indices, device=device)] = True
                all_masks.append(batch_mask)

            index_masks = torch.stack(all_masks, dim=0)  # (B, N)

            # ========== Debug输出 ==========
            if enable_debug:
                print("\n" + "="*80)
                print("THCP Debug Information")
                print("="*80)

                # 1. 超参数
                print(f"\n[Hyperparameters]")
                print(f"  Coverage Weight: {coverage_weight}")
                print(f"  Diversity Weight: {diversity_weight}")
                print(f"  Visual Token Num: {self.visual_token_num}")

                # 2. 文本重要性统计（取第一个batch作为代表）
                if len(debug_info['text_importance_stats']) > 0:
                    text_stats = debug_info['text_importance_stats'][0]
                    print(f"\n[Text Importance Statistics] (Batch 0)")
                    print(f"  Mean: {text_stats['mean']:.4f}")
                    print(f"  Std: {text_stats['std']:.4f}")
                    print(f"  Max: {text_stats['max']:.4f}")
                    print(f"  Min: {text_stats['min']:.4f}")
                    print(f"  Top-5 Importance Values: {[f'{v:.4f}' for v in text_stats['top5_values']]}")
                    print(f"  Top-5 Text Indices: {text_stats['top5_indices']}")

                # 3. 覆盖进展（取第一个batch的前5步和后5步）
                if len(debug_info['coverage_progression']) > 0:
                    progression = debug_info['coverage_progression'][0]
                    print(f"\n[Coverage Progression] (Batch 0, First & Last 5 steps)")
                    print(f"  {'Step':<6} {'CovGain':<10} {'DivScore':<10} {'TotalScore':<12} {'CovRate':<10}")
                    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

                    # 前5步
                    for info in progression[:5]:
                        print(f"  {info['step']:<6} "
                            f"{info['coverage_gain']:<10.4f} "
                            f"{info['diversity_score']:<10.4f} "
                            f"{info['total_score']:<12.4f} "
                            f"{info['current_coverage_rate']:<10.2%}")

                    if len(progression) > 10:
                        print(f"  {'...':<6} {'...':<10} {'...':<10} {'...':<12} {'...':<10}")

                    # 后5步
                    for info in progression[-5:]:
                        print(f"  {info['step']:<6} "
                            f"{info['coverage_gain']:<10.4f} "
                            f"{info['diversity_score']:<10.4f} "
                            f"{info['total_score']:<12.4f} "
                            f"{info['current_coverage_rate']:<10.2%}")

                # 4. 最终覆盖统计（所有batch的平均）
                if len(debug_info['final_coverage']) > 0:
                    avg_stats = {
                        'coverage_rate_0.3': sum(x['coverage_rate_0.3'] for x in debug_info['final_coverage']) / len(debug_info['final_coverage']),
                        'coverage_rate_0.5': sum(x['coverage_rate_0.5'] for x in debug_info['final_coverage']) / len(debug_info['final_coverage']),
                        'coverage_rate_0.7': sum(x['coverage_rate_0.7'] for x in debug_info['final_coverage']) / len(debug_info['final_coverage']),
                        'weighted_coverage': sum(x['weighted_coverage'] for x in debug_info['final_coverage']) / len(debug_info['final_coverage']),
                        'mean_coverage': sum(x['mean_coverage'] for x in debug_info['final_coverage']) / len(debug_info['final_coverage']),
                    }

                    print(f"\n[Final Coverage Statistics] (Averaged over {len(debug_info['final_coverage'])} batches)")
                    print(f"  Coverage Rate (>0.3): {avg_stats['coverage_rate_0.3']:.2%}")
                    print(f"  Coverage Rate (>0.5): {avg_stats['coverage_rate_0.5']:.2%}")
                    print(f"  Coverage Rate (>0.7): {avg_stats['coverage_rate_0.7']:.2%}")
                    print(f"  Weighted Coverage: {avg_stats['weighted_coverage']:.4f}")
                    print(f"  Mean Coverage: {avg_stats['mean_coverage']:.4f}")

                # 5. 选择多样性统计
                if len(debug_info['selection_diversity']) > 0:
                    avg_diversity = {
                        'avg_similarity': sum(x['avg_pairwise_similarity'] for x in debug_info['selection_diversity']) / len(debug_info['selection_diversity']),
                        'max_similarity': sum(x['max_pairwise_similarity'] for x in debug_info['selection_diversity']) / len(debug_info['selection_diversity']),
                        'diversity_score': sum(x['diversity_score'] for x in debug_info['selection_diversity']) / len(debug_info['selection_diversity']),
                    }

                    print(f"\n[Selection Diversity] (Averaged over {len(debug_info['selection_diversity'])} batches)")
                    print(f"  Avg Pairwise Similarity: {avg_diversity['avg_similarity']:.4f}")
                    print(f"  Max Pairwise Similarity: {avg_diversity['max_similarity']:.4f}")
                    print(f"  Diversity Score: {avg_diversity['diversity_score']:.4f}")

                print("\n" + "="*80 + "\n")

                # 将debug信息保存到self，以便后续分析
                if not hasattr(self, 'thcp_debug_history'):
                    self.thcp_debug_history = []
                self.thcp_debug_history.append(debug_info)


        elif 'thcp' in self.pruning_method:
            # Text-Concept Coverage Maximization (Adaptive)
            # M>1: 使用coverage机制
            # M=1: 使用relevance+diversity机制（类似CDP3但用贪心而非DPP）
            
            # ========== 超参数配置 ==========
            coverage_weight = 0.7  # M>1时的覆盖权重
            relevance_weight = 0.5  # M=1时的相关性权重
            diversity_weight = 0.5  # 多样性权重
            
            # ========== Debug配置 ==========
            enable_debug = True
            debug_info = {
                'mode': None,
                'text_stats': [],
                'progression': [],
                'final_stats': []
            }
            
            M = text_embeds.shape[0]
            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
            
            # 判断使用哪种模式
            use_coverage_mode = (M > 1)
            debug_info['mode'] = 'coverage' if use_coverage_mode else 'relevance'
            
            if enable_debug:
                print(f"\n{'='*80}")
                print(f"THCP Debug - Mode: {debug_info['mode'].upper()}")
                print(f"{'='*80}")
                print(f"\n[Text Embeddings]")
                print(f"  Shape: {text_embeds.shape}")
                print(f"  M (num tokens): {M}")
                print(f"  Using {'Coverage' if use_coverage_mode else 'Relevance'} strategy")
            
            all_masks = []
            
            for b in range(B):
                image_emb_b = image_embeds[b]  # (N, C)
                image_emb_b_norm = image_emb_b / (image_emb_b.norm(dim=-1, keepdim=True) + 1e-8)
                
                # 计算文本-视觉响应矩阵
                response_matrix = torch.matmul(image_emb_b_norm, text_normalized.t())  # (N, M)
                
                selected_indices = []
                available_mask = torch.ones(N, dtype=torch.bool, device=device)
                
                # 预计算视觉相似度矩阵
                visual_feat_b = image_features[b]  # (N, D)
                visual_feat_b_norm = visual_feat_b / (visual_feat_b.norm(dim=-1, keepdim=True) + 1e-8)
                visual_similarity = torch.matmul(visual_feat_b_norm, visual_feat_b_norm.t())  # (N, N)
                
                step_debug = []
                
                if use_coverage_mode:
                    # ==================== 模式1：Coverage Mode (M > 1) ====================
                    # [保持原来的coverage代码]
                    text_visual_activation = response_matrix.max(dim=0).values
                    text_sim_matrix = torch.matmul(text_normalized, text_normalized.t())
                    text_uniqueness = 1 - (text_sim_matrix.sum(dim=-1) - 1) / max(M - 1, 1)
                    text_importance = 0.7 * text_visual_activation + 0.3 * text_uniqueness
                    text_importance = text_importance / (text_importance.sum() + 1e-8)
                    
                    text_coverage = torch.zeros(M, device=device)
                    
                    for step in range(self.visual_token_num):
                        if not available_mask.any():
                            break
                        
                        if step == 0:
                            coverage_scores = (response_matrix * text_importance.unsqueeze(0)).sum(dim=-1)
                            scores = coverage_scores
                        else:
                            new_coverage = torch.maximum(text_coverage.unsqueeze(0), response_matrix)
                            coverage_gain = (new_coverage - text_coverage.unsqueeze(0)) * text_importance.unsqueeze(0)
                            total_coverage_gain = coverage_gain.sum(dim=-1)
                            
                            selected_tensor = torch.tensor(selected_indices, device=device)
                            max_similarity_to_selected = visual_similarity[:, selected_tensor].max(dim=1).values
                            diversity_scores = 1 - max_similarity_to_selected
                            
                            scores = coverage_weight * total_coverage_gain + diversity_weight * diversity_scores
                        
                        scores[~available_mask] = -float('inf')
                        selected_idx = torch.argmax(scores).item()
                        selected_indices.append(selected_idx)
                        available_mask[selected_idx] = False
                        text_coverage = torch.maximum(text_coverage, response_matrix[selected_idx])
                
                else:
                    # ==================== 模式2：Relevance Mode (M = 1) ====================
                    # 类似CDP3的思路，但用贪心而非DPP
                    
                    # 计算相关性得分（类似CDP3）
                    text_relevance = response_matrix.squeeze(-1)  # (N,)
                    text_relevance = -text_relevance  # 取负（和CDP3一样）
                    text_relevance = (text_relevance - text_relevance.min() + 1e-6) / (text_relevance.max() - text_relevance.min())
                    
                    if enable_debug and b == 0:
                        print(f"\n[Relevance Scores] (Batch {b})")
                        print(f"  Mean: {text_relevance.mean().item():.4f}")
                        print(f"  Std: {text_relevance.std().item():.4f}")
                        print(f"  Range: [{text_relevance.min().item():.4f}, {text_relevance.max().item():.4f}]")
                        print(f"  Top-5 values: {text_relevance.topk(5).values.cpu().tolist()}")
                        print(f"  Top-5 indices: {text_relevance.topk(5).indices.cpu().tolist()}")
                    
                    # 🔥 关键改进：用weighted scoring而不是coverage
                    # 核心思想：相关性高的token权重高，但要避免选择视觉上冗余的
                    
                    for step in range(self.visual_token_num):
                        if not available_mask.any():
                            break
                        
                        if step == 0:
                            # 第一个token：选择相关性最高的
                            scores = text_relevance.clone()
                        else:
                            # 后续tokens：平衡相关性和多样性
                            selected_tensor = torch.tensor(selected_indices, device=device)
                            
                            # 🔥 核心改进：使用加权方案而不是coverage
                            # 相关性得分（保持每个token的原始相关性）
                            relevance_scores = text_relevance.clone()
                            
                            # 多样性得分（与已选tokens的差异）
                            max_similarity_to_selected = visual_similarity[:, selected_tensor].max(dim=1).values
                            diversity_scores = 1 - max_similarity_to_selected
                            
                            # 🔥 关键：加权融合（类似CDP3的kernel思想）
                            # 但我们用加法而不是乘法，因为我们要保持interpretability
                            scores = relevance_weight * relevance_scores + diversity_weight * diversity_scores
                        
                        scores[~available_mask] = -float('inf')
                        selected_idx = torch.argmax(scores).item()
                        selected_indices.append(selected_idx)
                        available_mask[selected_idx] = False
                        
                        if enable_debug and step % 10 == 0 and b == 0:
                            step_debug.append({
                                'step': step,
                                'selected_idx': selected_idx,
                                'selected_relevance': text_relevance[selected_idx].item()
                            })
                
                # 构建mask
                batch_mask = torch.zeros(N, dtype=torch.bool, device=device)
                batch_mask[torch.tensor(selected_indices, device=device)] = True
                all_masks.append(batch_mask)
                
                # Debug最终统计
                if enable_debug and b == 0:
                    selected_tensor = torch.tensor(selected_indices, device=device)
                    
                    if use_coverage_mode:
                        coverage_rate = (text_coverage > 0.5).float().mean().item()
                        weighted_coverage = (text_coverage * text_importance).sum().item()
                        debug_info['final_stats'].append({
                            'batch': b,
                            'coverage_rate': coverage_rate,
                            'weighted_coverage': weighted_coverage
                        })
                    else:
                        avg_relevance = text_relevance[selected_tensor].mean().item()
                        selected_sim = visual_similarity[selected_tensor][:, selected_tensor]
                        mask = ~torch.eye(len(selected_indices), dtype=torch.bool, device=device)
                        avg_diversity = (1 - selected_sim[mask].mean()).item() if len(selected_indices) > 1 else 1.0
                        
                        # 对比随机选择
                        random_indices = torch.randperm(N, device=device)[:self.visual_token_num]
                        random_relevance = text_relevance[random_indices].mean().item()
                        
                        debug_info['final_stats'].append({
                            'batch': b,
                            'avg_relevance': avg_relevance,
                            'avg_diversity': avg_diversity,
                            'random_relevance': random_relevance,
                            'improvement': avg_relevance - random_relevance
                        })
                        
                        print(f"\n[Selection Progress] (every 10 steps)")
                        print(f"  {'Step':<6} {'SelIdx':<8} {'SelRel':<10}")
                        print(f"  {'-'*6} {'-'*8} {'-'*10}")
                        for info in step_debug:
                            print(f"  {info['step']:<6} {info['selected_idx']:<8} {info['selected_relevance']:<10.4f}")
                        
                        print(f"\n[Final Statistics] (Batch {b})")
                        print(f"  Avg relevance: {avg_relevance:.4f}")
                        print(f"  Avg diversity: {avg_diversity:.4f}")
                        print(f"  Random baseline: {random_relevance:.4f}")
                        print(f"  Improvement: {avg_relevance - random_relevance:.4f}")
    
            index_masks = torch.stack(all_masks, dim=0)
            
            if enable_debug:
                print(f"\n{'='*80}\n")

        elif self.pruning_method == 'star_v3':
            # STAR-V3: Two-Stage Framework with THCP Stage 1
            # Stage 1 (here): THCP text-concept coverage maximization - keep 50% tokens
            # Stage 2 (in modeling_llama_star): Multi-token text-guided progressive pruning

            print(f"\n{'='*80}")
            print(f"STAR-V3 Stage 1: THCP Text-Concept Coverage")
            print(f"{'='*80}")

            # ========== 超参数配置（与THCP一致） ==========
            coverage_weight = 0.7  # M>1时的覆盖权重
            relevance_weight = 0.5  # M=1时的相关性权重
            diversity_weight = 0.5  # 多样性权重

            M = text_embeds.shape[0]
            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)

            # Stage 1 target: keep 50% for Stage 2 (same as STAR-V2)
            stage1_keep_num = N // 2  # 576 -> 288

            print(f"[Stage 1 Config]")
            print(f"  Original tokens: {N}")
            print(f"  Stage 1 keeps: {stage1_keep_num} (50%)")
            print(f"  Final target: {self.visual_token_num}")
            print(f"  Text tokens (M): {M}")

            # 判断使用哪种模式
            use_coverage_mode = (M > 1)
            print(f"  Using {'Coverage' if use_coverage_mode else 'Relevance'} mode")

            all_masks = []

            for b in range(B):
                image_emb_b = image_embeds[b]  # (N, C)
                image_emb_b_norm = image_emb_b / (image_emb_b.norm(dim=-1, keepdim=True) + 1e-8)

                # 计算文本-视觉响应矩阵
                response_matrix = torch.matmul(image_emb_b_norm, text_normalized.t())  # (N, M)

                selected_indices = []
                available_mask = torch.ones(N, dtype=torch.bool, device=device)

                # 预计算视觉相似度矩阵
                visual_feat_b = image_features[b]  # (N, D)
                visual_feat_b_norm = visual_feat_b / (visual_feat_b.norm(dim=-1, keepdim=True) + 1e-8)
                visual_similarity = torch.matmul(visual_feat_b_norm, visual_feat_b_norm.t())  # (N, N)

                if use_coverage_mode:
                    # ==================== Coverage Mode (M > 1) ====================
                    text_visual_activation = response_matrix.max(dim=0).values
                    text_sim_matrix = torch.matmul(text_normalized, text_normalized.t())
                    text_uniqueness = 1 - (text_sim_matrix.sum(dim=-1) - 1) / max(M - 1, 1)
                    text_importance = 0.7 * text_visual_activation + 0.3 * text_uniqueness
                    text_importance = text_importance / (text_importance.sum() + 1e-8)

                    text_coverage = torch.zeros(M, device=device)

                    # 🔥 关键：选择 stage1_keep_num 个tokens（而不是 self.visual_token_num）
                    for step in range(stage1_keep_num):
                        if not available_mask.any():
                            break

                        if step == 0:
                            coverage_scores = (response_matrix * text_importance.unsqueeze(0)).sum(dim=-1)
                            scores = coverage_scores
                        else:
                            new_coverage = torch.maximum(text_coverage.unsqueeze(0), response_matrix)
                            coverage_gain = (new_coverage - text_coverage.unsqueeze(0)) * text_importance.unsqueeze(0)
                            total_coverage_gain = coverage_gain.sum(dim=-1)

                            selected_tensor = torch.tensor(selected_indices, device=device)
                            max_similarity_to_selected = visual_similarity[:, selected_tensor].max(dim=1).values
                            diversity_scores = 1 - max_similarity_to_selected

                            scores = coverage_weight * total_coverage_gain + diversity_weight * diversity_scores

                        scores[~available_mask] = -float('inf')
                        selected_idx = torch.argmax(scores).item()
                        selected_indices.append(selected_idx)
                        available_mask[selected_idx] = False
                        text_coverage = torch.maximum(text_coverage, response_matrix[selected_idx])

                else:
                    # ==================== Relevance Mode (M = 1) ====================
                    text_relevance = response_matrix.squeeze(-1)  # (N,)
                    text_relevance = -text_relevance
                    text_relevance = (text_relevance - text_relevance.min() + 1e-6) / (text_relevance.max() - text_relevance.min())

                    # 🔥 关键：选择 stage1_keep_num 个tokens（而不是 self.visual_token_num）
                    for step in range(stage1_keep_num):
                        if not available_mask.any():
                            break

                        if step == 0:
                            scores = text_relevance.clone()
                        else:
                            selected_tensor = torch.tensor(selected_indices, device=device)
                            relevance_scores = text_relevance.clone()
                            max_similarity_to_selected = visual_similarity[:, selected_tensor].max(dim=1).values
                            diversity_scores = 1 - max_similarity_to_selected
                            scores = relevance_weight * relevance_scores + diversity_weight * diversity_scores

                        scores[~available_mask] = -float('inf')
                        selected_idx = torch.argmax(scores).item()
                        selected_indices.append(selected_idx)
                        available_mask[selected_idx] = False

                # 构建mask（关键：保持 (N,) 维度，和THCP一样）
                batch_mask = torch.zeros(N, dtype=torch.bool, device=device)
                batch_mask[torch.tensor(selected_indices, device=device)] = True
                all_masks.append(batch_mask)

            # 🔥 关键：生成 (B, N) 的 index_masks，和THCP完全一样
            index_masks = torch.stack(all_masks, dim=0)

            print(f"\n[Stage 1 Output]")
            print(f"  index_masks shape: {index_masks.shape}  # (B, N)")
            print(f"  Selected tokens per batch: {index_masks.sum(dim=1).tolist()}")
            print(f"  image_features unchanged: {image_features.shape}")
            print(f"  Ready for mm_projector and Stage 2")
            print(f"{'='*80}\n")

        elif self.pruning_method == 'star_v2':
            # Two-Stage Pruning Framework
            # Stage 1 (here in llava_arch): Visual diversity-aware pruning - keep 50% tokens
            # Stage 2 (in modeling_llama_star): Text-guided progressive pruning

            print(f"\n{'='*80}")
            print(f"STAR-V2 Stage 1: Diversity-Aware Visual Pruning")
            print(f"{'='*80}")

            # Get visual self-attention
            image_features_attn, image_attentions, image_keys, image_cls = self.get_model().get_vision_tower()(images, output_attentions=True)
            image_features = image_features_attn

            B, N, C = image_features.shape
            device = image_features.device

            # Stage 1: Keep 50% of original tokens using diversity-aware selection
            stage1_keep_num = N // 2  # 576 -> 288, or 2880 -> 1440

            print(f"[Stage 1 Config]")
            print(f"  Original tokens: {N}")
            print(f"  Stage 1 keeps: {stage1_keep_num} (50%)")
            print(f"  Target tokens: {self.visual_token_num}")
            print(f"  Stage 2 will prune: {stage1_keep_num} -> {self.visual_token_num}")

            # CLS attention as importance score
            cls_attn = image_attentions.mean(dim=1)  # (B, N)

            # Compute similarity matrix for diversity
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity_matrix = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)

            print(f"\n[Stage 1 Selection: De-redundancy via Self-Similarity]")
            print(f"  Method: Remove redundant tokens with high self-similarity")

            # De-redundancy: remove tokens with high similarity to others
            # For each token, find its max similarity to all other tokens
            # Tokens with high max_sim are redundant (can be represented by others)

            # Set diagonal to 0 to exclude self-similarity
            similarity_matrix_masked = similarity_matrix.clone()
            for b in range(B):
                similarity_matrix_masked[b].fill_diagonal_(0)

            # Find max similarity to any other token
            max_similarity, _ = similarity_matrix_masked.max(dim=2)  # (B, N)

            # Remove the most redundant N//2 tokens (highest max_similarity)
            # Keep the less redundant N//2 tokens (lowest max_similarity)
            redundancy_scores = max_similarity

            # Select tokens with LOWEST redundancy (keep diverse tokens)
            stage1_indices = redundancy_scores.topk(k=stage1_keep_num, dim=1, largest=False).indices
            stage1_indices = stage1_indices.sort(dim=1).values  # Keep spatial order

            print(f"  CLS attention stats: mean={cls_attn.mean():.4f}, std={cls_attn.std():.4f}")
            print(f"  Selected indices shape: {stage1_indices.shape}")

            # Check diversity of selected tokens
            selected_features = torch.gather(
                image_normalized,
                dim=1,
                index=stage1_indices.unsqueeze(-1).expand(-1, -1, C)
            )
            selected_sim = torch.matmul(selected_features, selected_features.transpose(1, 2))
            avg_sim = (selected_sim.sum(dim=(1,2)) - stage1_keep_num) / (stage1_keep_num * (stage1_keep_num - 1))
            print(f"  Avg pairwise similarity of selected: {avg_sim.mean():.4f} (lower is more diverse)")

            # Gather selected features
            stage1_features = torch.gather(
                image_features,
                dim=1,
                index=stage1_indices.unsqueeze(-1).expand(-1, -1, C)
            )

            # Update image_features for Stage 2
            image_features = stage1_features

            # Create index masks (all True for now, Stage 2 will further prune)
            index_masks = torch.ones(B, stage1_keep_num, dtype=torch.bool, device=device)
            merged_features = None

            print(f"\n[Stage 1 Output (before projection)]")
            print(f"  Output shape: {image_features.shape}")
            print(f"  Ready for Stage 2 text-guided progressive pruning")
            print(f"{'='*80}\n")

        elif self.pruning_method == 'star_v2_anchor':
            # Two-Stage Pruning Framework with Anchor-based De-redundancy
            # Stage 1: Select anchors (CLS attention) + remove similar neighbors
            # Stage 2: Text-guided progressive pruning

            print(f"\n{'='*80}")
            print(f"STAR-V2-Anchor Stage 1: Anchor-based De-redundancy")
            print(f"{'='*80}")

            # Get visual self-attention
            image_features_attn, image_attentions, image_keys, image_cls = self.get_model().get_vision_tower()(images, output_attentions=True)
            image_features = image_features_attn

            B, N, C = image_features.shape
            device = image_features.device

            stage1_keep_num = N // 2  # 576 -> 288

            print(f"[Stage 1 Config]")
            print(f"  Original tokens: {N}")
            print(f"  Stage 1 keeps: {stage1_keep_num} (50%)")
            print(f"  Target tokens: {self.visual_token_num}")

            # CLS attention as importance score
            cls_attn = image_attentions.mean(dim=1)  # (B, N)

            # Compute similarity matrix
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity_matrix = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)

            print(f"\n[Step 1: Select Anchor Tokens]")
            # Step 1: Select anchor tokens using CLS attention
            num_anchors = stage1_keep_num // 2  # 144 anchors
            anchor_indices = cls_attn.topk(k=num_anchors, dim=1).indices  # (B, num_anchors)
            print(f"  Anchors: {num_anchors} tokens (top CLS attention)")

            print(f"\n[Step 2: De-redundancy Around Anchors]")
            # Step 2: For non-anchor tokens, keep those with LOW similarity to anchors
            all_indices = set(range(N))
            selected_indices = []

            for b in range(B):
                anchors = anchor_indices[b].tolist()
                selected = set(anchors)  # Start with anchors
                non_anchors = list(all_indices - selected)

                if len(non_anchors) > 0:
                    non_anchor_tensor = torch.tensor(non_anchors, device=device)
                    anchor_tensor = anchor_indices[b]

                    # Similarity from non-anchors to anchors (anchor queries neighbors)
                    similarities = similarity_matrix[b, non_anchor_tensor[:, None], anchor_tensor]

                    # Max similarity to any anchor
                    max_sim_to_anchors, _ = similarities.max(dim=1)

                    # Keep non-anchors with LOW similarity (diverse, not redundant)
                    num_to_keep = stage1_keep_num - num_anchors
                    if num_to_keep > 0:
                        keep_mask = max_sim_to_anchors.topk(k=min(num_to_keep, len(non_anchors)),
                                                            dim=0, largest=False).indices
                        kept_non_anchors = non_anchor_tensor[keep_mask].tolist()
                        selected.update(kept_non_anchors)

                selected_indices.append(sorted(list(selected)))

            stage1_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
            print(f"  Non-anchors kept: {stage1_keep_num - num_anchors} (low similarity to anchors)")
            print(f"  Interpretation: Anchors query neighbors, remove redundant")

            # Gather selected features
            stage1_features = torch.gather(
                image_features,
                dim=1,
                index=stage1_indices.unsqueeze(-1).expand(-1, -1, C)
            )

            image_features = stage1_features
            index_masks = torch.ones(B, stage1_keep_num, dtype=torch.bool, device=device)
            merged_features = None

            print(f"\n[Stage 1 Output (before projection)]")
            print(f"  Output shape: {image_features.shape}")
            print(f"  Ready for Stage 2 text-guided progressive pruning")
            print(f"{'='*80}\n")

        # Note: mm_projector is already applied at line 307
        # No need to apply again here
        return image_features, index_masks, merged_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities=["image"], image_sizes=None, texts=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks, merged_features = self.encode_images(concat_images, texts=texts)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            if merged_features is not None:
                merged_features = torch.split(merged_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
                image_features = [x[m] for x, m in zip(image_features, index_masks)]
                if merged_features is not None:
                    image_features = [torch.cat((x.reshape(y.shape[0], x.shape[0] // y.shape[0], *x.shape[1:]), y), dim=1).flatten(0, 1) \
                                      for x, y in zip(image_features, merged_features)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                if merged_features is None:
                    merged_features = [None] * len(image_features)
                for image_idx, (image_feature, index_mask, merged_feature) in enumerate(zip(image_features, index_masks, merged_features)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        base_index_mask = index_mask[0]
                        index_mask = index_mask[1:]
                        if merged_feature is not None:
                            base_merged_feature = merged_feature[0]
                            merged_feature = merged_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                            if merged_feature is not None:
                                image_feature = torch.cat((
                                    image_feature,
                                    merged_feature.flatten(0, 1)
                                ))
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                            if merged_feature is not None:
                                image_feature = torch.cat((
                                    image_feature,
                                    merged_feature.flatten(0, 1)
                                ))
                        base_image_feature = base_image_feature[base_index_mask]
                        if merged_feature is not None:
                            base_image_feature = torch.cat((base_image_feature, base_merged_feature))
                        image_feature = torch.cat((base_image_feature, image_feature))
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if merged_feature is not None:
                            merged_feature = merged_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                        if merged_feature is not None:
                            image_feature = torch.cat((image_feature, merged_feature))
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, index_masks, merged_features = self.encode_images(images, texts=texts)
            image_features = image_features[index_masks].unsqueeze(0)
            if merged_features is not None:
                image_features = torch.cat((image_features, merged_features), dim=1)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.z
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features[0].shape[0]

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
