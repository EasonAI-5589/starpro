# MustDrop 技术实现文档

> 分析日期：2026-01-23
> 源码路径：/tmp/MustDrop/

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MustDrop 三阶段 Token Pruning                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: Vision Encoder (CLIP ViT)                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Layer 0:  Token Merging (3×3 window, cosine similarity ≥ 0.8)    │  │
│  │           576 tokens → ~64 tokens (自适应)                        │  │
│  │ Layer 23: Key Token Set Extraction (top 8% by CLS attention)     │  │
│  │           提取关键 token 索引，传递给 LLM                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  Stage 2: LLM Prefilling                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Pruning Layers: [2, 6, 10, 14]                                   │  │
│  │ 每层使用 Dual Attention Filter:                                   │  │
│  │   - Global Attention: sum(text→img) < threshold                  │  │
│  │   - Individual Attention: max(text→img) < threshold              │  │
│  │ Key Token Set 始终被保护，不会被剪枝                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  Stage 3: KV Cache Sparsification                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 将 Prefill 阶段的剪枝索引反向传播到 KV Cache                       │  │
│  │ 不同层的 KV Cache 被稀疏化到不同程度                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、关键文件列表

| 文件 | 作用 |
|------|------|
| `llava/model/multimodal_encoder/clip_encoder.py` | Vision encoder 修改，Layer 0 merging + Layer 23 key set |
| `llava/model/multimodal_encoder/local_merge.py` | Token merging 算法实现 |
| `llava/model/language_model/modelling_sparse_llama.py` | LLM 剪枝核心，Dual Attention Filter |
| `llava/model/language_model/sparse_llava_llama.py` | LLaVA 集成层 |
| `llava/model/llava_arch.py` | 多模态输入准备，key_set 传递 |

---

## 三、Stage 1: Vision Encoder 修改

### 3.1 clip_encoder.py - CLIPVisionModelAcc

**位置**：`clip_encoder.py:30-126`

重写 `custom_forward` 方法：

```python
# clip_encoder.py:86-112
def custom_forward(self, inputs_embeds, ...):
    for idx, encoder_layer in enumerate(layers):

        # Layer 23: 提取 Key Token Set
        if idx == 23:
            output_attentions = True  # 强制输出 attention

        layer_outputs = encoder_layer(hidden_states, ...)
        hidden_states = layer_outputs[0]

        # Layer 23: 使用 CLS attention 提取关键 token
        if idx == 23:
            self.keep_rate = 0.08  # 保留 8%
            left_tokens = math.ceil(self.keep_rate * (N-1))  # 0.08 * 576 ≈ 46
            cls_attn = layer_outputs[1][:, :, 0, 1:]  # [B, nhead, 1, 576] → [B, nhead, 576]
            cls_attn = cls_attn.mean(dim=1)  # [B, 576]
            _, index = torch.topk(cls_attn, left_tokens, dim=1, largest=True)

        # Layer 0: Token Merging
        if idx == 0:
            merge = conditional_pooling(hidden_states, 0.8, (3, 3))
            hidden_states, size = merge_wavg(merge, hidden_states, None)

    # 关键：将 key_set (index) 通过 attentions 字段传递出去
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=index  # ⚠️ 不是 attention，而是 key token indices
    )
```

### 3.2 local_merge.py - conditional_pooling

**位置**：`local_merge.py:30-133`

**算法逻辑**：
1. 将 576 tokens 重排成 24×24 grid
2. 划分为 (24/3)×(24/3) = 8×8 = 64 个 super patches
3. 计算每个 super patch 内 9 个 token 的两两余弦相似度
4. 根据 threshold (0.8) 自适应选择高相似度的 super patches 进行合并

```python
# local_merge.py:30-133
def conditional_pooling(feat, threshold, window_size):
    ws_h, ws_w = int(window_size[0]), int(window_size[1])  # 3, 3
    num_token_window = ws_h * ws_w  # 9

    x_cls, feat = feat[:, :1, :], feat[:, 1:, :]  # 分离 CLS token
    B, N, D = feat.size()  # N = 576

    # 重排为 grid
    feat = rearrange(feat, "b (h w) c -> b c h w", h=24)
    feat = rearrange(feat, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w',
                     gh=8, gw=8)  # [B, 8, 8, C, 3, 3]

    # 计算每个 3×3 窗口内的两两余弦相似度
    tensor_flattened = feat.reshape(b, 8, 8, c, -1)  # [B, 8, 8, C, 9]
    tensor_1 = tensor_flattened.unsqueeze(-1)
    tensor_2 = tensor_flattened.unsqueeze(-2)
    sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

    # 排除自相似度
    sims_mask = 1 - torch.eye(9).to(sims.device)
    sims = sims * sims_mask

    # 计算每个 super patch 的平均相似度
    similarity_map = sims.sum(-1).sum(-1) / (9 * 8)  # [B, 64]

    # 自适应选择：相似度 ≥ threshold 的 super patches 数量
    node_mean = torch.tensor(threshold).cuda(sims.device)
    r = torch.ge(similarity_map, node_mean).sum(dim=1).min()
    _, sim_super_patch_idxs = similarity_map.topk(r, dim=-1)

    # 创建索引映射
    # gathered_tensor: 被合并的 super patches
    # remaining_tensor: 未合并的 super patches
    # ...

    # 创建 merge 函数
    def merge(x, mode="mean"):
        x_cls, x_feat = x[:, :1, :], x[:, 1:, :]
        src = x_feat.gather(dim=-2, index=src_idx.expand(...))
        dst = x_feat.gather(dim=-2, index=dst_idx.expand(...))
        unm = x_feat.gather(dim=-2, index=unm_idx.expand(...))
        dst = dst.scatter_reduce(-2, merge_idx, src, reduce=mode)
        x = torch.cat([dst, unm], dim=1)
        x = torch.cat((x_cls, x), dim=1)
        return x

    return merge
```

---

## 四、Stage 2: LLM Prefilling 修改

### 4.1 modelling_sparse_llama.py - LlamaDynamicvitModel

**位置**：`modelling_sparse_llama.py:90-354`

**关键配置**：
```python
# modelling_sparse_llama.py:103
self.pruning_layers = [2, 6, 10, 14]  # 在这4层执行剪枝
```

### 4.2 Dual Attention Filter（核心剪枝逻辑）

**位置**：`modelling_sparse_llama.py:336-354`

```python
def dual_attention_filter(self, attn, v_token_start, img_seq, global_thr=1.0, individual_thr=0.0):
    """
    attn: [B, nhead, total_seq, total_seq]
    """
    b, head, total_seq, _ = attn.shape
    attn = attn[0]  # 只取第一个 batch

    text_seq = total_seq - img_seq - v_token_start

    # 提取 text → image 的 attention
    # attn[text_tokens, image_tokens]
    attn = attn.narrow(1, v_token_start + img_seq, text_seq) \
               .narrow(2, v_token_start, img_seq)  # [nhead, text_seq, img_seq]

    attn = attn.mean(dim=0)  # [text_seq, img_seq]
    attn_sum = attn.sum(dim=0)  # [img_seq], 全局注意力

    # 双重条件过滤：
    # 1. Global: 某个 image token 的总注意力 < 阈值
    # 2. Individual: 某个 image token 对任何 text token 的最大注意力 < 阈值
    candidate_mask = (attn_sum < attn_sum.sum(dim=0) * global_thr) & \
                     (attn.max(dim=0)[0] < individual_thr)

    # 返回需要保留的 token indices
    all_indices = torch.arange(img_seq, device=attn.device)
    important_indices = all_indices[~candidate_mask]
    return important_indices, text_seq
```

### 4.3 Prefilling 阶段的剪枝执行

**位置**：`modelling_sparse_llama.py:252-267`

```python
for layer_idx, decoder_layer in enumerate(self.layers):

    # 只在指定层执行剪枝
    if shape[1] != 1 and len(pre_prompt_length_list) != 0 and layer_idx in self.pruning_layers:

        # 开启 attention 输出
        output_attentions = True

        layer_outputs = decoder_layer(hidden_states, ..., output_attentions=output_attentions)
        hidden_states = layer_outputs[0]

        # 获取需要保留的 vision token indices
        vis_indices, text_seq = self.dual_attention_filter(
            layer_outputs[1], v_token_start, img_seq, global_thr, individual_thr
        )

        # ⚠️ 关键：保护 Key Token Set
        vis_indices = torch.cat((vis_indices, key_set))  # 合并
        vis_indices = torch.unique(vis_indices)           # 去重

        # 更新 key_set 在新索引中的位置
        key_set = torch.nonzero(torch.isin(vis_indices, key_set)).squeeze()

        # 重构 hidden states
        sys_indices = position_ids[:, :v_token_start][0]
        text_indices = position_ids[:, v_token_start + img_seq:][0]
        merged_indices = torch.cat((sys_indices, vis_indices + v_token_start, text_indices), dim=0)

        img_seq = vis_indices.shape[0]  # 更新剩余 token 数
        final_indices.append(merged_indices)  # 记录每层的索引

        position_ids = position_ids.narrow(1, 0, v_token_start + img_seq + text_seq)
        hidden_states = torch.index_select(hidden_states, 1, merged_indices)
```

---

## 五、Stage 3: KV Cache Sparsification

**位置**：`modelling_sparse_llama.py:300-325`

```python
if shape[1] != 1 and len(pre_prompt_length_list) != 0:
    stage = 0
    next_decoder_cache_new = []

    # 反向传播索引：后面层的索引映射到前面层
    for i in range(len(final_indices) - 1, 0, -1):
        final_indices[i-1] = torch.index_select(final_indices[i-1], 0, final_indices[i])

    # Layer 0-1: 不剪枝，保留原始 KV Cache
    for i in range(self.pruning_layers[0]):  # range(2)
        next_decoder_cache_new.append((next_cache[i][0], next_cache[i][1]))

    # Layer 2-14: 渐进式稀疏化 KV Cache
    for i in range(self.pruning_layers[0], self.pruning_layers[-1] + 1):  # range(2, 15)
        next_decoder_cache_new.append((
            torch.index_select(next_cache[i][0], 2, final_indices[stage]),
            torch.index_select(next_cache[i][1], 2, final_indices[stage])
        ))
        if i == self.pruning_layers[stage]:
            stage += 1

    # Layer 15-31: 使用最终稀疏索引（不再变化）
    for i in range(self.pruning_layers[-1] + 1, 32):  # range(15, 32)
        next_decoder_cache_new.append((next_cache[i][0], next_cache[i][1]))

    next_decoder_cache = tuple(next_decoder_cache_new)
```

---

## 六、数据流与接口

### 6.1 CLIPVisionTower.forward 返回值

**位置**：`clip_encoder.py:167-180`

```python
@torch.no_grad()
def forward(self, images):
    image_forward_outs = self.vision_tower(images, output_hidden_states=True)
    image_features = self.feature_select(image_forward_outs).to(images.dtype)

    # 返回 (image_features, key_set)
    # key_set 是通过 attentions 字段传递的
    return image_features, image_forward_outs.attentions
```

### 6.2 llava_arch.py - 多模态输入准备

```python
def prepare_sparse_inputs_labels_for_multimodal(self, ...):
    # 1. 编码图像，获取 features 和 key_set
    image_features, key_set = self.encode_images(images)

    # 2. 存储到模型属性
    self.img_seq = image_features[0].shape[0]
    self.key_set = [key_set.shape[0], key_set.flatten()]  # [num_keys, key_indices]
    self.token_length_list = token_length_list
    self.pre_prompt_length_list = pre_prompt_length_list

    return (..., img_seq, token_length_list, pre_prompt_length_list)
```

### 6.3 sparse_llava_llama.py - 集成层

**位置**：`sparse_llava_llama.py:49-108`

```python
class LlavaLlamaDynamicForCausalLM(LlamaDynamicvitForCausalLM, LlavaMetaForCausalLM):

    def __init__(self, config):
        LlamaDynamicvitForCausalLM.__init__(self, config)
        self.model = LlavaLlamaDynamicModel(config)
        self.img_seq = 576
        self.key_set = None
        self.token_length_list = []
        self.pre_prompt_length_list = []

    def forward(self, ..., images=None, global_thr=1.0, individual_thr=0.0):

        if inputs_embeds is None:
            # 调用多模态输入准备
            (..., img_seq, token_length_list, pre_prompt_length_list) = \
                self.prepare_sparse_inputs_labels_for_multimodal(...)

        # 调用父类 forward，传递稀疏参数
        return super().forward(
            ...,
            img_seq=self.img_seq,
            key_set=self.key_set,
            token_length_list=token_length_list,
            pre_prompt_length_list=pre_prompt_length_list,
            global_thr=global_thr,
            individual_thr=individual_thr
        )
```

---

## 七、关键参数汇总

| 阶段 | 参数名 | 默认值 | 位置 | 说明 |
|------|--------|--------|------|------|
| Vision | `window_size` | `(3, 3)` | clip_encoder.py:111 | Token merging 窗口大小 |
| Vision | `threshold` | `0.8` | clip_encoder.py:111 | 余弦相似度阈值 |
| Vision | `keep_rate` | `0.08` | clip_encoder.py:102 | Key token 保留比例 (8%) |
| LLM | `pruning_layers` | `[2, 6, 10, 14]` | modelling_sparse_llama.py:103 | 执行剪枝的层 |
| LLM | `global_thr` | `1.0` | modelling_sparse_llama.py:135 | 全局注意力阈值 |
| LLM | `individual_thr` | `0.0` | modelling_sparse_llama.py:136 | 单独注意力阈值 |

---

## 八、与 STAR-Pro 的关键差异

| 方面 | MustDrop | STAR-Pro |
|------|----------|----------|
| **Vision 阶段剪枝** | ✅ Layer 0 合并 + Layer 23 Key Set | ❌ 无 |
| **剪枝位置** | 固定层 [2,6,10,14] | 可配置 |
| **剪枝策略** | Dual Attention Filter (global + individual) | Relevance + Diversity Score |
| **保护机制** | Key Token Set (8% by CLS attn) | 无显式保护 |
| **KV Cache 处理** | 渐进式稀疏化（不同层不同） | 统一处理 |
| **Attention 来源** | text→image attention | text→image attention |
| **自适应程度** | threshold-based adaptive | score-based ranking |

---

## 九、复现注意事项

1. **Vision Encoder 输出格式变化**：
   - `attentions` 字段被复用为 `key_set` 索引
   - 需要修改 `CLIPVisionTower.forward()` 的返回值处理

2. **LLM 需要修改的类**：
   - `LlamaModel` → `LlamaDynamicvitModel`
   - `LlamaSdpaAttention` → `LlamaDynamicvitSdpaAttention`
   - `LlamaDecoderLayer` → `LlamaDynamicvitDecoderLayer`
   - `LlamaForCausalLM` → `LlamaDynamicvitForCausalLM`

3. **接口变化**：
   - `forward()` 新增参数：`img_seq`, `key_set`, `token_length_list`, `pre_prompt_length_list`, `global_thr`, `individual_thr`
   - `generate()` 也需要相应修改

4. **只支持推理**：
   - `modelling_sparse_llama.py:208`: `assert self.training == False`
   - 不支持训练模式

---

## 十、STAR-Pro 代码库集成状态

> 更新日期：2026-01-23
> **状态：✅ 完整版 MustDrop 已集成（包含三阶段全部功能）**

### 10.1 已创建的文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `llava/model/multimodal_encoder/local_merge.py` | Token merging 算法 (conditional_pooling, merge_wavg) | ✅ 已创建 |
| `llava/model/multimodal_encoder/clip_encoder_mustdrop.py` | Vision encoder: Layer 0 merging + Layer 23 key set | ✅ **已启用** |
| `llava/model/language_model/modelling_llama_mustdrop.py` | LLM Dual Attention Filter 核心 | ✅ 已创建 |
| `llava/model/language_model/llava_llama_mustdrop.py` | LLaVA 独立集成层 | ✅ 已创建 (备用) |

### 10.2 已修改的文件

| 文件 | 修改内容 |
|------|----------|
| `llava/model/multimodal_encoder/builder.py` | 添加 `use_mustdrop` 条件，使用 `CLIPVisionTowerMustDrop` |
| `llava/model/llava_arch.py` | 添加 `mustdrop` case 在 `encode_images()`，存储 `_mustdrop_img_seq` 和 `_mustdrop_key_set` |
| `llava/model/language_model/llava_llama.py` | 设置 `config.use_mustdrop`，传递 Vision Encoder 输出到 LLM |
| `llava/eval/model_vqa_loader.py` | 添加 MustDrop 配置和调用 |

### 10.3 使用方法

```bash
# 运行 MustDrop baseline 评估
python llava/eval/model_vqa_loader.py \
    --model-path <your_model_path> \
    --pruning_method mustdrop \
    --visual_token_num 64 \
    --question-file <question_file> \
    --answers-file <output_file> \
    --image-folder <image_folder>
```

### 10.4 完整数据流

**当前实现：✅ 完整版 MustDrop（三阶段全部功能）**

```
Images [B, 3, 336, 336]
    ↓
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1: CLIPVisionTowerMustDrop (clip_encoder_mustdrop.py)        │
│   Layer 0:  conditional_pooling() → Token Merging                  │
│             576 patches → M patches (M < 576, adaptive)            │
│   Layer 23: CLS attention → Key Set Extraction                     │
│             Top 8% tokens → K key indices (~46)                    │
│   Output: (image_features [B, M, D], key_set [B, K])               │
└────────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────────┐
│ llava_arch.py: encode_images()                                     │
│   Store: self._mustdrop_img_seq = M                                │
│          self._mustdrop_key_set = key_set                          │
└────────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────────┐
│ llava_llama.py: forward() / generate()                             │
│   Set: model.img_seq = _mustdrop_img_seq                           │
│        model.key_set = [num_keys, key_indices_tensor]              │
│        model.pre_prompt_length_list = [35]                         │
│        model.token_length_list = [total_seq_len]                   │
└────────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────────┐
│ Stage 2: MustDropLlamaModel (modelling_llama_mustdrop.py)          │
│   Pruning Layers: [2, 6, 10, 14]                                   │
│   Dual Attention Filter:                                           │
│     - Global: sum(text→img attention) threshold                    │
│     - Individual: max(text→img attention) threshold                │
│   Key Set Protection:                                              │
│     - vis_indices = cat(vis_indices, key_set_indices)              │
│     - vis_indices = unique(vis_indices)                            │
└────────────────────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────────────────────┐
│ Stage 3: KV Cache Sparsification                                   │
│   - Layer 0-1: Full KV Cache (no pruning)                          │
│   - Layer 2-14: Progressive sparsification                         │
│   - Layer 15-31: Final sparse KV Cache                             │
└────────────────────────────────────────────────────────────────────┘
```

### 10.5 关键修改说明

#### builder.py - Vision Tower 选择
```python
use_mustdrop = getattr(vision_tower_cfg, 'use_mustdrop', False)
if use_mustdrop:
    from .clip_encoder_mustdrop import CLIPVisionTowerMustDrop
    return CLIPVisionTowerMustDrop(vision_tower, args=vision_tower_cfg, **kwargs)
```

#### llava_llama.py - Config 设置（关键！）
```python
# 在 super().__init__() 之前设置 config.use_mustdrop
# 这确保 build_vision_tower() 使用 CLIPVisionTowerMustDrop
if use_mustdrop:
    config.use_mustdrop = True
    config.mustdrop_config = mustdrop_config if mustdrop_config else {}
```

#### llava_arch.py - Vision Encoder 输出处理
```python
elif self.pruning_method == 'mustdrop':
    image_features, key_set = self.get_model().get_vision_tower()(images)
    B, M, C = image_features.shape
    self._mustdrop_img_seq = M  # After merging, M < 576
    self._mustdrop_key_set = key_set  # [B, K]
```

### 10.6 依赖要求

```
transformers==4.37.2  # pyproject.toml 指定版本
torch>=2.1.2
einops  # For rearrange in local_merge.py
```

⚠️ **注意**：当前 transformers 4.57.3 版本与代码库存在 API 兼容性问题（`_prepare_4d_causal_attention_mask` 已移除）。需要使用正确的环境版本运行。

### 10.7 验证清单

- [x] Vision Tower: Token merging at Layer 0 (conditional_pooling)
- [x] Vision Tower: Key set extraction at Layer 23 (top 8% by CLS attention)
- [x] Vision Tower: Returns (features, key_set) tuple
- [x] builder.py: Conditionally uses CLIPVisionTowerMustDrop
- [x] llava_arch.py: Stores _mustdrop_img_seq, _mustdrop_key_set
- [x] llava_llama.py: Sets config.use_mustdrop BEFORE model creation
- [x] llava_llama.py: Passes actual values to model.img_seq, model.key_set
- [x] MustDropLlamaModel: Dual Attention Filter at layers [2, 6, 10, 14]
- [x] MustDropLlamaModel: Key set protection (tokens never pruned)
- [x] MustDropLlamaModel: KV Cache progressive sparsification
