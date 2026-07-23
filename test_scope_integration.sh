#!/bin/bash

# SCOPE 集成快速验证脚本

echo "=========================================="
echo "SCOPE 集成验证测试"
echo "=========================================="

# 激活环境
source /mnt/world_foundational_model/gyc/miniconda3/etc/profile.d/conda.sh
conda activate llava

echo ""
echo "1. 测试 scope_utils 导入..."
python -c "
from llava.model.language_model.scope_utils import scope_select, scope_select_with_env
import torch
B, N, D = 2, 576, 1024
features = torch.randn(B, N, D)
cls_attn = torch.rand(B, N)
idx, _ = scope_select(features, 64, cls_attn)
print(f'   ✓ scope_select OK: {idx.shape}')
"

echo ""
echo "2. 测试与官方实现一致性..."
python << 'EOF'
import os
import torch
from llava.model.language_model.scope_utils import scope_select

# 官方实现
def official_scope(visual_feature_vectors, num_selected_token, cls_attn=None):
    norm_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-1, keepdim=True)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    B, N = visual_feature_vectors.shape[:2]
    device = visual_feature_vectors.device
    dtype = visual_feature_vectors.dtype
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_idx = torch.empty(B, num_selected_token, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, dtype=dtype, device=device)
    alpha = float(os.environ.get('ALPHA', '1.0'))
    if cls_attn is not None:
        cls_attn_powered = cls_attn ** alpha
    else:
        cls_attn_powered = torch.ones(B, N, dtype=dtype, device=device)
    for i in range(num_selected_token):
        unselected_mask = ~selected
        gains = torch.maximum(
            torch.zeros(1, dtype=dtype, device=device),
            cosine_simi.masked_fill(~unselected_mask.unsqueeze(1), 0) - cur_max.unsqueeze(2)
        ).sum(dim=1)
        combined = os.environ.get('COMBINED', 'multi')
        if combined == 'multi':
            gains = gains * cls_attn_powered
        elif combined == 'add':
            gains = gains + cls_attn_powered
        gains = gains.masked_fill(~unselected_mask, float('-inf'))
        best_idx = gains.argmax(dim=1)
        selected[torch.arange(B, device=device), best_idx] = True
        selected_idx[:, i] = best_idx
        cur_max = torch.maximum(cur_max, cosine_simi[torch.arange(B, device=device), best_idx])
    return selected_idx, cosine_simi

# 测试
B, N, D = 2, 576, 1024
features = torch.randn(B, N, D)
cls_attn = torch.rand(B, N)
official_idx, _ = official_scope(features, 64, cls_attn)
my_idx, _ = scope_select(features, 64, cls_attn)

if torch.equal(official_idx, my_idx):
    print("   ✓ 100% 一致！")
else:
    print("   ✗ 不一致")
EOF

echo ""
echo "3. 检查文件修改..."
if grep -q "pruning_method == 'scope'" llava/model/llava_arch.py; then
    echo "   ✓ llava_arch.py 已添加 scope 分支"
else
    echo "   ✗ llava_arch.py 未找到 scope 分支"
fi

if grep -q "scope 64" eval.sh; then
    echo "   ✓ eval.sh 已添加运行示例"
else
    echo "   ✗ eval.sh 未找到运行示例"
fi

echo ""
echo "=========================================="
echo "验证完成！"
echo "=========================================="
echo ""
echo "下一步: 运行基准测试"
echo "  export CKPT_DIR=/mnt/world_foundational_model/gyc/models"
echo "  export DATA_DIR=/mnt/world_foundational_model/gyc/LLaVA-Eval"
echo "  export RESULT_DIR=./results"
echo ""
echo "  # 测试 SCOPE"
echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/mme.sh scope 64"
echo ""
