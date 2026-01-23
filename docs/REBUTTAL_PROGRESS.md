# STAR-Pro Rebuttal 进度追踪

> 论文 Rebuttal 所需的实验和代码工作进度

---

## ✅ 已完成

### 1. MustDrop Baseline 复现 (2025-01-24)
**状态**: ✅ 完成并 push

**工作内容**:
- 从官方 MustDrop 仓库完整复现实现
- Vision Encoder: Token Merge + Key Set 提取
- LLM: Dual Attention Filter 剪枝
- 所有核心算法与官方 **100% 一致**

**相关文件**:
- `llava/model/multimodal_encoder/clip_encoder_mustdrop.py`
- `llava/model/multimodal_encoder/local_merge.py`
- `llava/model/language_model/llava_llama_mustdrop.py`
- `llava/model/language_model/modelling_llama_mustdrop.py`
- `docs/MustDrop_Code_Review_Handoff.md`
- `docs/MustDrop_Technical_Analysis.md`

**Commit**: `7421a5c`

---

## 🔄 进行中

(暂无)

---

## 📋 待完成

### 2. MustDrop Benchmark 测试
**优先级**: 高
**说明**: 在 VQA 任务上运行 MustDrop baseline，收集精度数据用于对比

**测试命令**:
```bash
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /path/to/test_questions.jsonl \
    --image-folder /path/to/images \
    --answers-file /tmp/mustdrop_test.jsonl \
    --pruning_method mustdrop \
    --visual_token_num 64
```

### 3. 其他 Baseline 复现
- [ ] FastV
- [ ] SparseVLM
- [ ] PDrop
- [ ] (根据审稿意见补充)

### 4. 消融实验补充
- [ ] Stage 1 Lambda 消融 (已有脚本: `STAGE1_LAMBDA_ABLATION.md`)
- [ ] Stage 2 Pruning Schedule 消融 (已有脚本: `STAGE2_PRUNING_SCHEDULE_ABLATION.md`)
- [ ] Stage 2 Text Aggregation 消融 (已有脚本: `STAGE2_TEXT_AGGREGATION_ABLATION.md`)

### 5. 审稿意见回复
- [ ] 整理审稿意见要点
- [ ] 撰写 rebuttal 回复

---

## 📊 实验数据汇总

| Method | Visual Tokens | VQAv2 | GQA | TextVQA | POPE | 备注 |
|--------|---------------|-------|-----|---------|------|------|
| LLaVA-1.5 | 576 | - | - | - | - | Baseline (无剪枝) |
| STAR-Pro | 64 | - | - | - | - | 待填充 |
| MustDrop | 64 | - | - | - | - | 待测试 |
| FastV | 64 | - | - | - | - | 待测试 |

---

*最后更新: 2025-01-24*
