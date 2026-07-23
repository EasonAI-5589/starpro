# Bot1 实验监控协调文件

## 协调规则
- **1号机职责**：监控结果 → 整理数据 → 写入 Overleaf (X_suppl.tex 新 section)
- **4号机职责**：跑实验 → 实验完成后在下方"已完成实验"里记一条

## 4号机：实验完成后请填写
格式：`[完成时间] [模型] [方法] [task] [vtn] [主要指标=分数]`

### 已完成实验
（4号机填写）

## Overleaf 写入规则
- 新开 Section（不改 Section D）
- 标题类似："Latest Ablation Study Results"
- 包含：模型、方法、token budget、各 benchmark 分数

### 已完成实验
[2026-02-27 02:45] [LLaVA-1.5-7b] [star_pro D-αR] [textvqa] [128T] [α=2.0: 56.77%]
[2026-02-27 02:45] [LLaVA-1.5-7b] [star_pro D-αR] [textvqa] [64T] [α=0.5: 55.25%, α=1.0: 55.07%, α=2.0: 54.51%]
[2026-02-27 02:45] [LLaVA-1.5-7b] [star_pro D-αR] [textvqa] [32T] [α=0.5: 53.86%, α=1.0: 53.71%, α=2.0: 52.55%]

## SQA λ ablation 완료 (8号机, 2025)
[완료] llava-1.5-7b star_pro sqa lambda=0.5 vtn=128 IMG-Accuracy=67.72%
[완료] llava-1.5-7b star_pro sqa lambda=0.5 vtn=64  IMG-Accuracy=67.48%
[완료] llava-1.5-7b star_pro sqa lambda=0.5 vtn=32  IMG-Accuracy=68.12%
[완료] llava-1.5-7b star_pro sqa lambda=1.0 vtn=128 IMG-Accuracy=67.28%
[완료] llava-1.5-7b star_pro sqa lambda=1.0 vtn=64  IMG-Accuracy=67.87%
[완료] llava-1.5-7b star_pro sqa lambda=1.0 vtn=32  IMG-Accuracy=68.62%
[완료] llava-1.5-7b star_pro sqa lambda=2.0 vtn=128 IMG-Accuracy=67.48%
[완료] llava-1.5-7b star_pro sqa lambda=2.0 vtn=64  IMG-Accuracy=68.52%
[완료] llava-1.5-7b star_pro sqa lambda=2.0 vtn=32  IMG-Accuracy=68.86%
Overleaf D.4 updated → commit 851a387 (5 benchmarks 포함)
