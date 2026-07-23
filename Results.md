# STAR-Pro-LLaVA 实验结果记录

## 环境配置

```bash
export CKPT_DIR=/mnt/world_foundational_model/pd_ckp
export DATA_DIR=/mnt/world_foundational_model/pd_data/LLaVA-Eval
export RESULT_DIR=/mnt/world_foundational_model/gyc/STAR-Pro-LLaVA/results
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
```

---

## 实验脚本

### 1. Baseline 批量运行脚本

#### 单方法多配置运行（推荐）

```bash
# 定义配置数组
tasks=(mme pope textvqa gqa mmbench mmbench_cn)
method=fastv                  # 要运行的方法
tokens=(128 64 32)           # 要测试的 token 配置

# 一个方法跑完所有配置
for token in "${tokens[@]}"; do
    echo "=== Running $method with $token tokens ==="
    for task in "${tasks[@]}"; do
        echo "  → Task: $task"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
    done
done
```

#### 批量运行多个方法

```bash
# 定义配置数组
tasks=(mme pope textvqa gqa mmbench mmbench_cn)

# Vanilla: 只测试 576 tokens
method=vanilla
tokens=(576)
for token in "${tokens[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $method on $task with $token tokens..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
    done
done

# FastV: 测试多个 token 配置
method=fastv
tokens=(128 64 32)
for token in "${tokens[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $method on $task with $token tokens..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
    done
done

# DivPrune: 测试多个 token 配置
method=divprune
tokens=(128 64 32)
for token in "${tokens[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $method on $task with $token tokens..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
    done
done
```

#### 快速测试（只跑核心任务）

```bash
# 核心任务：MME + POPE
tasks=(mme pope)

# 方法 1: 测试单个方法的多个配置
method=fastv
tokens=(128 64 32)
for token in "${tokens[@]}"; do
    echo "=== Running $method with $token tokens ==="
    for task in "${tasks[@]}"; do
        echo "  → Task: $task"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
    done
done

# 方法 2: 快速对比多个方法（各用默认配置）
for method in vanilla divprune fastv; do
    token=128
    [ "$method" = "vanilla" ] && token=576
    echo "=== Running $method with $token tokens ==="
    for task in "${tasks[@]}"; do
        echo "  → Task: $task"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
    done
done
```

### 2. 单个任务运行

#### MME 评测

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh vanilla 576
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh divprune 128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh fastv 128
```

#### POPE 评测

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh vanilla 576
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh mustdrop 128
```

#### TextVQA 评测

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh vanilla 576
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh mustdrop 128
```

### 3. 多 Token 配置评测

#### 测试不同 token 数量的影响

```bash
method=mustdrop
task=mme

for token in 32 64 128 256; do
    echo "Running $method on $task with $token tokens..."
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/$task.sh $method $token
done
```

---

## 实验结果

### Table 1. 多阶段剪枝方法对比 (LLaVA-1.5-7B)

#### Retain 128 Tokens

| Method                    | VQA^v2         | GQA            | VizWiz         | SQA^I          | VQA^T          | POPE           | MME              | MMB^EN         | MMB^CN         | MMVet          | Avg.           | Rel.            |
| ------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | ---------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| MustDrop (arXiv'24)       | -              | 56.2           | -              | 68.9           | 56.1           | 78.7           | 1391.6           | 61.4           | 55.2           | 31.1           | -              | -               |
| VScan (arXiv'24)          | -              | 59.1           | -              | -              | 57.2           | 85.1           | 1441.8           | 62.3           | -              | -              | -              | -               |
| **STAR-Pro (Ours)** | **76.9** | **60.4** | **50.1** | **67.3** | **56.7** | **87.3** | **1444.0** | **62.9** | **55.2** | **32.8** | **62.2** | **98.4%** |

#### Retain 64 Tokens

| Method                    | VQA^v2         | GQA            | VizWiz         | SQA^I          | VQA^T          | POPE           | MME              | MMB^EN         | MMB^CN         | MMVet          | Avg.           | Rel.            |
| ------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | ---------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| MustDrop (arXiv'24)       | -              | 52.7           | -              | 68.91          | 54.5           | 68.0           | 1237.6           | 61.4           | 53.6           | 28.5           | -              | -               |
| VScan (arXiv'24)          | -              | 57.9           | -              | -              | 56.2           | 85.0           | 1381.5           | 62.1           | -              | -              | -              | -               |
| **STAR-Pro (Ours)** | **75.1** | **58.5** | **51.2** | **67.9** | **55.1** | **87.1** | **1392.9** | **62.0** | **53.1** | **32.1** | **61.2** | **96.8%** |

---

### MME

| 方法     | Token数 | 得分 | 日志文件                                         |
| -------- | ------- | ---- | ------------------------------------------------ |
| DivPrune | 128     | -    | `results/mme_scripts_v1_5_divprune_vtn128.log` |
| FastV    | 128     | -    | `results/mme_scripts_v1_5_fastv_vtn128.log`    |
| Vanilla  | 576     | -    | `results/mme_scripts_v1_5_vanilla_vtn576.log`  |
| STAR-Pro | 128     | -    | `results/mme_scripts_v1_5_star_pro_vtn128.log` |
| STAR-Pro | 64      | -    | `results/mme_scripts_v1_5_star_pro_vtn64.log`  |
| STAR-Pro | 32      | -    | `results/mme_scripts_v1_5_star_pro_vtn32.log`  |

### GQA

| 方法     | Token数 | Accuracy | IMG-Accuracy | 日志文件 |
| -------- | ------- | -------- | ------------ | -------- |
| MustDrop | 128     | 32.75%   | 68.86%       | -        |

### SQA (Science QA)

| 方法     | Token数 | Accuracy | IMG-Accuracy | 日志文件 |
| -------- | ------- | -------- | ------------ | -------- |
| MustDrop | 128     | -        | -            | -        |

### TextVQA

| 方法     | Token数 | 得分 | 日志文件                                             |
| -------- | ------- | ---- | ---------------------------------------------------- |
| MustDrop | 128     | -    | `results/textvqa_scripts_v1_5_mustdrop_vtn128.log` |
| MustDrop | 64      | -    | `results/textvqa_scripts_v1_5_mustdrop_vtn64.log`  |

### Ablation Study

| 配置                       | 数据集  | 得分 | 日志文件                                                        |
| -------------------------- | ------- | ---- | --------------------------------------------------------------- |
| S2 only T128 (front-heavy) | GQA     | -    | `results/ablation_star_v5_s2only_T128_gqa_frontheavy.txt`     |
| S2 only T128               | MME     | -    | `results/ablation_star_v5_s2only_T128_mme.txt`                |
| S2 only T128               | POPE    | -    | `results/ablation_star_v5_s2only_T128_pope.txt`               |
| S2 only T128 (front-heavy) | POPE    | -    | `results/ablation_star_v5_s2only_T128_pope_frontheavy.txt`    |
| S2 only T128 (front-heavy) | TextVQA | -    | `results/ablation_star_v5_s2only_T128_textvqa_frontheavy.txt` |

---

## 待运行实验

- [ ] STAR-Pro 其他 token 配置
- [ ] 更多基线方法对比
- [ ] 完整 benchmark 评测

---

## 备注

- 使用 8 卡 GPU (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
- 代理端口：192.168.32.28:18000
- 结果保存路径：`/mnt/world_foundational_model/gyc/STAR-Pro-LLaVA/results`
