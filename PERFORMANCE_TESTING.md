# STAR-LLaVA Performance Testing Guide

本文档介绍如何测试 STAR-LLaVA 模型的各项性能指标。

## 支持的性能指标

✅ **Prefill Time** - 预填充阶段的延迟（处理图像和prompt的时间）
✅ **Decode Time** - 解码阶段的延迟（逐token生成的时间）
✅ **GPU Memory** - GPU显存使用（当前使用量和峰值）
✅ **FLOPs** - 浮点运算次数（基于每层视觉token数动态计算）
✅ **Latency** - 总延迟（Prefill + Decode）
✅ **Throughput** - 吞吐量（tokens/s, images/s）
✅ **End-to-End Time** - 端到端推理时间

## 测试方法

### 方法 1: 完整性能基准测试（推荐）

使用 `benchmark_performance.py` 进行全面的性能测试，包含所有指标并支持JSON输出。

```bash
python llava/eval/benchmark_performance.py \
    --model-path /path/to/model \
    --question-file /path/to/pope/questions.jsonl \
    --image-folder /path/to/coco/val2014 \
    --answers-file ./outputs/pope_answers.jsonl \
    --pruning_method star_pro \
    --visual_token_num 128 \
    --warmup_samples 50 \
    --num_samples 1000 \
    --output-json ./performance_report.json
```

**参数说明：**
- `--warmup_samples`: 预热样本数，用于消除冷启动影响（默认50）
- `--num_samples`: 测试样本数，-1表示全部（默认-1）
- `--output-json`: 输出JSON格式的性能报告（可选）

**输出示例：**
```
======================================================================
Performance Benchmarking Results
======================================================================

[Latency Metrics]
  Prefill latency (avg): 45.23 ms
  Decode latency (avg): 123.45 ms
  Total latency (avg): 168.68 ms

[GPU Memory]
  Average: 12.34 GB
  Peak: 13.56 GB
  Min: 11.98 GB

[FLOPs]
  Average per sample: 2.45 TFLOPs
  Layer visual tokens: [256, 256, 192, 192, 128, 128, ...]

[Throughput]
  Tokens/second: 234.56
  Images/second: 5.93
  Seconds/image: 0.17

[Visual Tokens]
  Average: 128.0
  Max: 128
  Min: 128
```

### 方法 2: 标准评估（带性能测试）

使用 `model_vqa_loader.py` 在评估任务中同时测量性能指标。

```bash
python llava/eval/model_vqa_loader.py \
    --model-path /path/to/model \
    --question-file /path/to/pope/questions.jsonl \
    --image-folder /path/to/coco/val2014 \
    --answers-file ./outputs/pope_answers.jsonl \
    --pruning_method star_pro \
    --visual_token_num 128
```

**特点：**
- 同时生成评估结果和性能指标
- 自动在第50个样本后开始性能测量（warmup）
- 输出所有性能指标到终端

### 方法 3: 快速性能测试

使用 `model_vqa_loader_accelerate.py` 进行快速性能测试（不保存评估结果）。

```bash
python llava/eval/model_vqa_loader_accelerate.py \
    --model-path /path/to/model \
    --question-file /path/to/pope/questions.jsonl \
    --image-folder /path/to/coco/val2014 \
    --answers-file /dev/null \
    --pruning_method star_pro \
    --visual_token_num 128
```

**特点：**
- 禁用文件I/O，专注于性能测量
- 固定测试1000个样本（50 warmup + 1000 measurement）
- 执行速度更快

## POPE 数据集上的测试

POPE（Polling-based Object Probing Evaluation）是常用的视觉幻觉评估数据集，也是性能测试的标准选择。

### 数据准备

```bash
# POPE 问题文件路径
QUESTION_FILE=playground/data/eval/pope/llava_pope_test.jsonl

# COCO 图片路径
IMAGE_FOLDER=playground/data/eval/pope/val2014
```

### 完整示例

```bash
# 测试 STAR-PRO (128 tokens)
python llava/eval/benchmark_performance.py \
    --model-path liuhaotian/llava-v1.6-vicuna-13b \
    --question-file playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder playground/data/eval/pope/val2014 \
    --answers-file ./outputs/pope_star_pro_128.jsonl \
    --pruning_method star_pro \
    --visual_token_num 128 \
    --num_latent 20 \
    --latent_pool_h 5 \
    --latent_pool_w 4 \
    --warmup_samples 50 \
    --num_samples 1000 \
    --output-json ./performance_star_pro_128.json
```

## FLOPs 计算说明

FLOPs 使用以下公式动态计算（基于每层实际的视觉token数）：

```
每层 FLOPs = 8 × n × d² + 4 × n² × d + 6 × n × d × m
```

其中：
- `n`: 该层的视觉token数量
- `d`: 隐藏维度（hidden_size）
- `m`: FFN中间维度（intermediate_size）

对于 STAR 模型，每层的 `n` 会根据 pruning schedule 动态变化，系统会自动记录并计算。

## 性能对比实验

### 测试不同 token budgets

```bash
# STAR-PRO with different budgets
for TOKENS in 32 64 128 192; do
    python llava/eval/benchmark_performance.py \
        --model-path /path/to/model \
        --question-file playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder playground/data/eval/pope/val2014 \
        --answers-file ./outputs/star_pro_${TOKENS}.jsonl \
        --pruning_method star_pro \
        --visual_token_num ${TOKENS} \
        --num_samples 1000 \
        --output-json ./perf_star_pro_${TOKENS}.json
done
```

### 测试不同方法

```bash
# 对比 STAR, STAR-V2, STAR-PRO
for METHOD in star star_v2 star_pro; do
    python llava/eval/benchmark_performance.py \
        --model-path /path/to/model \
        --question-file playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder playground/data/eval/pope/val2014 \
        --answers-file ./outputs/${METHOD}_128.jsonl \
        --pruning_method ${METHOD} \
        --visual_token_num 128 \
        --num_samples 1000 \
        --output-json ./perf_${METHOD}_128.json
done
```

## JSON 报告格式

使用 `--output-json` 参数会生成结构化的性能报告：

```json
{
  "timestamp": "2025-11-09T10:30:00",
  "model": {
    "name": "llava-v1.6-vicuna-13b",
    "path": "/path/to/model",
    "pruning_method": "star_pro",
    "visual_token_budget": 128
  },
  "benchmark_config": {
    "warmup_samples": 50,
    "benchmark_samples": 1000,
    "temperature": 0.2,
    "max_new_tokens": 128
  },
  "latency": {
    "prefill_ms": 45.23,
    "decode_ms": 123.45,
    "total_ms": 168.68
  },
  "gpu_memory": {
    "avg_gb": 12.34,
    "peak_gb": 13.56,
    "min_gb": 11.98
  },
  "flops": {
    "avg_tflops": 2.45,
    "layer_visual_tokens": [256, 256, 192, ...]
  },
  "throughput": {
    "tokens_per_second": 234.56,
    "images_per_second": 5.93,
    "seconds_per_image": 0.17
  },
  "visual_tokens": {
    "avg": 128.0,
    "max": 128,
    "min": 128
  }
}
```

## 注意事项

1. **Warmup**: 前50个样本用于预热，性能指标从第51个样本开始统计
2. **GPU Memory**: 在第50个样本时会重置峰值显存统计
3. **Batch Size**: 所有测试脚本强制使用 `batch_size=1`
4. **CUDA Events**: 延迟测量使用 CUDA Events 确保精确计时
5. **Debug 模式**: benchmark 时建议关闭 debug 模式（`"debug": False`）以避免打印干扰

## 实现细节

### 修改的文件

1. **llava/model/language_model/llava_llama.py**
   - 添加 FLOPS 追踪功能
   - 实现 `calculate_flops()` 和 `record_layer_tokens()` 方法

2. **llava/model/language_model/modelling_llama_star.py**
   - 在 forward 过程中记录每层的视觉token数量
   - 支持 FLOPS 自动计算

3. **llava/eval/model_vqa_loader.py**
   - 添加端到端时间测量
   - 添加吞吐量统计
   - 增强性能指标输出

4. **llava/eval/model_vqa_loader_accelerate.py**
   - 清理残留代码
   - 添加完整的性能测量支持

5. **llava/eval/benchmark_performance.py** (新建)
   - 专门的性能基准测试脚本
   - 支持 JSON 格式输出

## 相关工具

- **cal_flops.py**: 基于配置的理论 FLOPS 计算（手动配置）
- **eval.sh**: 统一的评估脚本入口
- **scripts/v1_6/*/pope.sh**: POPE 评估的便捷脚本

## 故障排除

### Q: 显示 "No performance data collected"
A: 确保数据集至少有 51 个样本（50 warmup + 1 测试）

### Q: FLOPS 显示为 0
A: 检查是否使用了 STAR 模型，其他模型可能不支持自动 FLOPS 计算

### Q: 吞吐量很低
A: 检查是否有其他进程占用 GPU，或考虑使用更小的 max_new_tokens

## 参考

- STAR-LLaVA 论文: [链接]
- POPE 数据集: https://github.com/AoiDragon/POPE
- LLaVA 项目: https://github.com/haotian-liu/LLaVA
