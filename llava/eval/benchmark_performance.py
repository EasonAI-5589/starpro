"""
Unified Performance Benchmarking Script for STAR-LLaVA

This script provides comprehensive performance testing including:
- Prefill time and Decode time
- GPU Memory usage (current and peak)
- FLOPs calculation (based on layer-wise visual tokens)
- Throughput (tokens/s, images/s)
- End-to-end latency

Usage:
    python llava/eval/benchmark_performance.py \
        --model-path /path/to/model \
        --question-file /path/to/questions.jsonl \
        --image-folder /path/to/images \
        --answers-file /path/to/output.jsonl \
        --pruning_method star_v3 \
        --visual_token_num 128 \
        --output-json performance_report.json
"""

import argparse
import torch
import os
import json
import time
from tqdm import tqdm
from datetime import datetime

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


class PerformanceTracker:
    """Track and compute performance metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_samples = 0
        self.total_tokens_generated = 0
        self.total_end_to_end_time = 0.0
        self.gpu_memory_samples = []
        self.visual_token_samples = []

    def add_sample(self, num_tokens, end_to_end_time, gpu_memory, visual_tokens):
        """Add metrics for one sample"""
        self.total_samples += 1
        self.total_tokens_generated += num_tokens
        self.total_end_to_end_time += end_to_end_time
        self.gpu_memory_samples.append(gpu_memory)
        self.visual_token_samples.append(visual_tokens)

    def get_throughput(self):
        """Calculate throughput metrics"""
        if self.total_end_to_end_time == 0:
            return {
                "tokens_per_second": 0.0,
                "images_per_second": 0.0,
                "seconds_per_image": 0.0
            }

        return {
            "tokens_per_second": self.total_tokens_generated / self.total_end_to_end_time,
            "images_per_second": self.total_samples / self.total_end_to_end_time,
            "seconds_per_image": self.total_end_to_end_time / self.total_samples
        }

    def get_gpu_memory_stats(self):
        """Get GPU memory statistics"""
        if not self.gpu_memory_samples:
            return {"avg_gb": 0.0, "max_gb": 0.0, "min_gb": 0.0}

        return {
            "avg_gb": sum(self.gpu_memory_samples) / len(self.gpu_memory_samples),
            "max_gb": max(self.gpu_memory_samples),
            "min_gb": min(self.gpu_memory_samples)
        }

    def get_visual_token_stats(self):
        """Get visual token statistics"""
        if not self.visual_token_samples:
            return {"avg": 0.0, "max": 0, "min": 0}

        return {
            "avg": sum(self.visual_token_samples) / len(self.visual_token_samples),
            "max": max(self.visual_token_samples),
            "min": min(self.visual_token_samples)
        }


def benchmark_performance(args):
    """Main benchmarking function"""

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Method Configuration
    use_fastv = True if args.pruning_method == "fastv" else False
    fastv_config = {"K": 2, "T": args.visual_token_num}

    use_sparsevlm = True if args.pruning_method == "sparsevlm" else False
    sparsevlm_config = {"T": args.visual_token_num}

    use_pdrop = True if args.pruning_method == "pdrop" else False
    pdrop_config = {"T": args.visual_token_num}

    use_star = True if args.pruning_method in ["star", "star_v2", "star_v3"] else False
    star_config = {
        "T": args.visual_token_num,
        "num_latent": args.num_latent,
        "latent_pool_size": (args.latent_pool_h, args.latent_pool_w),
        "mode": "star_v3" if args.pruning_method == "star_v3" else ("star_v2" if args.pruning_method == "star_v2" else "star"),
        "debug": False,  # Disable debug prints for benchmarking
    }

    use_text_tower = True if args.pruning_method == "trim" or "cdp3" in args.pruning_method or "thcp" in args.pruning_method or args.pruning_method == "star_v3" else False

    print(f"\n{'='*70}")
    print(f"Performance Benchmarking for STAR-LLaVA")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Pruning method: {args.pruning_method}")
    print(f"Visual token budget: {args.visual_token_num}")
    print(f"Warmup samples: {args.warmup_samples}")
    print(f"Benchmark samples: {args.num_samples}")
    print(f"{'='*70}\n")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        pruning_method=args.pruning_method,
        visual_token_num=args.visual_token_num,
        use_fastv=use_fastv, fastv_config=fastv_config,
        use_sparsevlm=use_sparsevlm, sparsevlm_config=sparsevlm_config,
        use_pdrop=use_pdrop, pdrop_config=pdrop_config,
        use_star=use_star, star_config=star_config,
        use_text_tower=use_text_tower,
    )

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # Limit to specified number of samples
    if args.num_samples > 0:
        questions = questions[:args.warmup_samples + args.num_samples]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching to {args.conv_mode} for plain model.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    # Performance tracking
    tracker = PerformanceTracker()

    data_bar = tqdm(zip(data_loader, questions), total=len(questions), desc="Benchmarking")
    data_num = 0

    for (input_ids, image_tensors, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]
        question = cur_prompt.replace("\nAnswer the question using a single word or phrase.", "")

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        # Start tracking after warmup
        if data_num == args.warmup_samples:
            model.start_latency = True
            model.track_flops = True
            torch.cuda.reset_peak_memory_stats()
            print(f"\n[Warmup complete] Starting performance measurement...\n")

        with torch.inference_mode():
            start_time = time.time()

            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                texts=question,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

            end_time = time.time()

            # Get visual token num from model if available
            if hasattr(model.model, 'visual_token_num'):
                visual_token_num = model.model.visual_token_num

            # Track metrics after warmup
            if data_num >= args.warmup_samples:
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                num_tokens = output_ids.shape[1]
                end_to_end_time = end_time - start_time

                tracker.add_sample(num_tokens, end_to_end_time, gpu_memory_gb, visual_token_num)

                # Update progress bar
                throughput = tracker.get_throughput()
                data_bar.set_postfix(
                    vtn=f"{visual_token_num}",
                    gpu=f"{gpu_memory_gb:.2f}GB",
                    img_s=f"{throughput['images_per_second']:.2f}img/s"
                )

            data_num += 1

        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Save results
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()

    # Compute final metrics
    print(f"\n{'='*70}")
    print(f"Performance Benchmarking Results")
    print(f"{'='*70}")

    # Latency metrics
    prefill_latency_ms = model.prefill_latency / tracker.total_samples if tracker.total_samples > 0 else 0
    decode_latency_ms = model.decode_latency / tracker.total_samples if tracker.total_samples > 0 else 0
    total_latency_ms = prefill_latency_ms + decode_latency_ms

    print(f"\n[Latency Metrics]")
    print(f"  Prefill latency (avg): {prefill_latency_ms:.2f} ms")
    print(f"  Decode latency (avg): {decode_latency_ms:.2f} ms")
    print(f"  Total latency (avg): {total_latency_ms:.2f} ms")

    # GPU Memory
    gpu_stats = tracker.get_gpu_memory_stats()
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n[GPU Memory]")
    print(f"  Average: {gpu_stats['avg_gb']:.2f} GB")
    print(f"  Peak: {peak_memory_gb:.2f} GB")
    print(f"  Min: {gpu_stats['min_gb']:.2f} GB")

    # FLOPs
    avg_flops = 0.0
    if model.flops_count > 0:
        avg_flops = model.total_flops / model.flops_count
        print(f"\n[FLOPs]")
        print(f"  Average per sample: {avg_flops:.2f} TFLOPs")
        if hasattr(model, 'layer_visual_tokens') and model.layer_visual_tokens:
            print(f"  Layer visual tokens: {model.layer_visual_tokens}")

    # Throughput
    throughput = tracker.get_throughput()
    print(f"\n[Throughput]")
    print(f"  Tokens/second: {throughput['tokens_per_second']:.2f}")
    print(f"  Images/second: {throughput['images_per_second']:.2f}")
    print(f"  Seconds/image: {throughput['seconds_per_image']:.2f}")

    # Visual tokens
    vt_stats = tracker.get_visual_token_stats()
    print(f"\n[Visual Tokens]")
    print(f"  Average: {vt_stats['avg']:.2f}")
    print(f"  Max: {vt_stats['max']}")
    print(f"  Min: {vt_stats['min']}")

    print(f"\n{'='*70}\n")

    # Save to JSON if requested
    if args.output_json:
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "name": model_name,
                "path": model_path,
                "pruning_method": args.pruning_method,
                "visual_token_budget": args.visual_token_num
            },
            "benchmark_config": {
                "warmup_samples": args.warmup_samples,
                "benchmark_samples": tracker.total_samples,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens
            },
            "latency": {
                "prefill_ms": float(prefill_latency_ms),
                "decode_ms": float(decode_latency_ms),
                "total_ms": float(total_latency_ms)
            },
            "gpu_memory": {
                "avg_gb": float(gpu_stats['avg_gb']),
                "peak_gb": float(peak_memory_gb),
                "min_gb": float(gpu_stats['min_gb'])
            },
            "flops": {
                "avg_tflops": float(avg_flops),
                "layer_visual_tokens": model.layer_visual_tokens if hasattr(model, 'layer_visual_tokens') else []
            },
            "throughput": {
                "tokens_per_second": float(throughput['tokens_per_second']),
                "images_per_second": float(throughput['images_per_second']),
                "seconds_per_image": float(throughput['seconds_per_image'])
            },
            "visual_tokens": {
                "avg": float(vt_stats['avg']),
                "max": int(vt_stats['max']),
                "min": int(vt_stats['min'])
            }
        }

        output_json_path = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance benchmarking for STAR-LLaVA")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    # Data arguments
    parser.add_argument("--image-folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--question-file", type=str, required=True, help="Path to question JSONL file")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to output answers file")

    # Pruning arguments
    parser.add_argument("--pruning_method", type=str, default=None,
                        help="Pruning method: fastv, sparsevlm, pdrop, star, star_v2, star_v3")
    parser.add_argument("--visual_token_num", type=int, default=576,
                        help="Target number of visual tokens")

    # STAR specific arguments
    parser.add_argument("--num_latent", type=int, default=20,
                        help="Number of latent tokens for STAR")
    parser.add_argument("--latent_pool_h", type=int, default=5,
                        help="Height of latent pooling grid for STAR")
    parser.add_argument("--latent_pool_w", type=int, default=4,
                        help="Width of latent pooling grid for STAR")

    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Benchmark arguments
    parser.add_argument("--warmup_samples", type=int, default=50,
                        help="Number of warmup samples before measurement")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to benchmark (-1 for all)")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    # Output arguments
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save JSON performance report")

    args = parser.parse_args()
    benchmark_performance(args)
