"""
Visualize PrefixVLM_2 progressive token pruning.

For each sample, runs inference and saves one image per pruning stage,
showing which original visual tokens are retained at each step.

Usage:
    export STAGE1_KEEP=320
    export ALPHA_1=0.5
    export LAMBDA_1=0.4
    export ALPHA_2=0.5
    export LAMBDA_2=0.4
    export COVERAGE=SCOPE

    python visualize_prefixvlm2.py \
        --model_path /mnt/eason_ckp/models/llava-v1.6-vicuna-7b \
        --token_budget 32 \
        --samples_file samples.json \
        --output_dir /mnt/eason/LLaVA-STAR-Pro2/vis_output

samples.json format (list of dicts):
    [{"image": "/path/to/img.jpg", "question": "Is this ...", "name": "landmark_0"}, ...]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                             tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

WHITE_BLEND = 0.78  # pruned patches: blend original toward white (0=original, 1=pure white)


def load_model(model_path: str, token_budget: int):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name,
        pruning_method="prefixvlm_2",
        visual_token_num=token_budget,
        use_prefixvlm_2=True,
        prefixvlm_2_config={"T": token_budget},
        device_map="auto",
    )
    model.eval()
    return tokenizer, model, image_processor


@torch.inference_mode()
def run_inference(image_path: str, question: str, tokenizer, model, image_processor):
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.float16, device=model.device)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    out_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        max_new_tokens=32,
        use_cache=True,
    )
    if isinstance(out_ids, tuple):
        out_ids = out_ids[0]
    answer = tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    pruning_history = getattr(model.model, "pruning_history", [])
    stage1_indices = getattr(model, "stage1_visual_indices", None)
    total_tokens = getattr(model, "stage1_total_tokens", None)

    # 将 LLM 内部的相对索引（相对 STAGE1_KEEP）映射回原始 N 个 token 空间
    if stage1_indices is not None:
        mapped_history = [stage1_indices[h] for h in pruning_history]
    else:
        mapped_history = pruning_history

    return image, stage1_indices, mapped_history, total_tokens, answer


def infer_grid(n_tokens: int, img_w: int, img_h: int):
    """Infer (grid_h, grid_w) closest to image aspect ratio for n_tokens patches."""
    aspect = img_w / img_h
    grid_h = max(1, round(math.sqrt(n_tokens / aspect)))
    grid_w = max(1, round(n_tokens / grid_h))
    # adjust until product >= n_tokens
    while grid_h * grid_w < n_tokens:
        grid_w += 1
    return grid_h, grid_w


def render_stage(image: Image.Image, kept_indices, total_tokens: int,
                 stage_label: str) -> Image.Image:
    """
    Render one pruning stage:
      - Pruned tokens: darkened
      - Kept tokens: original brightness + white border
      - Extra grid cells beyond total_tokens: fully black (padding)
    """
    img_w, img_h = image.size
    grid_h, grid_w = infer_grid(total_tokens, img_w, img_h)

    # mask over [0, grid_h*grid_w), only indices < total_tokens are valid
    mask = np.zeros(grid_h * grid_w, dtype=bool)
    idx = kept_indices.numpy() if isinstance(kept_indices, torch.Tensor) else np.array(kept_indices)
    valid = idx[idx < total_tokens]
    mask[valid] = True
    mask = mask.reshape(grid_h, grid_w)

    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    orig = image.convert("RGB")
    # pruned patches: wash toward white so original content is faintly visible
    washed = orig.point(lambda p: int(p * (1 - WHITE_BLEND) + 255 * WHITE_BLEND))
    result = washed.copy()
    draw = ImageDraw.Draw(result)

    for row in range(grid_h):
        for col in range(grid_w):
            linear = row * grid_w + col
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            x1 = int((col + 1) * cell_w)
            y1 = int((row + 1) * cell_h)
            if linear >= total_tokens:
                # padding cell
                draw.rectangle([x0, y0, x1, y1], fill=(245, 245, 245))
            elif mask[row, col]:
                patch = orig.crop((x0, y0, x1, y1))
                result.paste(patch, (x0, y0))
                draw.rectangle([x0, y0, x1 - 1, y1 - 1],
                                outline=(255, 255, 255), width=1)

    # label bar at top
    bar_h = 28
    canvas = Image.new("RGB", (img_w, img_h + bar_h), (30, 30, 30))
    canvas.paste(result, (0, bar_h))
    bar_draw = ImageDraw.Draw(canvas)
    bar_draw.text((6, 5), stage_label, fill=(255, 220, 60))
    return canvas


def make_combined(stages: list, sample_name: str, question: str, answer: str,
                  orig_image: Image.Image, bar_h: int = 28) -> Image.Image:
    """Horizontally concatenate original + all stage images with a title strip."""
    w, h = stages[0].size   # all stages same size (img_w, img_h + bar_h)

    # build original image panel to match stage panel size
    orig_panel = Image.new("RGB", (w, h), (30, 30, 30))
    orig_resized = orig_image.convert("RGB").resize((w, h - bar_h))
    orig_panel.paste(orig_resized, (0, bar_h))
    orig_draw = ImageDraw.Draw(orig_panel)
    orig_draw.text((6, 5), "Original", fill=(180, 255, 180))

    panels = [orig_panel] + stages
    n = len(panels)
    title_h = 44
    canvas = Image.new("RGB", (w * n, h + title_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    title = f"{sample_name}  |  Q: {question[:90]}  |  Pred: {answer}"
    draw.text((8, 12), title, fill=(200, 200, 200))

    for i, img in enumerate(panels):
        canvas.paste(img, (i * w, title_h))

    return canvas


def visualize_sample(image_path: str, question: str, name: str,
                     stage1_indices, pruning_history: list, answer: str,
                     total_tokens: int, output_dir: str):
    image = Image.open(image_path).convert("RGB")

    # 用 name 构建子目录；对 mmvet 格式(v1_60_xxx)保留前两段，其余直接用 name
    parts = name.split("_")
    sample_id = "_".join(parts[:2]) if len(parts) >= 2 and parts[0].startswith("v") else name
    sample_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    if stage1_indices is None and not pruning_history:
        print(f"  [WARN] no pruning data, skipping {name}")
        return

    # 保存原图
    orig_path = os.path.join(sample_dir, "00_original.png")
    image.save(orig_path)
    print(f"  Saved original: {orig_path}")

    # Stage 1: llava_arch 剪枝（视觉编码器后）
    if stage1_indices is not None:
        n_kept = len(stage1_indices)
        label = f"Stage 1 (arch)  kept={n_kept}/{total_tokens}"
        stage_img = render_stage(image, stage1_indices, total_tokens, label)
        stage_path = os.path.join(sample_dir, "01_stage1_arch.png")
        stage_img.save(stage_path)
        print(f"  Saved stage1_arch: {stage_path}")

    # Stage 2+: LLM 内部渐进剪枝（最多保存 3 张）
    for step_idx, kept_idx in enumerate(pruning_history[:3]):
        n_kept = len(kept_idx)
        label = f"Stage {step_idx + 2} (LLM layer)  kept={n_kept}/{total_tokens}"
        stage_img = render_stage(image, kept_idx, total_tokens, label)
        stage_path = os.path.join(sample_dir, f"0{step_idx + 2}_stage{step_idx + 2}_llm.png")
        stage_img.save(stage_path)
        print(f"  Saved stage{step_idx + 2}_llm: {stage_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/mnt/eason_ckp/models/llava-v1.6-vicuna-7b")
    parser.add_argument("--token_budget", type=int, default=32)
    parser.add_argument("--output_dir", default="vis_prefixvlm2")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to process")
    parser.add_argument("--stage1_keep", type=int, default=None,
                        help="Override STAGE1_KEEP (visual tokens entering LLM)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset",
                     choices=["mme", "mmvet", "pope", "textvqa", "gqa", "sqa"],
                     help="Use a standard eval dataset")
    src.add_argument("--samples_file",
                     help="JSON file with list of {image, question, name}")
    args = parser.parse_args()

    if args.stage1_keep is not None:
        os.environ["STAGE1_KEEP"] = str(args.stage1_keep)

    print(f"STAGE1_KEEP = {os.environ.get('STAGE1_KEEP', '1280')} (from env)")
    print("Loading model...")
    tokenizer, model, image_processor = load_model(args.model_path, args.token_budget)

    # ── load samples ──────────────────────────────────────────────────────────
    CKPT_DIR = os.environ.get("CKPT_DIR", "/mnt/eason_ckp/models")
    DATA_DIR  = os.environ.get("DATA_DIR",  "/mnt/eason_ckp/LLaVA-Eval")
    DATASETS = {
        "mme":     {"question_file": f"{DATA_DIR}/MME/llava_mme_test.jsonl",
                    "image_folder":  f"{DATA_DIR}/MME/MME_Benchmark_release_version"},
        "mmvet":   {"question_file": "/mnt/eason/LLaVA-STAR-Pro2/playground/data/eval/mm-vet/llava-mm-vet.jsonl",
                    "image_folder":  f"{DATA_DIR}/mm-vet/images"},
        "pope":    {"question_file": f"{DATA_DIR}/pope/llava_pope_test.jsonl",
                    "image_folder":  f"{DATA_DIR}/pope/val2014"},
        "textvqa": {"question_file": f"{DATA_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl",
                    "image_folder":  f"{DATA_DIR}/textvqa/train_images"},
        "gqa":     {"question_file": f"{DATA_DIR}/gqa/llava_gqa_testdev_balanced.jsonl",
                    "image_folder":  f"{DATA_DIR}/gqa/images/testdev_balanced"},
        "sqa":     {"question_file": f"{DATA_DIR}/scienceqa/llava_test_CQM-A.json",
                    "image_folder":  f"{DATA_DIR}/scienceqa/images/test"},
    }

    if args.dataset:
        cfg = DATASETS[args.dataset]
        qfile = cfg["question_file"]
        rows = [json.loads(l) for l in open(qfile)] if qfile.endswith(".jsonl") \
               else (lambda d: list(d.values()) if isinstance(d, dict) else d)(json.load(open(qfile)))
        samples = [
            {
                "image":    os.path.join(cfg["image_folder"], row.get("image", "")),
                "question": row.get("text", "Describe the image."),
                "name":     str(row.get("question_id", i)),
            }
            for i, row in enumerate(rows)
        ]
        print(f"Dataset: {args.dataset}  |  {len(samples)} total samples.")
    else:
        samples = json.load(open(args.samples_file))
        print(f"Custom file: {args.samples_file}  |  {len(samples)} total samples.")

    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"Processing {len(samples)} samples.")
    # ─────────────────────────────────────────────────────────────────────────
    for sample in samples:
        img_path = sample["image"]
        question = sample["question"]
        name = sample.get("name", os.path.splitext(os.path.basename(img_path))[0])

        print(f"\n[{name}] {question}")
        _, stage1_indices, mapped_history, total_tokens, answer = run_inference(
            img_path, question, tokenizer, model, image_processor
        )
        print(f"  Answer: {answer}")
        n_orig = total_tokens if total_tokens else "?"
        if stage1_indices is not None:
            print(f"  Stage1 (arch): {len(stage1_indices)}/{n_orig} tokens")
        for i, h in enumerate(mapped_history):
            print(f"  Stage{i+2} (LLM): {len(h)}/{n_orig} tokens")

        visualize_sample(
            img_path, question, name,
            stage1_indices, mapped_history, answer,
            total_tokens=total_tokens or int(os.environ.get("STAGE1_KEEP", "1280")),
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
