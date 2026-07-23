"""
Token pruning visualization for training-free VLM pruning methods.
Supports: divprune, scope, dart, fastv, vanilla, ...

Loads the model with the specified pruning_method, monkey-patches encode_images
to capture the selected token indices, then renders original vs. pruned images.

Usage:
    python visualize_pruning.py --method divprune --dataset textvqa --token_budget 64
    python visualize_pruning.py --method scope    --dataset mme     --token_budget 128
    python visualize_pruning.py --method divprune --samples_file vis_samples2.json --token_budget 64
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

DIM_FACTOR = 0.3   # brightness of pruned patches

# Global capture: filled by patched encode_images
_PRUNING_EVENT = {"select_idx": None, "total_tokens": None}


def _patch_encode_images():
    """
    Wrap encode_images once to capture any method's retained token indices.
    encode_images returns (image_features, index_masks, merged_features);
    we derive select_idx from index_masks without re-implementing any logic.
    """
    from llava.model.llava_arch import LlavaMetaForCausalLM
    orig_encode = LlavaMetaForCausalLM.encode_images

    def patched_encode(self, images, texts=None):
        image_features, index_masks, merged_features = orig_encode(self, images, texts)
        _PRUNING_EVENT["select_idx"] = index_masks[0].nonzero(as_tuple=False).squeeze(1).cpu()
        _PRUNING_EVENT["total_tokens"] = index_masks.shape[1]
        return image_features, index_masks, merged_features

    LlavaMetaForCausalLM.encode_images = patched_encode


def load_model(model_path: str, method: str, token_budget: int):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name,
        pruning_method=method,
        visual_token_num=token_budget,
        device_map="cuda:0",
    )
    model.eval()
    _patch_encode_images()
    return tokenizer, model, image_processor


@torch.inference_mode()
def run_inference(image_path, question, tokenizer, model, image_processor):
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

    _PRUNING_EVENT["select_idx"] = None
    _PRUNING_EVENT["total_tokens"] = None

    out_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        max_new_tokens=64,
        use_cache=True,
    )
    if isinstance(out_ids, tuple):
        out_ids = out_ids[0]
    answer = tokenizer.decode(
        out_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    return image, _PRUNING_EVENT["select_idx"], _PRUNING_EVENT["total_tokens"], answer


def infer_grid(n_tokens: int, img_w: int, img_h: int):
    aspect = img_w / img_h
    grid_h = max(1, round(math.sqrt(n_tokens / aspect)))
    grid_w = max(1, round(n_tokens / grid_h))
    while grid_h * grid_w < n_tokens:
        grid_w += 1
    return grid_h, grid_w


def render_pruned(image: Image.Image, kept_indices, total_tokens: int,
                  label: str) -> Image.Image:
    img_w, img_h = image.size
    grid_h, grid_w = infer_grid(total_tokens, img_w, img_h)

    mask = np.zeros(grid_h * grid_w, dtype=bool)
    idx = kept_indices.numpy() if isinstance(kept_indices, torch.Tensor) else np.array(kept_indices)
    valid = idx[idx < total_tokens]
    mask[valid] = True
    mask = mask.reshape(grid_h, grid_w)

    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    orig = image.convert("RGB")
    dim = orig.point(lambda p: int(p * DIM_FACTOR))
    result = dim.copy()
    draw = ImageDraw.Draw(result)

    for row in range(grid_h):
        for col in range(grid_w):
            linear = row * grid_w + col
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            x1 = int((col + 1) * cell_w)
            y1 = int((row + 1) * cell_h)
            if linear >= total_tokens:
                draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
            elif mask[row, col]:
                patch = orig.crop((x0, y0, x1, y1))
                result.paste(patch, (x0, y0))
                draw.rectangle([x0, y0, x1 - 1, y1 - 1],
                                outline=(255, 255, 255), width=1)

    bar_h = 28
    canvas = Image.new("RGB", (img_w, img_h + bar_h), (30, 30, 30))
    canvas.paste(result, (0, bar_h))
    bar_draw = ImageDraw.Draw(canvas)
    bar_draw.text((6, 5), label, fill=(255, 220, 60))
    return canvas


def make_combined(orig_image: Image.Image, pruned_panel: Image.Image,
                  question: str, answer: str, sample_name: str) -> Image.Image:
    w, h = pruned_panel.size
    bar_h = 28

    orig_panel = Image.new("RGB", (w, h), (30, 30, 30))
    orig_resized = orig_image.convert("RGB").resize((w, h - bar_h))
    orig_panel.paste(orig_resized, (0, bar_h))
    orig_draw = ImageDraw.Draw(orig_panel)
    orig_draw.text((6, 5), "Original", fill=(180, 255, 180))

    title_h = 44
    total_w = w * 2
    total_h = h + title_h
    out = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(out)

    title = f"{sample_name}  |  Q: {question[:80]}  |  A: {answer}"
    draw.text((8, 10), title, fill=(220, 220, 220))

    out.paste(orig_panel,   (0,     title_h))
    out.paste(pruned_panel, (w,     title_h))
    return out


# ── dataset registry (mirrors eval.sh script paths) ──────────────────────────
CKPT_DIR = os.environ.get("CKPT_DIR", "/mnt/eason_ckp/models")
DATA_DIR  = os.environ.get("DATA_DIR",  "/mnt/eason_ckp/LLaVA-Eval")

DATASETS = {
    "mme": {
        "question_file": f"{DATA_DIR}/MME/llava_mme_test.jsonl",
        "image_folder":  f"{DATA_DIR}/MME/MME_Benchmark_release_version",
        "image_key": "image", "text_key": "text",
    },
    "mmvet": {
        "question_file": "/mnt/eason/LLaVA-STAR-Pro2/playground/data/eval/mm-vet/llava-mm-vet.jsonl",
        "image_folder":  f"{DATA_DIR}/mm-vet/images",
        "image_key": "image", "text_key": "text",
    },
    "pope": {
        "question_file": f"{DATA_DIR}/pope/llava_pope_test.jsonl",
        "image_folder":  f"{DATA_DIR}/pope/val2014",
        "image_key": "image", "text_key": "text",
    },
    "textvqa": {
        "question_file": f"{DATA_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl",
        "image_folder":  f"{DATA_DIR}/textvqa/train_images",
        "image_key": "image", "text_key": "text",
    },
    "gqa": {
        "question_file": f"{DATA_DIR}/gqa/llava_gqa_testdev_balanced.jsonl",
        "image_folder":  f"{DATA_DIR}/gqa/images/testdev_balanced",
        "image_key": "image", "text_key": "text",
    },
    "sqa": {
        "question_file": f"{DATA_DIR}/scienceqa/llava_test_CQM-A.json",
        "image_folder":  f"{DATA_DIR}/scienceqa/images/test",
        "image_key": "image", "text_key": "text",
    },
}


def load_samples_from_dataset(dataset_name: str, max_samples: int):
    """Load samples from a standard eval dataset jsonl/json file."""
    cfg = DATASETS[dataset_name]
    qfile = cfg["question_file"]
    img_folder = cfg["image_folder"]
    image_key = cfg["image_key"]
    text_key  = cfg["text_key"]

    # Support both jsonl and json
    if qfile.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(qfile)]
    else:
        rows = json.load(open(qfile))
        if isinstance(rows, dict):
            rows = list(rows.values())

    samples = []
    for row in rows[:max_samples]:
        img_rel = row.get(image_key, "")
        img_path = os.path.join(img_folder, img_rel) if img_rel else ""
        samples.append({
            "name":     str(row.get("question_id", len(samples))),
            "image":    img_path,
            "question": row.get(text_key, "Describe the image."),
        })
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method",        default="divprune",
                   help="Pruning method: divprune, scope, dart, fastv, ...")
    p.add_argument("--model_path",    default=f"{CKPT_DIR}/llava-v1.5-7b")
    p.add_argument("--token_budget",  type=int, default=128)
    p.add_argument("--output_dir",    default=None,
                   help="Output dir (default: {method}_vis_output)")
    p.add_argument("--max_samples",   type=int, default=20)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset",      choices=list(DATASETS.keys()),
                     help="Use a standard eval dataset")
    src.add_argument("--samples_file", help="Custom JSON samples file")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/mnt/eason/LLaVA-STAR-Pro2/{args.method}_vis_output"

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model, image_processor = load_model(
        args.model_path, args.method, args.token_budget)
    print(f"Model loaded. method={args.method}  token_budget={args.token_budget}")

    if args.dataset:
        samples = load_samples_from_dataset(args.dataset, args.max_samples)
        print(f"Dataset: {args.dataset}  |  {len(samples)} samples.")
    else:
        samples = json.load(open(args.samples_file))[:args.max_samples]
        print(f"Custom file: {args.samples_file}  |  {len(samples)} samples.")

    for i, sample in enumerate(samples):
        name     = sample.get("name", f"sample_{i:03d}")
        img_path = sample["image"]
        question = sample.get("question", "Describe the image.")

        print(f"[{i+1}/{len(samples)}] {name}")

        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}, skipping.")
            continue

        try:
            image, select_idx, total_tokens, answer = run_inference(
                img_path, question, tokenizer, model, image_processor)
        except Exception as e:
            print(f"  Inference error: {e}")
            continue

        if select_idx is None:
            print("  No pruning event captured.")
            continue

        n_kept   = len(select_idx)
        n_total  = total_tokens
        pct      = 100.0 * n_kept / n_total
        print(f"  Kept {n_kept}/{n_total} tokens ({pct:.1f}%)  |  A: {answer[:60]}")

        label = f"{args.method.upper()}  kept {n_kept}/{n_total} ({pct:.0f}%)"
        pruned_panel = render_pruned(
            image.resize((336, 336)), select_idx, n_total, label)

        combined = make_combined(
            image, pruned_panel, question, answer, name)

        out_path = os.path.join(args.output_dir, f"{i:03d}_{name}.png")
        combined.save(out_path)
        print(f"  Saved: {out_path}")

        # Also save indices as npy for further analysis
        np.save(os.path.join(args.output_dir, f"{i:03d}_{name}_indices.npy"),
                select_idx.numpy())

    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
