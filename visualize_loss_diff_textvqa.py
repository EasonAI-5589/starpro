"""
visualize_loss_diff_textvqa.py — Pruning Loss Fidelity Comparison on TextVQA

Same methodology as visualize_loss_diff.py but using TextVQA (llava_textvqa_val_v051_ocr.jsonl).

Usage:
  cd /mnt/eason/LLaVA-STAR-Pro2

  # Step 1 (once):
  python visualize_loss_diff_textvqa.py --step collect_vanilla \
      --model_path /mnt/eason_ckp/models/llava-v1.5-7b \
      --n_samples 200 --output_dir ./loss_diff_results_textvqa_200

  # Step 2 (one per method):
  python visualize_loss_diff_textvqa.py --step collect_method --method fastv    --budgets 32 64 128 --output_dir ./loss_diff_results_textvqa_200
  python visualize_loss_diff_textvqa.py --step collect_method --method divprune --budgets 32 64 128 --output_dir ./loss_diff_results_textvqa_200
  python visualize_loss_diff_textvqa.py --step collect_method --method scope    --budgets 32 64 128 --output_dir ./loss_diff_results_textvqa_200
  python visualize_loss_diff_textvqa.py --step collect_method --method tops     --budgets 32 64 128 --output_dir ./loss_diff_results_textvqa_200

  # Step 3:
  python visualize_loss_diff_textvqa.py --step plot --output_dir ./loss_diff_results_textvqa_200
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import gc
import json
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEXTVQA_JSONL   = "/mnt/eason_ckp/LLaVA-Eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
TEXTVQA_IMG_DIR = "/mnt/eason_ckp/LLaVA-Eval/textvqa/train_images"

METHOD_COLORS = {
    "fastv":    "#F4845F",
    "divprune": "#4895EF",
    "scope":    "#2EC4B6",
    "tops":     "#9B5DE5",
}
METHOD_LABELS = {
    "fastv":    "FastV",
    "divprune": "DivPrune",
    "scope":    "SCOPE",
    "tops":     "TOPS (Ours)",
}


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path: str, method: str, budget: int):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    if method == "vanilla":
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name,
            pruning_method=None,
            device_map="auto",
        )
        model.pruning_method = 'vanilla'

    elif method == "fastv":
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name,
            pruning_method="fastv",
            use_fastv=True,
            fastv_config={"K": 2, "R": budget},
            device_map="auto",
        )

    elif method == "divprune":
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name,
            pruning_method="divprune",
            visual_token_num=budget,
            device_map="auto",
        )

    elif method == "scope":
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name,
            pruning_method="scope",
            visual_token_num=budget,
            device_map="auto",
        )

    elif method == "tops":
        os.environ["STAGE1_KEEP"] = "288"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name,
            use_prefixvlm_2=True,
            prefixvlm_2_config={"T": budget},
            pruning_method="prefixvlm_2",
            device_map="auto",
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    model.eval()
    return tokenizer, model, image_processor


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def get_logits(sample: dict, tokenizer, model, image_processor) -> torch.Tensor:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from PIL import Image

    img_path = os.path.join(TEXTVQA_IMG_DIR, sample["image"])
    image = Image.open(img_path).convert("RGB")

    question = DEFAULT_IMAGE_TOKEN + "\n" + sample["text"]
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    images_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    _, position_ids, attention_mask, _, inputs_embeds, _, _ = \
        model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, images_tensor
        )

    lm_out = model.model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    logits = model.lm_head(lm_out.last_hidden_state[0, -1])
    return logits.float().cpu()


# ─── Step 1: collect vanilla ──────────────────────────────────────────────────

def collect_vanilla(model_path: str, n_samples: int, output_dir: str):
    with open(TEXTVQA_JSONL) as f:
        all_samples = [json.loads(l) for l in f if l.strip()]

    valid = [s for s in all_samples
             if os.path.exists(os.path.join(TEXTVQA_IMG_DIR, s["image"]))]
    random.seed(42)
    random.shuffle(valid)
    samples = valid[:n_samples]

    print(f"\n>>> Loading vanilla model: {model_path}")
    print(f">>> Dataset: TextVQA ({TEXTVQA_JSONL}), {len(samples)} samples")
    tokenizer, model, image_processor = load_model(model_path, "vanilla", 0)

    results = []
    for i, sample in enumerate(samples):
        try:
            logits = get_logits(sample, tokenizer, model, image_processor)
        except Exception as e:
            print(f"  [{i+1}] Error: {e}")
            continue

        top1_id   = int(logits.argmax().item())
        top1_loss = float(F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([top1_id])
        ).item())

        results.append({
            "sample_id":    sample.get("question_id", i),
            "image":        sample["image"],
            "text":         sample["text"],
            "top1_id":      top1_id,
            "top1_token":   tokenizer.decode([top1_id]),
            "vanilla_loss": top1_loss,
        })
        print(f"  [{len(results)}/{n_samples}] top1={tokenizer.decode([top1_id])!r}  "
              f"loss={top1_loss:.4f}", end="\r")

    free_model(model)
    print(f"\n  Done. {len(results)} samples.")

    path = os.path.join(output_dir, "vanilla_losses.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")
    return results


# ─── Step 2: collect pruned method ────────────────────────────────────────────

def collect_method(model_path: str, method: str, budgets: list, output_dir: str):
    vanilla_path = os.path.join(output_dir, "vanilla_losses.json")
    if not os.path.exists(vanilla_path):
        raise FileNotFoundError(
            f"vanilla_losses.json not found in {output_dir}. "
            "Run --step collect_vanilla first."
        )
    with open(vanilla_path) as f:
        vanilla_data = json.load(f)

    for budget in budgets:
        print(f"\n>>> Loading {method} model (budget={budget})")
        tokenizer, model, image_processor = load_model(model_path, method, budget)

        diffs = []
        for i, vd in enumerate(vanilla_data):
            sample = {"image": vd["image"], "text": vd["text"]}
            try:
                logits = get_logits(sample, tokenizer, model, image_processor)
            except Exception as e:
                print(f"  [{i+1}] Error: {e}")
                diffs.append(None)
                continue

            ref_id = vd["top1_id"]
            pruned_loss = float(F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([ref_id])
            ).item())
            loss_diff = pruned_loss - vd["vanilla_loss"]
            diffs.append(loss_diff)
            print(f"  [{i+1}/{len(vanilla_data)}] loss_diff={loss_diff:+.4f}", end="\r")

        free_model(model)
        print(f"\n  Done budget={budget}. valid={sum(d is not None for d in diffs)}")

        path = os.path.join(output_dir, f"{method}_t{budget}_diffs.json")
        with open(path, "w") as f:
            json.dump(diffs, f, indent=2)
        print(f"Saved → {path}")


# ─── Step 3: plot ─────────────────────────────────────────────────────────────

def plot(output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for _style in ("Regular", "Bold", "Italic", "BoldItalic"):
        _path = f"/usr/share/fonts/truetype/liberation/LiberationSerif-{_style}.ttf"
        try:
            fm.fontManager.addfont(_path)
        except Exception:
            pass
    plt.rcParams["font.family"] = "Liberation Serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    methods = ["fastv", "divprune", "scope", "tops"]
    budgets = [128, 64, 32]

    data = {}
    for method in methods:
        data[method] = {}
        for budget in budgets:
            path = os.path.join(output_dir, f"{method}_t{budget}_diffs.json")
            if not os.path.exists(path):
                print(f"  Missing: {path}")
                data[method][budget] = []
                continue
            with open(path) as f:
                raw = json.load(f)
            data[method][budget] = [x for x in raw if x is not None]

    all_vals = np.array([v for m in methods for b in budgets for v in data[m][b]])
    y_lo = np.percentile(all_vals, 0.5)  - 0.02 * np.ptp(np.percentile(all_vals, [1, 99]))
    y_hi = np.percentile(all_vals, 99.5) + 0.36 * np.ptp(np.percentile(all_vals, [1, 99]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True, facecolor="white")

    n_methods = len(methods)
    positions = np.arange(n_methods)
    width     = 0.58

    for ax_i, budget in enumerate(budgets):
        ax = axes[ax_i]
        ax.set_facecolor("#FAFAFA")
        ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--", alpha=0.7)

        vp_data = []
        vp_pos  = []
        for j, method in enumerate(methods):
            vals = np.array(data[method][budget])
            color = METHOD_COLORS[method]
            pos   = j

            if len(vals) == 0:
                vp_data.append([0])
                vp_pos.append(pos)
                continue

            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(pos + jitter, vals,
                       color=color, alpha=0.4, s=14, zorder=2, edgecolors="none")
            ax.scatter([pos], [vals.mean()],
                       color=color, s=110, zorder=5,
                       edgecolors="white", linewidths=1.2, marker="D")

            vp_data.append(vals)
            vp_pos.append(pos)

        if any(len(v) > 1 for v in vp_data):
            vp = ax.violinplot(
                [v if len(v) > 1 else [0, 0] for v in vp_data],
                positions=vp_pos, widths=width,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for k, (body, method) in enumerate(zip(vp["bodies"], methods)):
                body.set_facecolor(METHOD_COLORS[method])
                body.set_alpha(0.32)
                body.set_edgecolor(METHOD_COLORS[method])
                body.set_linewidth(1.2)
            segs = vp["cmedians"].get_segments()
            vp["cmedians"].set_visible(False)
            for seg, method in zip(segs, methods):
                ax.plot(seg[:, 0], seg[:, 1],
                        color=METHOD_COLORS[method], linewidth=2.2, zorder=4,
                        solid_capstyle="round")

        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(positions)
        ax.set_xticklabels([""] * n_methods)
        ax.tick_params(bottom=False)
        ax.set_title(f"Token Budget = {budget}", fontsize=14, fontweight="bold",
                     fontfamily="Liberation Serif")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=11)
        if ax_i > 0:
            ax.tick_params(left=False)
        for lbl in ax.get_yticklabels():
            lbl.set_fontfamily("Liberation Serif")
        if ax_i == 0:
            ax.set_ylabel("loss_diff  (↓ better)", fontsize=13,
                          fontfamily="Liberation Serif")

        for j, method in enumerate(methods):
            vals = np.array(data[method][budget])
            if len(vals) == 0:
                continue
            ax.text(j, -0.07, f"{vals.mean():+.3f}\n±{vals.std():.3f}",
                    ha="center", va="top", fontsize=9,
                    color="black", fontweight="bold",
                    fontfamily="Liberation Serif", zorder=6,
                    clip_on=False,
                    transform=ax.get_xaxis_transform())

        legend_handles = [
            plt.scatter([], [], color=METHOD_COLORS[m], s=80,
                        label=METHOD_LABELS[m], marker="o")
            for m in methods
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=12, framealpha=0.9, edgecolor="gray",
                  prop={"family": "Liberation Serif"})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    out_path = os.path.join(output_dir, "loss_diff_comparison.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["collect_vanilla", "collect_method", "plot"])
    parser.add_argument("--model_path", default="/mnt/eason_ckp/models/llava-v1.5-7b")
    parser.add_argument("--method", choices=["fastv", "divprune", "scope", "tops"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="./loss_diff_results_textvqa_200")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.step == "collect_vanilla":
        collect_vanilla(args.model_path, args.n_samples, args.output_dir)

    elif args.step == "collect_method":
        if not args.method:
            parser.error("--method required for collect_method step")
        collect_method(args.model_path, args.method, args.budgets, args.output_dir)

    elif args.step == "plot":
        plot(args.output_dir)


if __name__ == "__main__":
    main()
