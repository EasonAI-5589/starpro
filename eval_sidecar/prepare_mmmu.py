#!/usr/bin/env python3
"""Materialize MMMU (val) for the native LLaVA evaluator (model_vqa_loader.py).

Source: VLMEvalKit ``MMMU_DEV_VAL.tsv`` (self-contained base64 images).
Output: images/<index>.jpg  +  questions.jsonl  +  metadata.jsonl

WHY SINGLE IMAGE PER QUESTION
-----------------------------
``llava/eval/model_vqa_loader.py`` opens exactly ONE image and emits ONE
``<image>`` token, and the STAR-Pro Stage-2 LLM pruner
(``modelling_llama_star.py``) hard-codes a SINGLE contiguous visual block
(``visual_start = system_prompt_length``; ``visual_end = +visual_token_length``).
Interleaving N images at their ``<image k>`` positions would (a) require loader
changes and (b) silently corrupt star_pro pruning, which would prune the wrong
token span. We therefore COLLAPSE every question to exactly one image at
conversion time, which keeps the validated single-image loader + star_pro path
byte-for-byte unchanged.

Two collapse modes (both yield ONE image file per question):
  * ``first``  (default): use image_1 only. This reproduces the OFFICIAL
    MMMU LLaVA-1.5 baseline (mmmu/utils/model_utils.py:call_llava_engine_df +
    data_utils.py:process_single_sample use ``data['image_1']`` and prepend a
    single ``<image>``), i.e. the ~35% reference.
  * ``concat``: horizontally concatenate all images into one composite so
    multi-image content is preserved while still presenting one ``<image>``.

Prompt format is the OFFICIAL MMMU one (mmmu/configs/llava1.5.yaml):
  MC   : "{question}\n\n(A) ..\n(B) ..\n\nAnswer with the option's letter from the given choices directly."
  open : "{question}\n\nAnswer the question using a single word or phrase."
model_vqa_loader.py prepends "<image>\n" and wraps with conv-mode vicuna_v1.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import json
import string
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

# Official MMMU prompt templates (mmmu/configs/llava1.5.yaml, task_instructions="").
MC_FMT = "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly."
OPEN_FMT = "{}\n\nAnswer the question using a single word or phrase."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/mnt/eason_ckp/LMUData/MMMU_DEV_VAL.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "dev", "all"],
        help="MMMU_DEV_VAL bundles dev(150)+validation(900); the reference "
        "number is on 'validation'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Take only the first N rows AFTER split filtering (0 = all). "
        "Use a small value for smoke tests.",
    )
    parser.add_argument(
        "--image-mode",
        default="first",
        choices=["first", "concat"],
        help="How to collapse multi-image questions to one image (see module docstring).",
    )
    parser.add_argument(
        "--concat-axis",
        default="h",
        choices=["h", "v"],
        help="Axis for --image-mode concat.",
    )
    return parser.parse_args()


def _as_image_list(cell) -> list[str]:
    """MMMU 'image' cell is either a single base64 string or a stringified
    Python list of base64 strings (VLMEvalKit convention)."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    s = str(cell)
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        return [str(x) for x in ast.literal_eval(s)]
    return [s]


def _decode(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
    except (binascii.Error, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Bad base64 image payload: {exc}") from exc
    return Image.open(BytesIO(raw)).convert("RGB")


def _concat(images: list[Image.Image], axis: str) -> Image.Image:
    if len(images) == 1:
        return images[0]
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    if axis == "h":
        canvas = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))
        x = 0
        for im in images:
            canvas.paste(im, (x, 0))
            x += im.width
    else:
        canvas = Image.new("RGB", (max(widths), sum(heights)), (255, 255, 255))
        y = 0
        for im in images:
            canvas.paste(im, (0, y))
            y += im.height
    return canvas


def _build_options(row: pd.Series) -> tuple[list[str], dict[str, str]]:
    """Return (all_choices, index2ans) using ORIGINAL column letters so the
    gold answer letter keeps pointing at the right option even with gaps."""
    all_choices: list[str] = []
    index2ans: dict[str, str] = {}
    for letter in string.ascii_uppercase:
        if letter in row and not pd.isna(row[letter]):
            all_choices.append(letter)
            index2ans[letter] = str(row[letter])
    return all_choices, index2ans


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    out = Path(args.output_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(source, sep="\t")
    for col in ("index", "question", "answer", "image"):
        if col not in data.columns:
            raise ValueError(f"MMMU TSV missing required column: {col!r}")

    if args.split != "all":
        if "split" not in data.columns:
            raise ValueError("TSV has no 'split' column; use --split all.")
        data = data[data["split"] == args.split].reset_index(drop=True)
        if len(data) == 0:
            raise ValueError(f"No rows for split={args.split!r}.")
    if args.limit > 0:
        data = data.head(args.limit)

    question_file = out / "questions.jsonl"
    metadata_file = out / "metadata.jsonl"
    n_mc = n_open = n_multi = 0

    with question_file.open("w", encoding="utf-8") as qf, metadata_file.open(
        "w", encoding="utf-8"
    ) as mf:
        for _, row in data.iterrows():
            index = str(row["index"])
            question = str(row["question"])
            # SUBJECT lives in 'l2-category' (e.g. 'Accounting'); the 'category' column is the DOMAIN
            # ('Business', ...). score_mmmu's DOMAIN_CAT2SUB_CAT maps subject->domain, so we must write the
            # SUBJECT here for both by_subject and by_domain breakdowns to populate. Overall is unaffected.
            if "l2-category" in row and not pd.isna(row.get("l2-category")):
                category = str(row["l2-category"])
            elif "category" in row and not pd.isna(row["category"]):
                category = str(row["category"])
            else:
                category = "Unknown"
            split = str(row["split"]) if "split" in row and not pd.isna(row.get("split")) else "none"

            all_choices, index2ans = _build_options(row)
            is_mc = len(all_choices) > 0

            if is_mc:
                n_mc += 1
                example = "".join(f"({L}) {index2ans[L]}\n" for L in all_choices)
                text = MC_FMT.format(question, example)
                gold = str(row["answer"]).strip().upper()
                question_type = "multiple-choice"
            else:
                n_open += 1
                text = OPEN_FMT.format(question)
                gold = str(row["answer"]).strip()
                question_type = "open"

            b64_list = _as_image_list(row["image"])
            if len(b64_list) > 1:
                n_multi += 1
            if args.image_mode == "first":
                pil_images = [_decode(b64_list[0])]
            else:  # concat
                pil_images = [_decode(b) for b in b64_list]
            composite = _concat(pil_images, args.concat_axis)
            image_name = f"{index}.jpg"
            composite.save(image_dir / image_name, format="JPEG", quality=95)

            qf.write(
                json.dumps(
                    {"question_id": index, "image": image_name, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            mf.write(
                json.dumps(
                    {
                        "question_id": index,
                        "question_type": question_type,
                        "answer": gold,
                        "all_choices": all_choices,
                        "index2ans": index2ans,
                        "options": [index2ans[L] for L in all_choices],
                        "category": category,
                        "split": split,
                        "num_images": len(b64_list),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Prepared {len(data)} MMMU[{args.split}] samples in {out} "
        f"(multiple-choice={n_mc}, open={n_open}, multi-image collapsed={n_multi}, "
        f"image-mode={args.image_mode})"
    )


if __name__ == "__main__":
    main()
