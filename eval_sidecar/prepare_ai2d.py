#!/usr/bin/env python3
"""Materialize AI2D_TEST for the native LLaVA evaluator (reportable path).

Reads the standard VLMEvalKit AI2D_TEST.tsv (base64 images + A/B/C/D options +
gold letter) and emits, next to each other under --output-dir:

  images/<index>.jpg   decoded diagram pixels (one per row)
  questions.jsonl      {"question_id","image","text"} for model_vqa_loader.py
  metadata.jsonl       {"question_id","question","options","answer","category",
                        "abcLabel"} for score_ai2d.py

The prompt is the canonical LLaVA multiple-choice prompt (identical shape to the
validated SEED run):

  <question>
  A. <optA>
  B. <optB>
  ...            # only NON-NULL option columns, labelled by their column letter
  Answer with the option's letter from the given choices directly.

Option letters are the TSV column letters (A/B/C/D). NaN option columns are
skipped; a present option keeps its own column letter, so the emitted letter
always equals the column name and matches the gold `answer` letter. The loader
prepends the image token + vicuna template itself, so `text` carries no image
token.
"""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

LETTERS = ("A", "B", "C", "D")
MC_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/mnt/eason_ckp/LMUData/AI2D_TEST.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all rows (full 3088); >0 = first N rows for a smoke.",
    )
    return parser.parse_args()


def build_options(row: pd.Series) -> dict[str, str]:
    """Non-null options keyed by their own column letter (A/B/C/D)."""
    options: dict[str, str] = {}
    for letter in LETTERS:
        if letter in row and not pd.isna(row[letter]):
            options[letter] = str(row[letter])
    return options


def build_prompt(question: str, options: dict[str, str]) -> str:
    lines = [question]
    for letter, text in options.items():
        lines.append(f"{letter}. {text}")
    lines.append(MC_INSTRUCTION)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    out = Path(args.output_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    nrows = args.limit if args.limit and args.limit > 0 else None
    data = pd.read_csv(source, sep="\t", nrows=nrows)
    required = {"index", "question", "answer", "image"}
    if missing := required.difference(data.columns):
        raise ValueError(f"AI2D_TEST TSV missing columns: {sorted(missing)}")

    question_file = out / "questions.jsonl"
    metadata_file = out / "metadata.jsonl"
    written = 0
    with question_file.open("w", encoding="utf-8") as qf, metadata_file.open(
        "w", encoding="utf-8"
    ) as mf:
        for _, row in data.iterrows():
            index = str(row["index"])
            image_name = f"{index}.jpg"
            raw_image = base64.b64decode(str(row["image"]))
            with Image.open(BytesIO(raw_image)) as image:
                image.convert("RGB").save(image_dir / image_name, format="JPEG")

            question = str(row["question"])
            options = build_options(row)
            gold = str(row["answer"]).strip()
            if gold not in options:
                raise ValueError(
                    f"Row index={index}: gold letter {gold!r} has no matching "
                    f"option column among {sorted(options)}"
                )
            prompt = build_prompt(question, options)

            qf.write(
                json.dumps(
                    {"question_id": index, "image": image_name, "text": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )
            mf.write(
                json.dumps(
                    {
                        "question_id": index,
                        "question": question,
                        "options": options,
                        "answer": gold,
                        "category": str(row["category"]) if "category" in row else "none",
                        "abcLabel": bool(row["abcLabel"]) if "abcLabel" in row else False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Prepared {written} AI2D samples in {out}")


if __name__ == "__main__":
    main()
