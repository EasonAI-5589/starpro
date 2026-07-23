#!/usr/bin/env python3
"""Materialize a small OCRBench subset for the native LLaVA evaluator."""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/mnt/eason_ckp/LMUData/OCRBench.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    out = Path(args.output_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(source, sep="\t", nrows=args.limit)
    required = {"index", "image", "question", "answer", "category"}
    if missing := required.difference(data.columns):
        raise ValueError(f"OCRBench TSV missing columns: {sorted(missing)}")

    question_file = out / "questions.jsonl"
    metadata_file = out / "metadata.jsonl"
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
            qf.write(
                json.dumps(
                    {"question_id": index, "image": image_name, "text": question},
                    ensure_ascii=False,
                )
                + "\n"
            )
            mf.write(
                json.dumps(
                    {
                        "question_id": index,
                        "question": question,
                        "answer": str(row["answer"]),
                        "category": str(row["category"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Prepared {len(data)} OCRBench samples in {out}")


if __name__ == "__main__":
    main()
