#!/usr/bin/env python3
"""Standalone official-style OCRBench scorer for native LLaVA outputs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


CATEGORIES = (
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_correct(prediction: str, answers: list[str], category: str) -> bool:
    if category == "Handwritten Mathematical Expression Recognition":
        normalized_prediction = prediction.strip().replace("\n", " ").replace(" ", "")
        return any(
            str(answer).strip().replace("\n", " ").replace(" ", "")
            in normalized_prediction
            for answer in answers
        )
    normalized_prediction = prediction.lower().strip().replace("\n", " ")
    return any(
        str(answer).lower().strip().replace("\n", " ") in normalized_prediction
        for answer in answers
    )


def main() -> None:
    args = parse_args()
    metadata = {str(x["question_id"]): x for x in read_jsonl(args.metadata)}
    predictions = {str(x["question_id"]): str(x["text"]) for x in read_jsonl(args.predictions)}
    missing = set(metadata).difference(predictions)
    if missing:
        raise ValueError(f"Missing predictions for {sorted(missing)}")

    counts = {category: 0 for category in CATEGORIES}
    rows = []
    for qid, item in metadata.items():
        category = item["category"]
        if category not in counts:
            raise ValueError(f"Unknown OCRBench category: {category}")
        answers = ast.literal_eval(item["answer"])
        prediction = predictions[qid]
        correct = is_correct(prediction, answers, category)
        counts[category] += int(correct)
        rows.append({"question_id": qid, "category": category, "correct": correct, "prediction": prediction})

    text_recognition = sum(counts[x] for x in CATEGORIES[:6])
    final_score = (
        text_recognition
        + counts["Scene Text-centric VQA"]
        + counts["Doc-oriented VQA"]
        + counts["Key Information Extraction"]
        + counts["Handwritten Mathematical Expression Recognition"]
    )
    result = {
        "samples": len(rows),
        "category_hits": counts,
        "Text Recognition": text_recognition,
        "Final Score": final_score,
        "Final Score Norm": final_score / 10.0,
        "smoke_accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
