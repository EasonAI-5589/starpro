#!/usr/bin/env python3
"""Convert MME answers with robust dataset-layout and whitespace handling."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


EXPECTED_CATEGORIES = {
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("."))
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_ground_truth(release_root: Path) -> dict[tuple[str, str, str], str]:
    ground_truth: dict[tuple[str, str, str], str] = {}

    for category_dir in sorted(path for path in release_root.iterdir() if path.is_dir()):
        category = category_dir.name
        candidate_dirs = [category_dir]
        nested = category_dir / "questions_answers_YN"
        if nested.is_dir():
            candidate_dirs.append(nested)

        for question_dir in candidate_dirs:
            for question_file in sorted(question_dir.glob("*.txt")):
                for line in question_file.read_text().splitlines():
                    question, answer = line.split("\t")
                    key = (
                        category,
                        question_file.name,
                        normalize_whitespace(question),
                    )
                    existing = ground_truth.get(key)
                    if existing is not None and existing != answer:
                        raise ValueError(f"conflicting ground truth for {key}")
                    ground_truth[key] = answer

    return ground_truth


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.replace(
        "Answer the question using a single word or phrase.", ""
    ).strip()
    if "Please answer yes or no." not in prompt:
        prompt += " Please answer yes or no."
    return normalize_whitespace(prompt)


def normalize_prediction(text: str) -> str:
    prediction = text.replace("\n", " ").strip().lower()
    if prediction.startswith("yes"):
        return "Yes"
    if prediction.startswith("no"):
        return "No"
    if re.search(r"\byes\b", prediction):
        return "Yes"
    if re.search(r"\bno\b", prediction):
        return "No"
    return prediction


def main() -> None:
    args = parse_args()
    data_path = args.data_path.resolve()
    answer_path = data_path / "answers" / f"{args.experiment}.jsonl"
    output_dir = data_path / "eval_tool" / "answers" / args.experiment
    release_root = data_path / "MME_Benchmark_release_version"

    ground_truth = load_ground_truth(release_root)
    answers = [json.loads(line) for line in answer_path.read_text().splitlines()]
    if len(answers) != 2374:
        raise ValueError(f"unexpected MME answer count: {len(answers)}")

    results: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for answer in answers:
        question_id = answer["question_id"]
        category = question_id.split("/")[0]
        file_name = question_id.split("/")[-1].split(".")[0] + ".txt"
        prompt = normalize_prompt(answer["prompt"])
        key = (category, file_name, prompt)
        if key not in ground_truth:
            raise KeyError(f"MME ground truth not found for {key}")
        results[category].append(
            (
                file_name,
                prompt,
                ground_truth[key],
                normalize_prediction(answer["text"]),
            )
        )

    if set(results) != EXPECTED_CATEGORIES:
        missing = sorted(EXPECTED_CATEGORIES - set(results))
        extra = sorted(set(results) - EXPECTED_CATEGORIES)
        raise ValueError(f"MME category mismatch: missing={missing}, extra={extra}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for category, rows in sorted(results.items()):
        output_file = output_dir / f"{category}.txt"
        temporary = output_file.with_suffix(".txt.tmp")
        with temporary.open("w") as stream:
            for row in rows:
                stream.write("\t".join(row) + "\n")
                written += 1
        temporary.replace(output_file)

    if written != len(answers):
        raise ValueError(f"MME output count mismatch: {written} != {len(answers)}")

    print(
        "MME_CONVERT_ROBUST_OK "
        f"ground_truth={len(ground_truth)} answers={len(answers)} "
        f"categories={len(results)} written={written} output={output_dir}"
    )


if __name__ == "__main__":
    main()
