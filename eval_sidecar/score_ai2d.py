#!/usr/bin/env python3
"""Standalone AI2D scorer for native LLaVA outputs.

Replicates the authoritative VLMEvalKit AI2D_TEST accuracy pipeline EXACTLY, with
NO external GPT judge (the default `exact_matching` policy):

  * choice extraction   = can_infer(prediction, choices)
                        = can_infer_option (single option-letter token after
                          punctuation stripping) OR, on failure, can_infer_text
                          (exactly one option's text is a substring of the pred)
  * exact_matching miss = opt 'Z' (gold is never 'Z' -> hit=0); judge never called
  * accuracy            = mean of the 0/1 hit column (report_acc Overall), with a
                          per-category breakdown (AI2D has no split column)

The three matcher functions below are transcribed verbatim from
/mnt/eason/VLMEvalKit-STAR-Pro/vlmeval/utils/matching_util.py so a local score
matches an in-toolkit `evaluate()` call under exact matching. `choices` is
rebuilt per item from metadata's `options` (VLMEvalKit's build_choices), so the
substring fallback sees the identical option texts.
"""

from __future__ import annotations

import argparse
import copy as cp
import json
import os
import string
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Verbatim VLMEvalKit matchers (matching_util.py)
# ---------------------------------------------------------------------------
def can_infer_option(answer, choices):
    verbose = os.environ.get("VERBOSE", 0)
    # Choices is a dictionary
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3 and verbose:
                # verbose-only guard; inert under default (VERBOSE unset)
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_opt(prediction: str, options: dict) -> str:
    """VLMEvalKit extract_answer_from_item under model=None (exact_matching)."""
    # Fresh dict per item: can_infer_text lowercases choices in place.
    choices = {k: options[k] for k in options}
    ret = can_infer(prediction, choices)
    return ret if ret else "Z"


def main() -> None:
    args = parse_args()
    metadata = {str(x["question_id"]): x for x in read_jsonl(args.metadata)}
    predictions = {
        str(x["question_id"]): str(x["text"]) for x in read_jsonl(args.predictions)
    }
    missing = set(metadata).difference(predictions)
    if missing:
        raise ValueError(f"Missing predictions for {sorted(missing)}")

    rows = []
    cat_hits: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    for qid, item in metadata.items():
        gold = str(item["answer"]).strip()
        options = item["options"]
        prediction = predictions[qid]
        opt = extract_opt(prediction, options)
        hit = int(opt == gold)
        category = item.get("category", "none")
        cat_hits[category] += hit
        cat_total[category] += 1
        rows.append(
            {
                "question_id": qid,
                "category": category,
                "gold": gold,
                "extracted": opt,
                "hit": hit,
                "prediction": prediction,
            }
        )

    total = len(rows)
    overall_hits = sum(r["hit"] for r in rows)
    overall_acc = overall_hits / total if total else 0.0
    per_category = {
        c: {
            "n": cat_total[c],
            "hits": cat_hits[c],
            "acc": cat_hits[c] / cat_total[c] if cat_total[c] else 0.0,
        }
        for c in sorted(cat_total)
    }
    result = {
        "dataset": "AI2D_TEST",
        "policy": "exact_matching (no GPT judge)",
        "samples": total,
        "overall_hits": overall_hits,
        "Overall": overall_acc,            # fraction, matches report_acc
        "Overall_pct": round(100.0 * overall_acc, 2),
        "per_category": per_category,
        "rows": rows,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {k: result[k] for k in ("dataset", "samples", "Overall", "Overall_pct")}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
