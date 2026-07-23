#!/usr/bin/env python3
"""Official-style MMMU (val) scorer for native LLaVA model_vqa_loader outputs.

Replicates the OFFICIAL MMMU evaluation (MMMU-Benchmark/MMMU,
mmmu/utils/eval_utils.py + main_eval_only.py) with NO external GPT judge:
  * multiple-choice -> parse_multi_choice_response + eval_multi_choice
  * open-ended      -> parse_open_response       + eval_open
Overall accuracy = instruction-level accuracy over all scored samples
(calculate_ins_level_acc), i.e. total_correct / total, plus per-subject and
per-domain breakdowns matching the official leaderboard layout.

Usage:
  python score_mmmu.py --metadata metadata.jsonl --predictions answers.jsonl \
      --output score.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict

import numpy as np

random.seed(42)  # official MMMU sets this; governs the tie/last-resort random pick

# =====================================================================
# VERBATIM from MMMU-Benchmark/MMMU mmmu/utils/eval_utils.py
# =====================================================================

# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ----------- Process Open -------------
def check_is_number(string):
    """Check if the given string a number."""
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """Normalize the str to lower case and make them float numbers if possible."""
    # check if characters in the string
    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """Exact all forms of numbers from a string with regex."""
    # Pattern for numbers with commas
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ',
                              'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


# ----------- Evaluation -------------
def eval_multi_choice(gold_i, pred_i):
    """Evaluate a multiple choice instance."""
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """Evaluate an open question instance"""
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def calculate_ins_level_acc(results: Dict):
    """Calculate the instruction level accuracy for given Subject results"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results['acc'] * cat_results['num_example']
        ins_num += cat_results['num_example']
    if ins_num == 0:
        return 0
    return acc / ins_num


# Domain -> subject grouping (mmmu/utils/data_utils.py:DOMAIN_CAT2SUB_CAT).
DOMAIN_CAT2SUB_CAT = {
    'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
    'Business': ['Accounting', 'Economics', 'Finance', 'Manage', 'Marketing'],
    'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics'],
    'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine',
                            'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
    'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
    'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science',
                            'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


# =====================================================================
# Driver
# =====================================================================
def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Score only samples that have predictions instead of erroring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = {str(x["question_id"]): x for x in read_jsonl(args.metadata)}
    predictions = {str(x["question_id"]): str(x["text"]) for x in read_jsonl(args.predictions)}

    missing = set(metadata) - set(predictions)
    if missing:
        msg = f"Missing predictions for {len(missing)} ids, e.g. {sorted(missing)[:5]}"
        if not args.allow_missing:
            raise ValueError(msg)
        print("[warn] " + msg)
        for qid in missing:
            metadata.pop(qid)

    # per-subject buckets
    cat_stats: dict[str, dict] = {}
    qtype_stats = {"multiple-choice": [0, 0], "open": [0, 0]}  # [correct, total]
    rows = []

    for qid, item in metadata.items():
        pred_text = predictions[qid]
        qtype = item["question_type"]
        gold = item["answer"]
        if qtype == "multiple-choice":
            all_choices = list(item["all_choices"])
            index2ans = {str(k): str(v) for k, v in item["index2ans"].items()}
            parsed = parse_multi_choice_response(pred_text, all_choices, index2ans)
            correct = eval_multi_choice(gold, parsed)
        else:
            parsed = parse_open_response(pred_text)
            correct = eval_open(gold, parsed)

        category = item.get("category", "Unknown")
        cat = cat_stats.setdefault(category, {"correct": 0, "num_example": 0})
        cat["correct"] += int(correct)
        cat["num_example"] += 1
        qtype_stats[qtype][0] += int(correct)
        qtype_stats[qtype][1] += 1
        rows.append({
            "question_id": qid,
            "question_type": qtype,
            "category": category,
            "gold": gold,
            "parsed_pred": parsed if qtype == "multiple-choice" else "<open-list>",
            "correct": bool(correct),
        })

    # per-subject accuracy (official layout expects {acc, num_example})
    evaluation_result = {
        cat: {"acc": s["correct"] / s["num_example"], "num_example": s["num_example"]}
        for cat, s in cat_stats.items()
    }

    # per-domain instruction-level accuracy
    domain_results = {}
    for domain, subs in DOMAIN_CAT2SUB_CAT.items():
        in_domain = {c: evaluation_result[c] for c in subs if c in evaluation_result}
        if not in_domain:
            continue
        domain_results[domain] = {
            "acc": round(calculate_ins_level_acc(in_domain), 4),
            "num_example": sum(v["num_example"] for v in in_domain.values()),
        }

    overall_acc = calculate_ins_level_acc(evaluation_result)
    total = sum(s["num_example"] for s in cat_stats.values())
    total_correct = sum(s["correct"] for s in cat_stats.values())

    result = {
        "Overall": {"acc": round(overall_acc, 4), "num_example": total, "num_correct": total_correct},
        "by_question_type": {
            qt: {"acc": (c / n if n else 0.0), "num_example": n}
            for qt, (c, n) in qtype_stats.items()
        },
        "by_domain": domain_results,
        "by_subject": {
            c: {"acc": round(v["acc"], 4), "num_example": v["num_example"]}
            for c, v in sorted(evaluation_result.items())
        },
        "rows": rows,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in ("Overall", "by_question_type", "by_domain")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
