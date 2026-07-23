#!/usr/bin/env python3
"""Run OCRBench or RefCOCO with the recovered STAR-Pro LLaVA codebase."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["ocrbench", "refcoco"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--model-path", default="/mnt/eason_ckp/models/llava-v1.5-7b")
    parser.add_argument("--llava-root", default="/mnt/eason/LLaVA-STAR-Pro2-qr_recover_20260723")
    parser.add_argument("--vlmeval-root", default="/mnt/eason/VLMEvalKit-STAR-Pro")
    parser.add_argument("--lmu-data", default="/mnt/eason_ckp/LMUData")
    parser.add_argument("--work-dir", default="/mnt/eason_ckp/starpro_eval/llava_refcoco_ocrbench")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--refcoco-category", default="RefCOCO val")
    parser.add_argument("--refcoco-tsv", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only; never use for reported scores.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    for root in (str(here), args.llava_root, args.vlmeval_root):
        if root not in sys.path:
            sys.path.insert(0, root)
    os.environ["LMUData"] = args.lmu_data

    from refcoco import RefCOCO
    from starpro_llava import StarProLLaVA
    from vlmeval.dataset.image_vqa import OCRBench
    from vlmeval.inference import infer_data_job

    if args.benchmark == "ocrbench":
        dataset = OCRBench(dataset="OCRBench")
    else:
        dataset = RefCOCO(
            category=args.refcoco_category,
            source_tsv=args.refcoco_tsv,
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        dataset.data = dataset.data.iloc[:args.limit].copy()

    model = StarProLLaVA(
        model_path=args.model_path,
        pruning_method=args.method,
        visual_token_num=args.tokens,
        max_new_tokens=args.max_new_tokens,
    )
    dataset_label = dataset.dataset_name
    model_label = f"llava_v1_5_7b_{args.method}_vtn{args.tokens}"
    work_dir = Path(args.work_dir) / args.benchmark / args.method / f"vtn_{args.tokens}"
    work_dir.mkdir(parents=True, exist_ok=True)

    infer_data_job(
        model=model,
        pruning_method=args.method,
        visual_token_num=args.tokens,
        work_dir=str(work_dir),
        model_name=model_label,
        dataset=dataset,
        verbose=False,
    )
    result_file = work_dir / f"{model_label}_{dataset_label}.xlsx"
    score = dataset.evaluate(str(result_file))
    manifest = {
        "benchmark": args.benchmark,
        "dataset": dataset_label,
        "method": args.method,
        "requested_visual_tokens": args.tokens,
        "model_path": args.model_path,
        "result_file": str(result_file),
        "score": score,
        "subset_limit": args.limit,
    }
    manifest_path = work_dir / f"{model_label}_{dataset_label}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
