"""RefCOCO data filtering and box-based evaluation for the STAR-Pro sidecar."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import LMUDataRoot


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_BRACKETED_BOX = re.compile(
    rf"[\[\(]\s*({_NUMBER})\s*,?\s*({_NUMBER})\s*,?\s*({_NUMBER})\s*,?\s*({_NUMBER})\s*[\]\)]"
)


def parse_normalized_box(prediction: object) -> list[float] | None:
    """Parse the first four-coordinate box and return a clipped xyxy box.

    Reference labels are normalized to [0, 1].  Some LLaVA variants answer in
    a 0--1000 coordinate convention, which is converted only when unambiguous.
    Invalid or inverted boxes are deliberately scored as zero rather than
    coerced into a plausible answer.
    """

    text = str(prediction).strip()
    values: list[float] | None = None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            values = [float(x) for x in parsed[:4]]
    except (SyntaxError, ValueError, TypeError):
        pass
    if values is None:
        match = _BRACKETED_BOX.search(text)
        if match is not None:
            values = [float(x) for x in match.groups()]
    if values is None or not all(np.isfinite(values)):
        return None

    if max(abs(x) for x in values) > 1.5:
        if max(abs(x) for x in values) <= 1000.0:
            values = [x / 1000.0 for x in values]
        else:
            return None
    x1, y1, x2, y2 = [min(1.0, max(0.0, x)) for x in values]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _box_from_answer(answer: object) -> list[float]:
    value = ast.literal_eval(answer) if isinstance(answer, str) else answer
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"RefCOCO label is not an xyxy box: {answer!r}")
    box = [float(x) for x in value]
    if not all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid RefCOCO label: {answer!r}")
    return box


def box_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1])
    union += (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return 0.0 if union <= 0 else inter / union


class RefCOCO(ImageBaseDataset):
    """The canonical RefCOCO category from ``mjuicem/RefCOCO-VLMEvalKit``."""

    TYPE = "VQA"
    MODALITY = "IMAGE"

    def __init__(
        self,
        category: str = "RefCOCO val",
        source_tsv: str | None = None,
        chunksize: int = 256,
    ) -> None:
        self.category = category
        self.source_tsv = Path(source_tsv) if source_tsv else Path(LMUDataRoot()) / "RefCOCO.tsv"
        self.chunksize = chunksize
        dataset_key = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_")
        super().__init__(dataset=f"RefCOCO_{dataset_key}")

    @classmethod
    def supported_datasets(cls) -> list[str]:
        return []

    def prepare_tsv(self, url: str, file_md5: str | None = None) -> pd.DataFrame:
        if not self.source_tsv.is_file():
            raise FileNotFoundError(
                f"Missing {self.source_tsv}. Download the official RefCOCO.tsv before evaluation."
            )
        rows: list[pd.DataFrame] = []
        for chunk in pd.read_csv(self.source_tsv, sep="\t", chunksize=self.chunksize):
            if "category" not in chunk:
                raise ValueError("RefCOCO.tsv has no category column")
            selected = chunk[chunk["category"].astype(str) == self.category]
            if len(selected):
                rows.append(selected)
        if not rows:
            raise ValueError(f"No rows match category {self.category!r}")
        data = pd.concat(rows, ignore_index=True)
        required = {"index", "image", "question", "answer", "category"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"RefCOCO.tsv missing columns: {sorted(missing)}")
        return data

    def evaluate(self, eval_file: str, **judge_kwargs: object) -> dict[str, float | int]:
        suffix = Path(eval_file).suffix
        if suffix not in {".xlsx", ".tsv", ".csv"}:
            raise ValueError(f"Unsupported prediction file: {eval_file}")
        if suffix == ".xlsx":
            data = pd.read_excel(eval_file)
        elif suffix == ".tsv":
            data = pd.read_csv(eval_file, sep="\t")
        else:
            data = pd.read_csv(eval_file)
        if not {"answer", "prediction"}.issubset(data.columns):
            raise ValueError("Prediction file needs answer and prediction columns")

        ious: list[float] = []
        parsed = 0
        for _, line in data.iterrows():
            target = _box_from_answer(line["answer"])
            predicted = parse_normalized_box(line["prediction"])
            if predicted is None:
                ious.append(0.0)
            else:
                parsed += 1
                ious.append(box_iou(target, predicted))
        metric = {
            "category": self.category,
            "samples": int(len(ious)),
            "parsed_predictions": parsed,
            "parse_rate": 100.0 * parsed / len(ious) if ious else 0.0,
            "acc@0.5": 100.0 * sum(iou >= 0.5 for iou in ious) / len(ious) if ious else 0.0,
            "mIoU": 100.0 * float(np.mean(ious)) if ious else 0.0,
        }
        score_path = Path(eval_file).with_name(Path(eval_file).stem + "_score.json")
        score_path.write_text(json.dumps(metric, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return metric
