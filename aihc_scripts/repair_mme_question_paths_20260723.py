#!/usr/bin/env python3
"""Create a validated MME question file without mutating the shared dataset.

The recovered MME archive stores five public subsets directly under
``<category>/<image>.jpg``, while the restored question JSONL refers to
``<category>/images/<image>.jpg``. This tool rewrites only paths that are
missing at the original location and present at the flattened location.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    rewritten = 0

    with args.source.open() as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError(f"missing image field at line {line_number}")

            original = args.image_root / image
            if not original.is_file():
                flattened = image.replace("/images/", "/")
                candidate = args.image_root / flattened
                if flattened == image or not candidate.is_file():
                    raise FileNotFoundError(
                        f"cannot resolve image at line {line_number}: {image}"
                    )
                row["image"] = flattened
                rewritten += 1
            rows.append(row)

    if len(rows) != 2374:
        raise ValueError(f"unexpected MME question count: {len(rows)}")

    unresolved = [
        row["image"]
        for row in rows
        if not (args.image_root / row["image"]).is_file()
    ]
    if unresolved:
        raise FileNotFoundError(
            f"{len(unresolved)} MME images remain unresolved: {unresolved[:5]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)

    print(
        "MME_PATH_REPAIR_OK "
        f"questions={len(rows)} rewritten_rows={rewritten} unresolved=0 "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
