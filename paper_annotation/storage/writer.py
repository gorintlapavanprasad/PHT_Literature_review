"""Storage utilities for JSONL output."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from pipeline.annotate import AnnotationRecord


def write_jsonl(records: Iterable[AnnotationRecord], output_path: Path) -> None:
    """Write annotation records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=output_path.parent, encoding="utf-8") as tmp:
        for record in records:
            tmp.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, output_path)


if __name__ == "__main__":
    print("Writer ready.")
