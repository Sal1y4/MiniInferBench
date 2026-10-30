"""JSONL result store: one line per run, raw quantities kept alongside summaries.

Raw samples (every TTFT, every inter-token gap) are stored so that metrics can
be recomputed later without re-running the experiment.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"schema_version": SCHEMA_VERSION, **record}, default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path}:{lineno}: unsupported schema_version")
            yield record


def find_record(path: Path, run_id: str) -> dict[str, Any]:
    for record in read_records(path):
        if record.get("run_id") == run_id:
            return record
    raise KeyError(f"run {run_id!r} not found in {path}")
