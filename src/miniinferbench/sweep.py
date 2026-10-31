"""Parameter sweep: one model load, many configurations, one record per point.

Unlike DeepInfer's sweep (one process per point), points here share a process
so the model is loaded once. The trade-off is shared allocator and clock state
between points; we free cached memory between points and keep warmup runs per
point to limit that. Every point is still a full harness run with its own
status, so a crash or OOM at one point does not stop the sweep.
"""

from __future__ import annotations

import gc
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from miniinferbench.bench import load_model
from miniinferbench.config import BenchmarkConfig
from miniinferbench.harness import HarnessOptions, run_task
from miniinferbench.manifest import build_manifest, new_run_id
from miniinferbench.memory import GIB, available_bytes, estimate, model_shape
from miniinferbench.results import append_record, read_records
from miniinferbench.runner import greedy_decode
from miniinferbench.status import RunStatus


@dataclass(frozen=True, slots=True)
class SweepGrid:
    batch_sizes: Sequence[int] = (1, 2, 4, 8)
    input_lens: Sequence[int] = (128, 512, 1024)
    output_lens: Sequence[int] = (32, 128, 256)

    def configs(self, base: BenchmarkConfig) -> Iterator[BenchmarkConfig]:
        for bs, il, ol in itertools.product(self.batch_sizes, self.input_lens, self.output_lens):
            yield replace(base, batch_size=bs, input_len=il, output_len=ol)


def _point_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(sorted(config.items()))


def completed_points(results: Path, tag: str) -> set[tuple[Any, ...]]:
    """Points of this tag that already ended OK or SKIPPED (used to resume)."""
    if not results.exists():
        return set()
    done = {RunStatus.OK.value, RunStatus.SKIPPED.value}
    return {
        _point_key(r["config"])
        for r in read_records(results)
        if r.get("sweep_tag") == tag and r.get("status") in done
    }


def skipped_record(config: BenchmarkConfig, reason: str, estimate_bytes: int) -> dict[str, Any]:
    run_id = new_run_id()
    return {
        "run_id": run_id,
        "config": config.to_dict(),
        "manifest": build_manifest(run_id),
        "status": RunStatus.SKIPPED.value,
        "reason": reason,
        "memory_estimate_bytes": estimate_bytes,
    }


def run_sweep(
    base: BenchmarkConfig,
    grid: SweepGrid,
    results: Path,
    tag: str,
    options: HarnessOptions | None = None,
) -> list[dict[str, Any]]:
    """Run every grid point not yet completed under `tag`; returns the new records."""
    options = options or HarnessOptions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, special_ids = load_model(base, device)
    shape = model_shape(model, model.config)

    def cached_loader(
        config: BenchmarkConfig, dev: torch.device
    ) -> tuple[PreTrainedModel, set[int]]:
        return model, special_ids

    # Initialize CUDA libraries (cuBLAS handles, kernel caches) before the
    # first point, so its first measured runs do not pay for it.
    warm = torch.randint(0, 100, (1, 32), device=device)
    greedy_decode(model, warm, output_len=4)

    done = completed_points(results, tag)
    records = []
    for config in grid.configs(base):
        point = f"bs={config.batch_size} in={config.input_len} out={config.output_len}"
        if _point_key(config.to_dict()) in done:
            print(f"[already done] {point}")
            continue
        need = estimate(shape, config.batch_size, config.input_len, config.output_len)
        budget = available_bytes()
        # Weights are already resident, so compare only what this point adds.
        extra = need.total - need.weights
        if budget is not None and extra > budget:
            reason = f"estimated {extra / GIB:.2f} GiB > available {budget / GIB:.2f} GiB"
            record = skipped_record(config, reason, need.total)
        else:
            record = run_task(config, options, cached_loader)
            record["memory_estimate_bytes"] = need.total
        record["sweep_tag"] = tag
        append_record(results, record)
        records.append(record)
        summary = record.get("summary", {})
        itl = summary.get("itl_ms_p50", float("nan"))
        print(f"[{record['status']}] {point} itl_p50={itl:.2f}ms {record.get('reason', '')}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records
