"""Profiler trace capture and a small analyzer for decode steps.

Capture runs in its own pass, never during measurement: the profiler adds
overhead (DeepInfer reports up to ~50% step inflation at batch size 1), so a
trace tells us *what* a step is made of, not how fast it is. We measure that
overhead explicitly by comparing traced step time with an untraced run.

The analyzer reads the exported Chrome trace JSON:
- `gpu_user_annotation` events are our `record_function` ranges placed on the
  GPU timeline; one per decode step.
- `kernel` events are GPU kernels.
- `cuda_runtime` events named like `cudaLaunchKernel` are CPU-side launches.

For each decode step it reports the span length, the time at least one kernel
was running (union of kernel intervals), and the kernel count. A low busy
fraction means the GPU sat idle inside the step, waiting for the CPU.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import PreTrainedModel

from miniinferbench.config import DecodeMode, Regression
from miniinferbench.runner import greedy_decode

DECODE_STEP = "decode_step"
PREFILL = "prefill"


def capture_trace(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    output_len: int,
    path: Path,
    regression: Regression = "none",
    decode_mode: DecodeMode = "eager",
) -> None:
    """Run one traced generation with each phase wrapped in record_function.

    Uses the same loop as the timed runs, so a trace shows exactly what was
    measured. "no_kv_cache" is not traced (its steps grow with sequence length).
    """
    if regression == "no_kv_cache":
        raise ValueError("tracing the no_kv_cache regression is not supported")
    activities = [ProfilerActivity.CPU]
    if input_ids.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        greedy_decode(
            model,
            input_ids,
            output_len,
            regression=regression,
            decode_mode=decode_mode,
            phase=record_function,
        )
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))


@dataclass(frozen=True, slots=True)
class StepStats:
    span_us: float
    busy_us: float
    kernels: int

    @property
    def busy_fraction(self) -> float:
        return self.busy_us / self.span_us if self.span_us else 0.0


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def step_stats(events: Sequence[dict[str, Any]], name: str = DECODE_STEP) -> list[StepStats]:
    spans = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and e.get("name") == name
    )
    kernels = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") == "kernel" and "dur" in e
    )
    stats = []
    for start, stop in spans:
        inside = [(max(a, start), min(b, stop)) for a, b in kernels if a < stop and b > start]
        stats.append(StepStats(stop - start, _union_length(inside), len(inside)))
    return stats


def top_kernels(events: Sequence[dict[str, Any]], limit: int = 10) -> list[tuple[str, float, int]]:
    """(name, total microseconds, count) for the most expensive kernels."""
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for e in events:
        if e.get("cat") == "kernel" and "dur" in e:
            total[e["name"]] += e["dur"]
            count[e["name"]] += 1
    ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [(name, us, count[name]) for name, us in ranked]


def count_launches(events: Sequence[dict[str, Any]]) -> int:
    return sum(
        1 for e in events if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name", "")
    )


def count_graph_launches(events: Sequence[dict[str, Any]]) -> int:
    """CUDA graph replays: one launch that runs a whole recorded sequence of kernels."""
    return sum(
        1 for e in events if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")
    )


def count_host_waits(events: Sequence[dict[str, Any]]) -> int:
    """CPU-side calls that block on the device: synchronizes and device-to-host copies."""
    return sum(
        1
        for e in events
        if e.get("cat") == "cuda_runtime"
        and ("Synchronize" in e.get("name", "") or "Memcpy" in e.get("name", ""))
    )


def analyze(trace_path: Path) -> dict[str, Any]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = data["traceEvents"] if isinstance(data, dict) else data
    steps = step_stats(events)
    if not steps:
        raise ValueError(f"no '{DECODE_STEP}' GPU annotations in {trace_path}")
    # Skip the first step: it pays one-off costs (allocations, cache growth).
    steady = steps[1:] or steps
    return {
        "decode_steps": len(steps),
        "step_span_us_median": statistics.median(s.span_us for s in steady),
        "step_busy_us_median": statistics.median(s.busy_us for s in steady),
        "step_busy_fraction_median": statistics.median(s.busy_fraction for s in steady),
        "kernels_per_step_median": statistics.median(s.kernels for s in steady),
        "kernel_launches_total": count_launches(events),
        "graph_launches_total": count_graph_launches(events),
        "host_wait_calls_total": count_host_waits(events),
        "top_kernels": [
            {"name": n, "total_us": us, "count": c} for n, us, c in top_kernels(events)
        ],
        "steps": [asdict(s) for s in steps],
    }
