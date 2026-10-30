"""Harness: turn one config into one classified, evidence-backed record.

Lifecycle of a run:

    preflight -> load model -> correctness check -> warmup -> measure
              -> stability check -> record (status + manifest + raw samples)

Whatever happens, the caller gets a record. Exceptions become ENV_ERROR or
PROGRAM_ERROR with the message and traceback kept as evidence.
"""

from __future__ import annotations

import dataclasses
import traceback
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedModel

from miniinferbench.bench import BenchmarkResult, default_device, load_model, run_benchmark
from miniinferbench.config import BenchmarkConfig
from miniinferbench.manifest import build_manifest, new_run_id
from miniinferbench.status import EnvironmentProblem, RunStatus, classify_exception

# A loader must return the model already placed on the given device, plus the
# tokenizer's special token ids (excluded from synthetic prompts).
ModelLoader = Callable[[BenchmarkConfig, torch.device], tuple[PreTrainedModel, set[int]]]

# Repeated end-to-end times varying more than this (as coefficient of
# variation) make a run UNSTABLE instead of OK.
DEFAULT_CV_THRESHOLD = 0.05


@dataclasses.dataclass(frozen=True, slots=True)
class HarnessOptions:
    require_cuda: bool = True
    cv_threshold: float = DEFAULT_CV_THRESHOLD


def _raw_samples(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "ttft_ms": [r.ttft_ms for r in result.runs],
        "itl_ms": [r.itl_ms for r in result.runs],
        "e2e_wall_ms": [r.e2e_wall_ms for r in result.runs],
        "peak_memory_bytes": [r.peak_memory_bytes for r in result.runs],
    }


def classify_result(result: BenchmarkResult, cv_threshold: float) -> tuple[RunStatus, str]:
    if not result.correctness.matches:
        seq, pos = result.correctness.first_mismatch or (-1, -1)
        return RunStatus.WRONG_RESULT, f"differs from generate() at sequence {seq}, token {pos}"
    cv = result.summary()["e2e_wall_ms_cv"]
    if cv > cv_threshold:
        return RunStatus.UNSTABLE, f"e2e CV {cv:.3f} exceeds threshold {cv_threshold:.3f}"
    return RunStatus.OK, ""


def run_task(
    config: BenchmarkConfig,
    options: HarnessOptions | None = None,
    loader: ModelLoader = load_model,
) -> dict[str, Any]:
    """Run one configuration and return its record (never raises for run failures)."""
    options = options or HarnessOptions()
    run_id = new_run_id()
    record: dict[str, Any] = {
        "run_id": run_id,
        "config": config.to_dict(),
        "manifest": build_manifest(run_id),
        "options": dataclasses.asdict(options),
    }
    try:
        if options.require_cuda and not torch.cuda.is_available():
            raise EnvironmentProblem("CUDA is not available (use --allow-cpu for a CPU smoke run)")
        device = default_device()
        model, special_ids = loader(config, device)
        record["model_revision"] = getattr(model.config, "_commit_hash", None)
        result = run_benchmark(config, model, special_ids, device)
    except Exception as exc:
        status = classify_exception(exc)
        record |= {
            "status": status.value,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return record
    status, reason = classify_result(result, options.cv_threshold)
    record |= {
        "status": status.value,
        "reason": reason,
        "input_fingerprint": result.input_fingerprint,
    }
    if result.runs:
        record |= {"summary": result.summary(), "raw": _raw_samples(result)}
    return record


@dataclasses.dataclass(frozen=True, slots=True)
class MetricDiff:
    name: str
    baseline: float
    candidate: float
    relative_change: float
    noise_band: float

    @property
    def significant(self) -> bool:
        return abs(self.relative_change) > self.noise_band


COMPARED_METRICS = ("ttft_ms_mean", "itl_ms_p50", "e2e_wall_ms_mean")


def compare_records(
    baseline: dict[str, Any], candidate: dict[str, Any], min_band: float = 0.02
) -> list[MetricDiff]:
    """Relative change per metric, judged against a noise band.

    The band is 3x the larger end-to-end CV of the two runs, but never below
    `min_band`: a change smaller than the runs' own spread is not evidence.
    """
    for rec in (baseline, candidate):
        if rec.get("status") not in (RunStatus.OK, RunStatus.UNSTABLE):
            raise ValueError(f"run {rec.get('run_id')} has status {rec.get('status')}; no metrics")
    if baseline["input_fingerprint"] != candidate["input_fingerprint"]:
        raise ValueError("runs used different inputs; they are not comparable")
    a, b = baseline["summary"], candidate["summary"]
    band = max(min_band, 3 * max(a["e2e_wall_ms_cv"], b["e2e_wall_ms_cv"]))
    diffs = []
    for name in COMPARED_METRICS:
        if name in a and name in b:
            change = (b[name] - a[name]) / a[name]
            diffs.append(MetricDiff(name, a[name], b[name], change, band))
    return diffs
