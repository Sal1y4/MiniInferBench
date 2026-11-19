"""Benchmark the generation path people actually call: `model.generate()`.

The rest of the harness times a hand-written decode loop, because a loop we
own is easy to instrument. Real code calls `generate()`, and what users feel
after upgrading a library is a change in `generate()`. This module measures
it with the same instruments (CUDA-event gaps per token, synchronized wall
clock, profiler trace features), without changing what it does:

- Per-token marks come from a logits processor that returns the scores
  untouched. A streamer would be the obvious hook, but `generate()` copies
  every token to the host for a streamer, which adds the very host sync we
  want to count.
- Timed runs come first, the traced run last, so no timing happens after the
  profiler was used (see `diagnosis.collect`).

It depends only on long-stable transformers API (`generate`,
`GenerationConfig`, `LogitsProcessor`), so the same code runs under old and
new library versions. A record holds the evidence in the shape the diagnosers
read, plus a fingerprint of the generated tokens: two versions that disagree
on the output are not comparable on speed.

`compare` turns two groups of records (two paths, or two library versions)
into one verdict: slower or not, by how much, and the most likely cause.
"""

from __future__ import annotations

import statistics
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from miniinferbench.config import BenchmarkConfig
from miniinferbench.manifest import build_manifest, new_run_id
from miniinferbench.status import EnvironmentProblem, RunStatus, classify_exception
from miniinferbench.timing import EventTimeline, WallClock
from miniinferbench.workload import fingerprint, make_prompts

GenPath = Literal["generate", "loop"]
PATHS: tuple[GenPath, ...] = ("generate", "loop")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
TASK = "generation_path"


def load_model(config: BenchmarkConfig, device: torch.device) -> tuple[Any, set[int]]:
    """Like `bench.load_model`, but without the `dtype=` keyword, whose name
    changed between transformers versions: load, then convert."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.to(device=device, dtype=DTYPES[config.dtype]).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, set(tokenizer.all_special_ids)


class _Ranges:
    """Consecutive profiler ranges: `next(name)` ends the open one and starts another."""

    def __init__(self) -> None:
        self._open: Any = None

    def next(self, name: str | None) -> None:
        if self._open is not None:
            self._open.__exit__(None, None, None)
            self._open = None
        if name is not None:
            self._open = record_function(name)
            self._open.__enter__()


def _marker(timeline: EventTimeline | None, ranges: _Ranges | None) -> Any:
    """A logits processor that marks the start of each decode step and
    returns the scores unchanged."""
    from transformers import LogitsProcessor

    class StepMarker(LogitsProcessor):  # type: ignore[misc]
        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            if timeline is not None:
                timeline.mark()
            if ranges is not None:
                ranges.next("decode_step")
            return scores

    return StepMarker()


@torch.inference_mode()
def run_generate(
    model: Any,
    input_ids: torch.Tensor,
    output_len: int,
    timeline: EventTimeline | None = None,
    ranges: _Ranges | None = None,
) -> torch.Tensor:
    """Greedy `generate()` of exactly `output_len` tokens; returns the new tokens.

    The marker is called once per generated token, right after that token's
    forward pass. So the first mark closes prefill, and the gap between two
    marks is one full loop iteration of `generate()`: token selection,
    stopping criteria, bookkeeping, and the next forward pass.
    """
    from transformers import GenerationConfig, LogitsProcessorList

    # Same settings as `runner.reference_generate`; see there for why EOS is
    # an empty list and why the model's own generation config is swapped out.
    config = GenerationConfig(
        do_sample=False,
        max_new_tokens=output_len,
        min_new_tokens=output_len,
        eos_token_id=[],
        pad_token_id=0,
    )
    saved = model.generation_config
    model.generation_config = config
    if timeline is not None:
        timeline.mark()
    if ranges is not None:
        ranges.next("prefill")
    try:
        out: torch.Tensor = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=config,
            logits_processor=LogitsProcessorList([_marker(timeline, ranges)]),
        )
    finally:
        model.generation_config = saved
        if ranges is not None:
            ranges.next(None)
    if timeline is not None:
        timeline.mark()
    return out[:, input_ids.shape[1] :]


def _timed(path: GenPath, model: Any, input_ids: torch.Tensor, config: BenchmarkConfig) -> Any:
    """One timed run: (ttft_ms, itl_ms, e2e_wall_ms, tokens)."""
    device = input_ids.device
    timeline = EventTimeline(device)
    wall = WallClock(device)
    wall.start()
    if path == "generate":
        tokens = run_generate(model, input_ids, config.output_len, timeline)
    else:
        from miniinferbench.runner import greedy_decode

        tokens = greedy_decode(model, input_ids, config.output_len, timeline)
    e2e = wall.stop()
    gaps = timeline.finish()
    # generate(): the last gap runs from the final marker to the return and
    # holds no forward pass, so it is not an inter-token gap.
    itl = gaps[1:-1] if path == "generate" else gaps[1:]
    return gaps[0], itl, e2e, tokens


def _traced(
    path: GenPath, model: Any, input_ids: torch.Tensor, config: BenchmarkConfig, trace_path: Path
) -> dict[str, Any]:
    from miniinferbench.trace import analyze, capture_trace

    if path == "loop":
        capture_trace(model, input_ids, config.output_len, trace_path)
    else:
        activities = [ProfilerActivity.CPU]
        if input_ids.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities) as prof:
            run_generate(model, input_ids, config.output_len, ranges=_Ranges())
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    return {k: v for k, v in analyze(trace_path).items() if k != "steps"}


def run_generation_task(
    config: BenchmarkConfig,
    path: GenPath,
    trace_path: Path,
    require_cuda: bool = True,
    keep_trace: bool = False,
    model_and_ids: tuple[Any, set[int]] | None = None,
) -> dict[str, Any]:
    """Time one generation path, then trace it; always returns a record."""
    run_id = new_run_id()
    record: dict[str, Any] = {
        "task": TASK,
        "run_id": run_id,
        "path": path,
        "config": config.to_dict(),
        "manifest": build_manifest(run_id),
    }
    try:
        if path not in PATHS:
            raise ValueError(f"unknown path {path!r}")
        if require_cuda and not torch.cuda.is_available():
            raise EnvironmentProblem("CUDA is not available (use --allow-cpu for a CPU smoke run)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, special_ids = model_and_ids or load_model(config, device)
        vocab = model.get_input_embeddings().weight.shape[0]
        prompts = make_prompts(config.batch_size, config.input_len, vocab, config.seed, special_ids)
        input_ids = prompts.to(device)
        for _ in range(config.warmup_runs):
            _timed(path, model, input_ids, config)
        runs = [_timed(path, model, input_ids, config) for _ in range(config.measure_runs)]
        tokens = runs[0][3]
        if tokens.shape != (config.batch_size, config.output_len):
            raise RuntimeError(f"expected {config.output_len} new tokens, got {tokens.shape}")
        trace = _traced(path, model, input_ids, config, trace_path)
        if not keep_trace:
            trace_path.unlink()
    except Exception as exc:
        record |= {
            "status": classify_exception(exc).value,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return record
    record |= {
        "status": RunStatus.OK.value,
        "reason": "",
        "input_fingerprint": fingerprint(prompts),
        "output_fingerprint": fingerprint(tokens),
        "evidence": {
            "timing": {
                "ttft_ms": [r[0] for r in runs],
                "itl_ms": [r[1] for r in runs],
                "e2e_wall_ms": [r[2] for r in runs],
            },
            "trace": trace,
        },
    }
    return record


# --- comparing two groups of records -----------------------------------------


def pooled_evidence(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Evidence of several records of the same thing, as one: all their timed
    runs together, and the trace of the first (trace features such as kernel
    counts barely vary between records; timings do)."""
    if not records:
        raise ValueError("no records to pool")
    timing: dict[str, list[Any]] = {"ttft_ms": [], "itl_ms": [], "e2e_wall_ms": []}
    for record in records:
        for key, values in timing.items():
            values.extend(record["evidence"]["timing"][key])
    return {"timing": timing, "trace": records[0]["evidence"]["trace"]}


def _per_token(evidence: dict[str, Any]) -> float:
    timing = evidence["timing"]
    return float(
        statistics.median(
            e2e / (len(gaps) + 1)
            for e2e, gaps in zip(timing["e2e_wall_ms"], timing["itl_ms"], strict=True)
        )
    )


def default_label(record: dict[str, Any]) -> str:
    return f"{record['path']}@{record['manifest']['env'].get('transformers')}"


def load_group(results: Path, label: str) -> list[dict[str, Any]]:
    """Usable generation records carrying `label`."""
    from miniinferbench.results import read_records

    return [
        r
        for r in read_records(results)
        if r.get("task") == TASK and r.get("label") == label and r.get("status") == "ok"
    ]


def compare(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Is the candidate group slower than the baseline group, and why?"""
    from miniinferbench.diagnosis import measured_slowdown, rule_diagnoser

    for name, group in (("baseline", baseline), ("candidate", candidate)):
        if not group:
            raise ValueError(f"no usable {name} records")
    inputs = {r["input_fingerprint"] for r in (*baseline, *candidate)}
    if len(inputs) != 1:
        raise ValueError("the two groups did not run the same input tokens")
    before, after = pooled_evidence(baseline), pooled_evidence(candidate)
    verdict = rule_diagnoser()(before, after)
    faster = rule_diagnoser()(after, before).regressed
    outputs = {r["output_fingerprint"] for r in (*baseline, *candidate)}
    trace_a, trace_b = before["trace"], after["trace"]
    steps_a, steps_b = trace_a["decode_steps"], trace_b["decode_steps"]
    return {
        "verdict": "slower" if verdict.regressed else "faster" if faster else "no change",
        "cause": verdict.cause if verdict.regressed else None,
        "notes": list(verdict.notes),
        "change_per_token": measured_slowdown(before, after),
        "ms_per_token": [_per_token(before), _per_token(after)],
        "timed_runs": [len(before["timing"]["e2e_wall_ms"]), len(after["timing"]["e2e_wall_ms"])],
        "same_output": len(outputs) == 1,
        "kernels_per_step": [
            trace_a["kernels_per_step_median"],
            trace_b["kernels_per_step_median"],
        ],
        "gpu_busy_us_per_step": [trace_a["step_busy_us_median"], trace_b["step_busy_us_median"]],
        "host_waits_per_step": [
            trace_a["host_wait_calls_total"] / steps_a,
            trace_b["host_wait_calls_total"] / steps_b,
        ],
    }


def format_comparison(result: dict[str, Any], baseline: str, candidate: str) -> str:
    before, after = result["ms_per_token"]
    lines = [
        f"**{candidate}** vs **{baseline}**: {result['verdict']}"
        + (f", most likely cause `{result['cause']}`" if result["cause"] else ""),
        "",
        f"| | {baseline} | {candidate} |",
        "| --- | --- | --- |",
        f"| Time per token (ms, median) | {before:.2f} | {after:.2f} "
        f"({result['change_per_token']:+.1%}) |",
        f"| Timed runs | {result['timed_runs'][0]} | {result['timed_runs'][1]} |",
        "| Kernels per decode step | {:.0f} | {:.0f} |".format(*result["kernels_per_step"]),
        "| GPU busy per step (us) | {:.0f} | {:.0f} |".format(*result["gpu_busy_us_per_step"]),
        "| Blocking host calls per step | {:.2f} | {:.2f} |".format(*result["host_waits_per_step"]),
        "",
        "Output tokens identical: " + ("yes" if result["same_output"] else "**no**"),
    ]
    if result["notes"]:
        lines += ["", "Evidence: " + "; ".join(result["notes"]) + "."]
    return "\n".join(lines)
