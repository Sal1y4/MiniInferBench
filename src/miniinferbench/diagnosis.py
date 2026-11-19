"""Diagnosis task set: can a tool tell that a run got slower, and why?

A *case* is a pair of runs of the same workload: a baseline, and a candidate
that differs by one hidden change. The change is either a fault (a slowdown
with a known root cause that leaves the output identical) or a control (no
change at all, or a change that is not a regression). Because the cause is
injected, the right answer is known.

A *diagnoser* sees only what was measured for the two runs, timings and
profiler trace features, never the configs. It answers two questions: did
the candidate regress, and what is the root cause. `score` then reports

- detection: share of faults flagged as regressions,
- false alarms: share of controls flagged as regressions,
- root cause: share of faults flagged *and* given the right cause.

A fault only counts if it actually slowed the run down. Some do not on some
workloads (a CPU stall is free while the GPU is the bottleneck), so a fault
case whose measured per-token time moved by less than `ABSORBED_BELOW` is
reported as absorbed and left out of the rates.

Cases are built from a suite file: every case is applied to every base
config. Bases are split into "dev" (rules may be tuned on these) and
"heldout" (a different model or shape, looked at only for the final score).

Two diagnosers are included as reference points: `rule_diagnoser` reads the
trace, `timing_only_diagnoser` does not. They are baselines to beat, not the
point of the task set.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniinferbench.config import BenchmarkConfig, ConfigError
from miniinferbench.decision import mann_whitney_p
from miniinferbench.results import append_record, read_records
from miniinferbench.trace import growth

# Root causes a diagnoser can name. "none" is the answer for a control.
CAUSES = (
    "none",
    "host_sync",  # the CPU blocks on the GPU every step
    "growing_work",  # each step does more GPU work than the one before
    "wasted_gpu_work",  # extra GPU computation whose result is not needed
    "launch_overhead",  # many extra tiny kernels; cost is launching them
    "allocator_churn",  # device memory is released and requested again
    "cpu_stall",  # the CPU is busy elsewhere while the GPU waits
)
UNKNOWN = "unknown"
ABSORBED_BELOW = 0.01
SPLITS = ("dev", "heldout")
USABLE = ("ok", "unstable")

Evidence = dict[str, Any]  # {"timing": {...}, "trace": {...}} for one run


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    cause: str
    overrides: dict[str, Any]
    description: str = ""

    @property
    def is_fault(self) -> bool:
        return self.cause != "none"


@dataclass(frozen=True, slots=True)
class Suite:
    bases: dict[str, list[BenchmarkConfig]]
    cases: list[Case]

    @classmethod
    def load(cls, path: Path) -> Suite:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            bases = {
                split: [BenchmarkConfig.from_dict(c) for c in data["bases"].get(split, [])]
                for split in SPLITS
            }
            cases = [Case(**c) for c in data["cases"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ConfigError(f"cannot read suite {path}: {exc}") from exc
        names = [c.name for c in cases]
        if len(set(names)) != len(names):
            raise ConfigError("case names must be unique")
        for case in cases:
            if case.cause not in CAUSES:
                raise ConfigError(f"case {case.name!r}: unknown cause {case.cause!r}")
            for configs in bases.values():
                for base in configs:
                    dataclasses.replace(base, **case.overrides)  # validates the combination
        return cls(bases=bases, cases=cases)


@dataclass(frozen=True, slots=True)
class Diagnosis:
    regressed: bool
    cause: str = "none"
    notes: tuple[str, ...] = ()


Diagnoser = Callable[[Evidence, Evidence], Diagnosis]


# --- collecting evidence -----------------------------------------------------


def make_evidence(timing: dict[str, Any], trace_report: dict[str, Any]) -> Evidence:
    """What a diagnoser may see of one run: raw timings and trace features.
    The config (which names the fault) is left out."""
    return {
        "timing": {k: timing[k] for k in ("ttft_ms", "itl_ms", "e2e_wall_ms")},
        "trace": {k: v for k, v in trace_report.items() if k != "steps"},
    }


def collect(
    suite: Suite,
    results: Path,
    tag: str,
    traces: Path,
    require_cuda: bool = True,
    keep_traces: bool = False,
) -> int:
    """Run every case on every base; returns the number of completed cases.

    Two passes over the suite. The first only times: each case gets its own
    fresh baseline run right before the candidate, so machine drift does not
    leak into the comparison and an unchanged control really is an A/A pair.
    The second only traces. They are kept apart so that no timed run happens
    in a process that has already used the profiler: the first run of this
    suite interleaved them, and 109 of its 140 timed runs came out unstable,
    with A/A pairs differing by up to 20%, while the same config timed without
    any profiling was stable. `scripts/profiler_aftereffect.py` tests that
    explanation in isolation.

    Resumable by tag: a rerun in a new process finishes missing timings
    first, then missing traces. Raw traces are large, so only their analysis
    is kept unless `keep_traces` is set.
    """
    import torch

    from miniinferbench.bench import load_model
    from miniinferbench.harness import HarnessOptions, run_task
    from miniinferbench.trace import analyze, capture_trace
    from miniinferbench.workload import make_prompts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = HarnessOptions(require_cuda=require_cuda)
    stored: dict[tuple[str, int, str], dict[str, Any]] = {}
    if results.exists():
        for row in read_records(results):
            if row.get("suite_tag") == tag:
                stored[(row["split"], row["base_index"], row["case"])] = row
    loaded_key: tuple[str, str] | None = None
    loaded: tuple[Any, set[int]] | None = None

    def bases() -> Any:
        nonlocal loaded, loaded_key
        for split in SPLITS:
            for index, base in enumerate(suite.bases[split]):
                key = (base.model_name, base.dtype)
                if key != loaded_key or loaded is None:
                    loaded = None  # one model in memory at a time
                    loaded, loaded_key = load_model(base, device), key
                yield split, index, base, loaded

    def timed(config: BenchmarkConfig, pair: tuple[Any, set[int]]) -> dict[str, Any]:
        record = run_task(config, options, _constant_loader(pair))
        out = {k: record[k] for k in ("run_id", "status", "reason", "manifest")}
        if record["status"] in USABLE:
            out["timing"] = record["raw"]
        return out

    def traced(config: BenchmarkConfig, pair: tuple[Any, set[int]], name: str) -> dict[str, Any]:
        model, special_ids = pair
        vocab = model.get_input_embeddings().weight.shape[0]
        prompts = make_prompts(config.batch_size, config.input_len, vocab, config.seed, special_ids)
        path = traces / f"{name}.trace.json"
        capture_trace(
            model,
            prompts.to(device),
            config.output_len,
            path,
            config.regression,
            config.decode_mode,
            config.step_delay_us,
            config.regression_level,
            config.step_delay_sync,
        )
        report = analyze(path)
        if not keep_traces:
            path.unlink()
        return report

    for split, index, base, pair in bases():
        for case in suite.cases:
            key = (split, index, case.name)
            if key in stored:
                continue
            baseline = timed(base, pair)
            candidate = timed(dataclasses.replace(base, **case.overrides), pair)
            row = {
                "suite_tag": tag,
                "stage": "timed",
                "split": split,
                "base_index": index,
                "base": base.to_dict(),
                "case": case.name,
                "cause": case.cause,
                "overrides": case.overrides,
                "usable": baseline["status"] in USABLE and candidate["status"] in USABLE,
                "baseline": baseline,
                "candidate": candidate,
            }
            append_record(results, row)
            stored[key] = row
            print(
                f"[timed {split}{index}] {case.name}: "
                f"baseline {baseline['status']}, candidate {candidate['status']}"
            )

    completed = 0
    for split, index, base, pair in bases():
        for case in suite.cases:
            row = stored[(split, index, case.name)]
            if row["stage"] == "complete":
                continue
            row = {k: v for k, v in row.items() if k != "schema_version"} | {"stage": "complete"}
            if row["usable"]:
                stem = f"{tag}-{split}{index}-{case.name}"
                configs = {
                    "baseline": base,
                    "candidate": dataclasses.replace(base, **case.overrides),
                }
                for side, config in configs.items():
                    run = dict(row[side])
                    run["evidence"] = make_evidence(
                        run.pop("timing"), traced(config, pair, f"{stem}-{side}")
                    )
                    row[side] = run
            append_record(results, row)
            completed += 1
            print(f"[traced {split}{index}] {case.name}")
    return completed


def _constant_loader(pair: tuple[Any, set[int]]) -> Any:
    return lambda config, device: pair


def load_cases(results: Path, tag: str) -> list[dict[str, Any]]:
    """Completed cases of one tag (timed and traced)."""
    return [
        r
        for r in read_records(results)
        if r.get("suite_tag") == tag and r.get("stage", "complete") == "complete"
    ]


# --- reference diagnosers ----------------------------------------------------


def _per_token_ms(timing: dict[str, Any]) -> list[float]:
    """End-to-end wall time per generated token, one value per measured run."""
    return [
        e2e / (len(gaps) + 1)
        for e2e, gaps in zip(timing["e2e_wall_ms"], timing["itl_ms"], strict=True)
    ]


def _mean_itl_ms(timing: dict[str, Any]) -> list[float]:
    return [statistics.fmean(gaps) for gaps in timing["itl_ms"]]


def _itl_growth(timing: dict[str, Any]) -> float:
    by_position = [statistics.median(gaps) for gaps in zip(*timing["itl_ms"], strict=True)]
    return growth(by_position[1:] or by_position)


def is_slower(a: Sequence[float], b: Sequence[float], min_effect: float, alpha: float) -> bool:
    """b is slower than a: a rank test rejects, and the medians differ by more
    than `min_effect` (so a tiny but consistent shift is not reported)."""
    change = statistics.median(b) / statistics.median(a) - 1.0
    if change <= min_effect:
        return False
    p, u = mann_whitney_p(a, b)
    return p < alpha and u > len(a) * len(b) / 2


def measured_slowdown(baseline: Evidence, candidate: Evidence) -> float:
    """Relative change of the median time per token, the larger of the GPU
    timeline's and the end-to-end wall clock's."""
    a, b = baseline["timing"], candidate["timing"]
    return max(
        statistics.median(f(b)) / statistics.median(f(a)) - 1.0
        for f in (_mean_itl_ms, _per_token_ms)
    )


def detect(baseline: Evidence, candidate: Evidence, min_effect: float, alpha: float) -> bool:
    """Slower per decode step on the GPU timeline, or per token end to end.
    The second catches slowdowns that only delay the CPU, which the gaps
    between GPU events do not always show."""
    a, b = baseline["timing"], candidate["timing"]
    return is_slower(_mean_itl_ms(a), _mean_itl_ms(b), min_effect, alpha) or is_slower(
        _per_token_ms(a), _per_token_ms(b), min_effect, alpha
    )


def _per_step(trace: dict[str, Any], total: float) -> float:
    return float(total) / float(trace["decode_steps"])


def _allocator_calls(trace: dict[str, Any]) -> float:
    calls = trace.get("runtime_calls", {})
    return float(sum(n for name, n in calls.items() if "Malloc" in name or "Free" in name))


def rule_diagnoser(min_effect: float = 0.03, alpha: float = 0.05) -> Diagnoser:
    """Detect from timings, then name a cause from what changed in the trace.

    The rules are checked in order; the first that fires wins.
      1. the same kernels take longer, or work grows
         along the sequence                            -> growing_work
      2. more allocator calls per step                 -> allocator_churn
      3. more kernels per step, each at least as heavy
         as the average kernel before                  -> wasted_gpu_work
      4. more kernels per step, lighter than average   -> launch_overhead
      5. more blocking host calls per step             -> host_sync
      6. none of the above, CPU side of the step longer -> cpu_stall

    Kernel-count rules come before the host-call rule because extra GPU work
    can bring its own blocking calls (a sort does), while a plain host copy
    adds no kernels. Thresholds were set on the dev split of run
    5070-diag-v1.
    """

    def diagnose(baseline: Evidence, candidate: Evidence) -> Diagnosis:
        if not detect(baseline, candidate, min_effect, alpha):
            return Diagnosis(regressed=False)
        a, b = baseline["trace"], candidate["trace"]
        notes = []

        kernels_a, kernels_b = a["kernels_per_step_median"], b["kernels_per_step_median"]
        busy_a, busy_b = a["step_busy_us_median"], b["step_busy_us_median"]
        per_kernel_a, per_kernel_b = busy_a / kernels_a, busy_b / kernels_b
        grow_a, grow_b = a["step_busy_growth"], b["step_busy_growth"]
        notes.append(f"busy time per kernel {per_kernel_a:.2f} -> {per_kernel_b:.2f} us")
        notes.append(f"busy time growth along the sequence {grow_a:.2f} -> {grow_b:.2f}")
        if per_kernel_b >= 1.3 * per_kernel_a or (grow_b > 1.3 and grow_b > 1.2 * grow_a):
            return Diagnosis(True, "growing_work", tuple(notes))

        alloc_a = _per_step(a, _allocator_calls(a))
        alloc_b = _per_step(b, _allocator_calls(b))
        notes.append(f"allocator calls per step {alloc_a:.2f} -> {alloc_b:.2f}")
        if alloc_b - alloc_a >= 0.5:
            return Diagnosis(True, "allocator_churn", tuple(notes))

        added_kernels = kernels_b - kernels_a
        notes.append(f"kernels per step {kernels_a:.0f} -> {kernels_b:.0f}")
        if added_kernels >= 2:
            per_added = (busy_b - busy_a) / added_kernels
            notes.append(f"each added kernel is busy {per_added:.2f} us")
            cause = "wasted_gpu_work" if per_added >= per_kernel_a else "launch_overhead"
            return Diagnosis(True, cause, tuple(notes))

        waits_a = _per_step(a, a["host_wait_calls_total"])
        waits_b = _per_step(b, b["host_wait_calls_total"])
        notes.append(f"blocking host calls per step {waits_a:.2f} -> {waits_b:.2f}")
        if waits_b - waits_a >= 0.5:
            return Diagnosis(True, "host_sync", tuple(notes))

        cpu_a, cpu_b = a.get("step_cpu_us_median"), b.get("step_cpu_us_median")
        if cpu_a and cpu_b:
            notes.append(f"CPU side of a step {cpu_a:.0f} -> {cpu_b:.0f} us")
            if cpu_b > 1.02 * cpu_a:
                return Diagnosis(True, "cpu_stall", tuple(notes))
        return Diagnosis(True, UNKNOWN, tuple(notes))

    return diagnose


def timing_only_diagnoser(min_effect: float = 0.03, alpha: float = 0.05) -> Diagnoser:
    """Same detection, no trace. The only cause visible in timings alone is
    step time that grows along the sequence."""

    def diagnose(baseline: Evidence, candidate: Evidence) -> Diagnosis:
        if not detect(baseline, candidate, min_effect, alpha):
            return Diagnosis(regressed=False)
        grow_a, grow_b = _itl_growth(baseline["timing"]), _itl_growth(candidate["timing"])
        note = (f"inter-token latency growth along the sequence {grow_a:.2f} -> {grow_b:.2f}",)
        if grow_b > 1.3 and grow_b > 1.2 * grow_a:
            return Diagnosis(True, "growing_work", note)
        return Diagnosis(True, UNKNOWN, note)

    return diagnose


DIAGNOSERS: dict[str, Diagnoser] = {
    "rules_on_trace": rule_diagnoser(),
    "timing_only": timing_only_diagnoser(),
}


# --- scoring -----------------------------------------------------------------


@dataclass(slots=True)
class Score:
    faults: int = 0
    detected: int = 0
    cause_correct: int = 0
    absorbed: int = 0
    controls: int = 0
    false_alarms: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "detection_rate": self.detected / self.faults if self.faults else None,
            "root_cause_rate": self.cause_correct / self.faults if self.faults else None,
            "false_alarm_rate": self.false_alarms / self.controls if self.controls else None,
        }


def score(cases: Sequence[dict[str, Any]], diagnoser: Diagnoser) -> dict[str, Any]:
    """Score one diagnoser per split. Cases whose runs failed are counted as
    `excluded` and faults without a measurable slowdown as `absorbed`; neither
    is scored. A fault that is not flagged counts against both detection and
    root cause."""
    scores = {split: Score() for split in SPLITS}
    rows = []
    excluded = 0
    for case in cases:
        if not case["usable"]:
            excluded += 1
            continue
        before, after = case["baseline"]["evidence"], case["candidate"]["evidence"]
        verdict = diagnoser(before, after)
        answer = verdict.cause if verdict.regressed else "none"
        s = scores[case["split"]]
        truth = case["cause"]
        slowdown = measured_slowdown(before, after)
        row = {
            "split": case["split"],
            "base_index": case["base_index"],
            "case": case["case"],
            "truth": truth,
            "answer": answer,
            "measured_slowdown": slowdown,
            "absorbed": truth != "none" and slowdown < ABSORBED_BELOW,
            "notes": list(verdict.notes),
        }
        rows.append(row)
        if row["absorbed"]:
            s.absorbed += 1
            continue
        s.confusion.setdefault(truth, {}).setdefault(answer, 0)
        s.confusion[truth][answer] += 1
        if truth == "none":
            s.controls += 1
            s.false_alarms += verdict.regressed
        else:
            s.faults += 1
            s.detected += verdict.regressed
            s.cause_correct += answer == truth
    return {
        "excluded": excluded,
        "splits": {split: s.as_dict() for split, s in scores.items()},
        "cases": rows,
    }


def _ratio(hit: int, total: int) -> str:
    return f"{hit}/{total} ({hit / total:.0%})" if total else "n/a"


def format_scores(report: dict[str, dict[str, Any]]) -> str:
    """Markdown: one summary table, then per diagnoser the cases it got wrong."""
    lines = [
        "| Diagnoser | Split | Detected | Root cause right | False alarms on controls "
        "| Faults absorbed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in report.items():
        for split, s in result["splits"].items():
            lines.append(
                f"| `{name}` | {split} | {_ratio(s['detected'], s['faults'])} | "
                f"{_ratio(s['cause_correct'], s['faults'])} | "
                f"{_ratio(s['false_alarms'], s['controls'])} | {s['absorbed']} |"
            )
    for name, result in report.items():
        scored = [r for r in result["cases"] if not r["absorbed"]]
        wrong = [r for r in scored if r["answer"] != r["truth"]]
        lines += ["", f"### `{name}`: {len(wrong)} wrong, {result['excluded']} excluded", ""]
        if wrong:
            lines += [
                "| Split | Base | Case | Measured slowdown | Truth | Answer |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            lines += [
                f"| {r['split']} | {r['base_index']} | {r['case']} | "
                f"{r['measured_slowdown']:+.1%} | {r['truth']} | {r['answer']} |"
                for r in wrong
            ]
        absorbed = [r for r in result["cases"] if r["absorbed"]]
        if absorbed:
            names = ", ".join(f"{r['case']} ({r['split']} {r['base_index']})" for r in absorbed)
            lines += ["", f"Absorbed (no measurable slowdown, not scored): {names}"]
    return "\n".join(lines)
