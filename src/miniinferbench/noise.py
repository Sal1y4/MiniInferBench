"""How far can a comparison be trusted? Measure the harness's own error rates.

Two questions, answered from repeated runs of one configuration:

- False alarms. Runs of the *same* code and config are compared pairwise
  (an A/A test). Any "changed" verdict is wrong by construction, so the
  share of flagged pairs is the rule's false-positive rate on this machine.
- Sensitivity. A slowdown of known size is added and the share of pairs
  flagged as slower is the detection rate. The slowdown comes from two
  sources, reported separately:
    * injected: real runs with `step_delay_us` (a CPU busy-wait per decode
      step). The realized slowdown is measured, not assumed, and that
      matters: on a GPU-bound workload a plain busy-wait is absorbed while
      the GPU works through its queue (the first run on the RTX 5070 showed
      +0.0% at 750 us per step). Set `step_delay_sync` in the base config so
      the delay lands on every step.
    * simulated: A/A records with every timing multiplied by (1 + s). This
      gives a full curve for free but assumes a slowdown only shifts the
      timings and leaves their spread unchanged.

Repeats are run round-robin over the delays (0, d1, d2, 0, d1, d2, ...), so
slow drift of the machine affects every group alike.
"""

from __future__ import annotations

import copy
import gc
import itertools
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from miniinferbench.config import BenchmarkConfig
from miniinferbench.decision import METRICS, RULES, Record, Rule, run_samples
from miniinferbench.results import append_record, read_records

USABLE = ("ok", "unstable")
SIMULATED_SLOWDOWNS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
_TIME_SUFFIXES = ("_ms_mean", "_ms_p50", "_ms_p95", "_ms_p99")


def load_groups(results: Path, tag: str) -> dict[int, list[Record]]:
    """Usable records of one repeat tag, grouped by `step_delay_us`."""
    groups: dict[int, list[Record]] = {}
    for record in read_records(results):
        if record.get("repeat_tag") == tag and record.get("status") in USABLE:
            groups.setdefault(record["config"].get("step_delay_us", 0), []).append(record)
    return groups


def run_repeats(
    base: BenchmarkConfig,
    delays_us: Sequence[int],
    repeats: int,
    results: Path,
    tag: str,
    require_cuda: bool = True,
) -> list[Record]:
    """Run `repeats` records per delay, round-robin; resumes an unfinished tag."""
    import torch

    from miniinferbench.bench import load_model
    from miniinferbench.harness import HarnessOptions, ModelLoader, run_task
    from miniinferbench.runner import greedy_decode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, special_ids = load_model(base, device)
    cached: ModelLoader = lambda config, dev: (model, special_ids)  # noqa: E731
    greedy_decode(model, torch.randint(0, 100, (1, 32), device=device), output_len=4)

    done: dict[int, int] = {}
    if results.exists():
        for record in read_records(results):
            if record.get("repeat_tag") == tag:
                delay = record["config"].get("step_delay_us", 0)
                done[delay] = done.get(delay, 0) + 1
    options = HarnessOptions(require_cuda=require_cuda)
    records = []
    for index in range(repeats):
        for delay in delays_us:
            if done.get(delay, 0) > index:
                continue
            record = run_task(replace(base, step_delay_us=delay), options, cached)
            record |= {"repeat_tag": tag, "repeat_index": index}
            append_record(results, record)
            records.append(record)
            e2e = record.get("summary", {}).get("e2e_wall_ms_mean", float("nan"))
            print(f"[{record['status']}] repeat {index} delay={delay}us e2e={e2e:.1f}ms")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def scale_record(record: Record, factor: float) -> Record:
    """A copy of `record` with every timing multiplied by `factor`."""
    out = copy.deepcopy(record)
    raw = out["raw"]
    raw["ttft_ms"] = [v * factor for v in raw["ttft_ms"]]
    raw["e2e_wall_ms"] = [v * factor for v in raw["e2e_wall_ms"]]
    raw["itl_ms"] = [[v * factor for v in run] for run in raw["itl_ms"]]
    for key, value in out["summary"].items():
        if key.endswith(_TIME_SUFFIXES):
            out["summary"][key] = value * factor
    return out


def false_positive_rate(records: Sequence[Record], rule: Rule, metric: str) -> float:
    """Share of A/A pairs the rule calls changed (in either direction)."""
    pairs = list(itertools.combinations(records, 2))
    if not pairs:
        raise ValueError("an A/A test needs at least two records")
    return sum(rule(a, b, metric) != 0 for a, b in pairs) / len(pairs)


def detection_rate(
    baseline: Sequence[Record], slowed: Sequence[Record], rule: Rule, metric: str
) -> float:
    """Share of (baseline, slowed) pairs the rule calls slower."""
    pairs = list(itertools.product(baseline, slowed))
    if not pairs:
        raise ValueError("need at least one baseline and one slowed record")
    return sum(rule(a, b, metric) == 1 for a, b in pairs) / len(pairs)


def simulated_detection_rate(
    records: Sequence[Record], rule: Rule, metric: str, slowdown: float
) -> float:
    """Detection rate when one record of each A/A pair is slowed by `slowdown`."""
    scaled = [scale_record(r, 1.0 + slowdown) for r in records]
    pairs = [(i, j) for i in range(len(records)) for j in range(len(records)) if i < j]
    if not pairs:
        raise ValueError("need at least two records")
    return sum(rule(records[i], scaled[j], metric) == 1 for i, j in pairs) / len(pairs)


def minimum_detectable(curve: Iterable[tuple[float, float]], power: float = 0.8) -> float | None:
    """Smallest slowdown from which the detection rate stays at or above `power`."""
    best = None
    for slowdown, rate in sorted(curve, reverse=True):
        if rate < power:
            break
        best = slowdown
    return best


def _group_median(records: Sequence[Record], metric: str) -> float:
    return statistics.median(v for r in records for v in run_samples(r, metric))


def build_report(
    groups: dict[int, list[Record]],
    rules: dict[str, Rule] | None = None,
    metrics_: Sequence[str] = METRICS,
    slowdowns: Sequence[float] = SIMULATED_SLOWDOWNS,
) -> dict[str, Any]:
    """False-positive rate, detection rates and minimum detectable slowdown,
    for every rule and metric."""
    rules = rules or RULES
    baseline = groups.get(0, [])
    if len(baseline) < 2:
        raise ValueError("need at least two usable records with step_delay_us = 0")
    report: dict[str, Any] = {
        "baseline_records": len(baseline),
        "aa_pairs": len(baseline) * (len(baseline) - 1) // 2,
        "runs_per_record": len(baseline[0]["raw"]["e2e_wall_ms"]),
        "unstable_records": sum(r["status"] == "unstable" for r in baseline),
        "metrics": {},
    }
    for metric in metrics_:
        base_median = _group_median(baseline, metric)
        injected = {
            delay: {
                "records": len(recs),
                "realized_slowdown": _group_median(recs, metric) / base_median - 1.0,
            }
            for delay, recs in sorted(groups.items())
            if delay != 0
        }
        per_rule: dict[str, Any] = {}
        for name, rule in rules.items():
            simulated = [
                (s, simulated_detection_rate(baseline, rule, metric, s)) for s in slowdowns
            ]
            per_rule[name] = {
                "false_positive_rate": false_positive_rate(baseline, rule, metric),
                "simulated_detection": [list(point) for point in simulated],
                "minimum_detectable_simulated": minimum_detectable(simulated),
                "injected_detection": {
                    delay: detection_rate(baseline, groups[delay], rule, metric)
                    for delay in injected
                },
            }
        report["metrics"][metric] = {
            "baseline_median": base_median,
            "injected": injected,
            "rules": per_rule,
        }
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_report(report: dict[str, Any]) -> str:
    """The report as Markdown tables, one per metric."""
    lines = [
        f"{report['baseline_records']} baseline records "
        f"({report['unstable_records']} unstable), {report['runs_per_record']} measured runs "
        f"each, {report['aa_pairs']} A/A pairs.",
    ]
    for metric, data in report["metrics"].items():
        delays = list(data["injected"])
        head = ["Rule", "False positives (A/A)", "Min detectable (simulated, 80%)"]
        head += [
            f"Detected at +{data['injected'][d]['realized_slowdown']:.1%} ({d} us/step)"
            for d in delays
        ]
        lines += ["", f"### {metric} (baseline median {data['baseline_median']:.2f} ms)", ""]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + " --- |" * len(head))
        for name, rule in data["rules"].items():
            cells = [
                f"`{name}`",
                _pct(rule["false_positive_rate"]),
                _pct(rule["minimum_detectable_simulated"]),
            ]
            cells += [_pct(rule["injected_detection"][d]) for d in delays]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
