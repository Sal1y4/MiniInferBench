import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel

import miniinferbench.bench as bench
import miniinferbench.trace as trace
from miniinferbench.cli import main
from miniinferbench.config import BenchmarkConfig, ConfigError
from miniinferbench.diagnosis import (
    DIAGNOSERS,
    UNKNOWN,
    Case,
    Suite,
    collect,
    format_scores,
    is_slower,
    load_cases,
    make_evidence,
    rule_diagnoser,
    score,
    timing_only_diagnoser,
)

from .tiny_model import make_tiny_model

STEPS = 10


def evidence(
    itl: float = 10.0,
    *,
    grow: float = 0.0,
    span: float = 10000.0,
    busy: float = 4000.0,
    kernels: float = 100.0,
    host_waits: int = 0,
    busy_growth: float = 1.0,
    runtime_calls: dict[str, int] | None = None,
    runs: int = 5,
) -> dict[str, Any]:
    """Synthetic evidence for one run: `runs` measured runs of STEPS gaps of
    about `itl` ms (run-to-run jitter 0.1%), rising by `grow` ms per step."""
    gaps = [[itl * (1 + 0.001 * r) + grow * s for s in range(STEPS)] for r in range(runs)]
    return {
        "timing": {
            "ttft_ms": [20.0] * runs,
            "itl_ms": gaps,
            "e2e_wall_ms": [20.0 + sum(g) for g in gaps],
        },
        "trace": {
            "decode_steps": STEPS,
            "step_span_us_median": span,
            "step_busy_us_median": busy,
            "kernels_per_step_median": kernels,
            "host_wait_calls_total": host_waits,
            "step_busy_growth": busy_growth,
            "runtime_calls": runtime_calls or {},
        },
    }


BASE = evidence()


def test_is_slower_needs_both_significance_and_effect() -> None:
    a = [10.0, 10.1, 10.2, 10.3, 10.4]
    assert is_slower(a, [v * 1.10 for v in a], min_effect=0.03, alpha=0.05)
    assert not is_slower(a, [v * 1.02 for v in a], min_effect=0.03, alpha=0.05)  # too small
    assert not is_slower(a, [9.0, 10.2, 12.0, 14.0, 9.5], min_effect=0.03, alpha=0.05)  # noisy
    assert not is_slower(a, [v * 0.8 for v in a], min_effect=0.03, alpha=0.05)  # faster


def test_unchanged_run_is_not_flagged() -> None:
    verdict = rule_diagnoser()(BASE, evidence(10.05))
    assert not verdict.regressed and verdict.cause == "none"


@pytest.mark.parametrize(
    ("candidate", "cause"),
    [
        (evidence(12.0, grow=2.0, busy_growth=3.0), "growing_work"),
        (evidence(12.0, span=12000.0, host_waits=STEPS), "host_sync"),
        (
            evidence(12.0, span=12000.0, runtime_calls={"cudaMalloc": STEPS, "cudaFree": STEPS}),
            "allocator_churn",
        ),
        (evidence(12.0, span=12000.0, busy=5800.0, kernels=110.0), "wasted_gpu_work"),
        (evidence(12.0, span=12000.0, busy=4100.0, kernels=160.0), "launch_overhead"),
        (evidence(12.0, span=12000.0), "cpu_stall"),
        (evidence(12.0, span=12000.0, busy=6000.0), UNKNOWN),
    ],
)
def test_rule_diagnoser_names_the_cause(candidate: dict[str, Any], cause: str) -> None:
    verdict = rule_diagnoser()(BASE, candidate)
    assert verdict.regressed
    assert verdict.cause == cause
    assert verdict.notes


def test_timing_only_diagnoser_sees_growth_and_nothing_else() -> None:
    diagnose = timing_only_diagnoser()
    assert diagnose(BASE, evidence(12.0, grow=2.0)).cause == "growing_work"
    assert diagnose(BASE, evidence(12.0, host_waits=STEPS)).cause == UNKNOWN
    assert not diagnose(BASE, evidence(10.0)).regressed


def test_longer_output_is_not_a_regression() -> None:
    longer = evidence()
    longer["timing"]["itl_ms"] = [g + g for g in longer["timing"]["itl_ms"]]
    longer["timing"]["e2e_wall_ms"] = [20.0 + sum(g) for g in longer["timing"]["itl_ms"]]
    assert not rule_diagnoser()(BASE, longer).regressed


def case_record(split: str, name: str, cause: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "base_index": 0,
        "case": name,
        "cause": cause,
        "usable": True,
        "baseline": {"evidence": BASE},
        "candidate": {"evidence": candidate},
    }


def test_score_counts_detection_cause_and_false_alarms() -> None:
    cases = [
        case_record("dev", "aa", "none", evidence(10.02)),
        case_record("dev", "false_alarm", "none", evidence(12.0, span=12000.0)),
        case_record("dev", "sync", "host_sync", evidence(12.0, span=12000.0, host_waits=STEPS)),
        case_record("dev", "missed", "cpu_stall", evidence(10.2)),
        case_record("dev", "absorbed", "cpu_stall", evidence(10.01)),
        case_record("heldout", "wrong_cause", "host_sync", evidence(12.0, span=12000.0)),
        case_record("heldout", "broken", "host_sync", BASE) | {"usable": False},
    ]
    result = score(cases, rule_diagnoser())
    dev, heldout = result["splits"]["dev"], result["splits"]["heldout"]
    assert (dev["controls"], dev["false_alarms"]) == (2, 1)
    assert (dev["faults"], dev["detected"], dev["cause_correct"]) == (2, 1, 1)
    assert dev["confusion"]["cpu_stall"] == {"none": 1}
    assert (heldout["faults"], heldout["detected"], heldout["cause_correct"]) == (1, 1, 0)
    assert heldout["false_alarm_rate"] is None
    assert result["excluded"] == 1
    assert dev["absorbed"] == 1
    text = format_scores({"rules_on_trace": result})
    assert "| `rules_on_trace` | dev | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 1 |" in text
    assert "Absorbed (no measurable slowdown, not scored): absorbed (dev 0)" in text
    assert "wrong_cause" in text and "| dev | 0 | sync |" not in text


def write_suite(path: Path, cases: list[dict[str, Any]]) -> Path:
    base = BenchmarkConfig(
        model_name="tiny", batch_size=2, input_len=8, output_len=6, warmup_runs=0, measure_runs=2
    )
    path.write_text(json.dumps({"bases": {"dev": [base.to_dict()]}, "cases": cases}))
    return path


def test_suite_rejects_bad_cases(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown cause"):
        Suite.load(
            write_suite(tmp_path / "a.json", [{"name": "x", "cause": "gremlins", "overrides": {}}])
        )
    with pytest.raises(ConfigError, match="unique"):
        same = {"name": "x", "cause": "none", "overrides": {}}
        Suite.load(write_suite(tmp_path / "b.json", [same, same]))
    with pytest.raises(ConfigError):
        bad = {"name": "x", "cause": "none", "overrides": {"batch_size": 0}}
        Suite.load(write_suite(tmp_path / "c.json", [bad]))


def test_shipped_suite_loads() -> None:
    suite = Suite.load(Path("configs/diagnosis_suite.json"))
    assert suite.bases["dev"] and suite.bases["heldout"]
    assert {c.cause for c in suite.cases} >= {"none", "host_sync", "cpu_stall", "growing_work"}
    assert isinstance(suite.cases[0], Case)


def test_evidence_hides_the_config() -> None:
    record = {"config": {"regression": "host_sync_each_step"}, "raw": BASE["timing"] | {"x": 1}}
    out = make_evidence(record, {"decode_steps": 3, "steps": [1, 2, 3]})
    assert out == {"timing": BASE["timing"], "trace": {"decode_steps": 3}}
    assert "host_sync" not in json.dumps(out)


def test_collect_and_diagnose_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # CPU traces carry no GPU events, so the analysis is stubbed; everything
    # else (runs, pairing, resume, scoring, CLI) is real.
    model = make_tiny_model()

    def tiny(config: BenchmarkConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
        return model, set()

    monkeypatch.setattr(bench, "load_model", tiny)
    monkeypatch.setattr(trace, "analyze", lambda path: BASE["trace"] | {"steps": []})
    suite_path = write_suite(
        tmp_path / "suite.json",
        [
            {"name": "aa", "cause": "none", "overrides": {}},
            {
                "name": "sync",
                "cause": "host_sync",
                "overrides": {"regression": "host_sync_each_step"},
            },
        ],
    )
    results, traces = tmp_path / "d.jsonl", tmp_path / "traces"
    assert collect(Suite.load(suite_path), results, "t", traces, require_cuda=False) == 2
    assert collect(Suite.load(suite_path), results, "t", traces, require_cuda=False) == 0  # resumed
    cases = load_cases(results, "t")
    assert [c["case"] for c in cases] == ["aa", "sync"]
    assert all(c["usable"] for c in cases)
    assert "config" not in cases[1]["candidate"]["evidence"]
    assert not list(traces.glob("*.json"))  # raw traces are not kept by default
    for diagnoser in DIAGNOSERS.values():
        assert score(cases, diagnoser)["excluded"] == 0

    assert main(["diagnose", "--tag", "t", "--results", str(results)]) == 0
    assert "| `rules_on_trace` | dev |" in capsys.readouterr().out
    assert main(["diagnose", "--tag", "missing", "--results", str(results)]) == 1
