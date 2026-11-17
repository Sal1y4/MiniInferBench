import random
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel

import miniinferbench.bench as bench
from miniinferbench.cli import main
from miniinferbench.config import BenchmarkConfig
from miniinferbench.decision import RULES, fixed_threshold
from miniinferbench.noise import (
    build_report,
    detection_rate,
    false_positive_rate,
    format_report,
    load_groups,
    minimum_detectable,
    run_repeats,
    scale_record,
    simulated_detection_rate,
)
from miniinferbench.results import append_record, read_records

from .test_decision import make_record
from .tiny_model import make_tiny_model


def noisy_records(count: int, mean: float, spread: float, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [make_record([rng.gauss(mean, spread) for _ in range(5)]) for _ in range(count)]


def test_scale_record_scales_timings_only() -> None:
    record = make_record([100.0, 102.0])
    scaled = scale_record(record, 1.1)
    assert scaled["raw"]["e2e_wall_ms"] == pytest.approx([110.0, 112.2])
    assert scaled["raw"]["itl_ms"][0] == pytest.approx([11.0, 12.1, 33.0])
    assert scaled["summary"]["e2e_wall_ms_mean"] == pytest.approx(111.1)
    assert scaled["summary"]["e2e_wall_ms_cv"] == record["summary"]["e2e_wall_ms_cv"]
    assert record["raw"]["e2e_wall_ms"] == [100.0, 102.0]  # original untouched


def test_false_positive_rate_counts_flagged_aa_pairs() -> None:
    records = [make_record([100.0] * 5), make_record([100.0] * 5), make_record([120.0] * 5)]
    # 3 pairs; the two that include the 120 ms record differ by 20%.
    assert false_positive_rate(records, fixed_threshold(0.05), "e2e_wall_ms") == pytest.approx(
        2 / 3
    )
    with pytest.raises(ValueError):
        false_positive_rate(records[:1], fixed_threshold(0.05), "e2e_wall_ms")


def test_detection_rate_only_counts_the_right_direction() -> None:
    base = [make_record([100.0] * 5)]
    slowed = [make_record([110.0] * 5), make_record([90.0] * 5)]
    assert detection_rate(base, slowed, fixed_threshold(0.05), "e2e_wall_ms") == 0.5


def test_simulated_detection_grows_with_the_slowdown() -> None:
    records = noisy_records(8, mean=100.0, spread=1.0)
    rule = RULES["rank_test"]
    small = simulated_detection_rate(records, rule, "e2e_wall_ms", 0.002)
    large = simulated_detection_rate(records, rule, "e2e_wall_ms", 0.10)
    assert small < 0.5
    assert large == 1.0


def test_minimum_detectable_needs_power_from_there_on() -> None:
    curve = [(0.01, 0.2), (0.02, 0.9), (0.05, 0.7), (0.10, 1.0), (0.20, 1.0)]
    assert minimum_detectable(curve) == 0.10  # 0.02 does not count: 0.05 dips below
    assert minimum_detectable([(0.01, 0.1)]) is None


def test_report_covers_every_rule_and_metric() -> None:
    groups = {
        0: noisy_records(6, 100.0, 1.0),
        500: noisy_records(4, 105.0, 1.0, seed=1),
    }
    report = build_report(groups)
    e2e = report["metrics"]["e2e_wall_ms"]
    assert report["aa_pairs"] == 15
    assert e2e["injected"][500]["realized_slowdown"] == pytest.approx(0.05, abs=0.02)
    assert set(e2e["rules"]) == set(RULES)
    assert e2e["rules"]["rank_test"]["injected_detection"][500] > 0.9
    # Fixed 7% cannot see a 5% slowdown, however clean the runs.
    assert e2e["rules"]["fixed_7pct"]["injected_detection"][500] == 0.0
    text = format_report(report)
    assert "`cv_band`" in text and "500 us/step" in text
    with pytest.raises(ValueError, match="step_delay_us = 0"):
        build_report({500: groups[500]})


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    def loader(config: BenchmarkConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
        return make_tiny_model().to(device), {0}

    monkeypatch.setattr(bench, "load_model", loader)


BASE = BenchmarkConfig(
    model_name="tiny", batch_size=1, input_len=8, output_len=6, warmup_runs=0, measure_runs=3
)


def test_repeats_run_round_robin_and_resume(tmp_path: Path, tiny: None) -> None:
    results = tmp_path / "repeat.jsonl"
    first = run_repeats(BASE, [0, 2000], 2, results, "t", require_cuda=False)
    assert [r["config"]["step_delay_us"] for r in first] == [0, 2000, 0, 2000]
    assert [r["repeat_index"] for r in first] == [0, 0, 1, 1]
    # Asking for one more repeat only runs the missing records.
    more = run_repeats(BASE, [0, 2000], 3, results, "t", require_cuda=False)
    assert len(more) == 2
    groups = load_groups(results, "t")
    assert {delay: len(recs) for delay, recs in groups.items()} == {0: 3, 2000: 3}


def test_step_delay_slows_decode_and_keeps_the_output(tmp_path: Path, tiny: None) -> None:
    results = tmp_path / "repeat.jsonl"
    run_repeats(BASE, [0, 5000], 1, results, "t", require_cuda=False)
    plain, delayed = read_records(results)
    assert plain["status"] in ("ok", "unstable") and delayed["status"] in ("ok", "unstable")
    assert delayed["correctness"]["kind"] == "exact"
    # Every decode step busy-waits 5 ms, so no gap can be shorter. (Comparing
    # end-to-end times of the two records would depend on machine load.)
    assert min(gap for run in delayed["raw"]["itl_ms"] for gap in run) >= 5.0


def test_noise_command_prints_and_writes_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "repeat.jsonl"
    for record in noisy_records(4, 100.0, 1.0):
        append_record(results, record | {"repeat_tag": "t", "config": {"step_delay_us": 0}})
    for record in noisy_records(2, 110.0, 1.0, seed=1):
        append_record(results, record | {"repeat_tag": "t", "config": {"step_delay_us": 900}})
    out = tmp_path / "report.json"
    assert main(["noise", "--tag", "t", "--results", str(results), "--json", str(out)]) == 0
    assert "### e2e_wall_ms" in capsys.readouterr().out
    assert out.exists()


def test_detection_figure_is_written(tmp_path: Path) -> None:
    from miniinferbench.plots import plot_detection

    report = build_report({0: noisy_records(5, 100.0, 1.0)})
    path = plot_detection(report, "e2e_wall_ms", tmp_path, "synthetic")
    assert path.name == "detection_e2e_wall_ms.png" and path.stat().st_size > 0
