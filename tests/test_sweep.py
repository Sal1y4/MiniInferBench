from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel

import miniinferbench.sweep as sweep
from miniinferbench.config import BenchmarkConfig
from miniinferbench.harness import HarnessOptions
from miniinferbench.results import read_records

from .tiny_model import make_tiny_model

BASE = BenchmarkConfig(model_name="tiny", warmup_runs=0, measure_runs=2)
GRID = sweep.SweepGrid(batch_sizes=(1, 2), input_lens=(8,), output_lens=(3, 4))
OPTIONS = HarnessOptions(require_cuda=False, cv_threshold=10.0)


@pytest.fixture(autouse=True)
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def loader(config: BenchmarkConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
        return make_tiny_model().to(device), {0}

    monkeypatch.setattr(sweep, "load_model", loader)


def test_grid_expands_to_all_points() -> None:
    points = [(c.batch_size, c.input_len, c.output_len) for c in GRID.configs(BASE)]
    assert points == [(1, 8, 3), (1, 8, 4), (2, 8, 3), (2, 8, 4)]


def test_sweep_records_every_point_and_resumes(tmp_path: Path) -> None:
    results = tmp_path / "sweep.jsonl"
    first = sweep.run_sweep(BASE, GRID, results, tag="t", options=OPTIONS)
    assert [r["status"] for r in first] == ["ok"] * 4
    assert all(r["sweep_tag"] == "t" for r in read_records(results))
    # Same tag again: nothing left to do.
    assert sweep.run_sweep(BASE, GRID, results, tag="t", options=OPTIONS) == []
    # A new tag starts a new curve.
    assert len(sweep.run_sweep(BASE, GRID, results, tag="u", options=OPTIONS)) == 4


def test_points_over_budget_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep, "available_bytes", lambda: 1)  # pretend almost nothing is free
    records: list[dict[str, Any]] = sweep.run_sweep(
        BASE, GRID, tmp_path / "s.jsonl", tag="t", options=OPTIONS
    )
    assert {r["status"] for r in records} == {"skipped"}
    assert "GiB" in records[0]["reason"]


def test_plots_from_a_sweep(tmp_path: Path) -> None:
    from miniinferbench.plots import plot_all

    grid = sweep.SweepGrid(batch_sizes=(1, 2), input_lens=(8, 512), output_lens=(3, 128))
    results = tmp_path / "sweep.jsonl"
    sweep.run_sweep(BASE, grid, results, tag="t", options=OPTIONS)
    written = plot_all(results, "t", tmp_path / "fig", output_len=128)
    assert len(written) == 6
    assert all(p.stat().st_size > 0 for p in written)


def test_mode_comparison_plots(tmp_path: Path) -> None:
    from dataclasses import replace

    from miniinferbench.plots import plot_modes

    grid = sweep.SweepGrid(batch_sizes=(1, 8), input_lens=(8, 512), output_lens=(128,))
    results = tmp_path / "sweep.jsonl"
    sweep.run_sweep(BASE, grid, results, tag="e", options=OPTIONS)
    graph_base = replace(BASE, decode_mode="cuda_graph")
    sweep.run_sweep(graph_base, grid, results, tag="g", options=OPTIONS)
    written = plot_modes(results, {"eager": "e", "cuda_graph": "g"}, tmp_path / "fig")
    assert [p.name for p in written] == [
        "modes_throughput_vs_batch.png",
        "modes_itl_vs_input.png",
    ]
