import math
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel

from miniinferbench.cli import main
from miniinferbench.config import ConfigError
from miniinferbench.results import read_records
from miniinferbench.status import RunStatus
from miniinferbench.training import (
    PHASES,
    Probe,
    TrainConfig,
    TrainOptions,
    evaluate_gate,
    forward_flops_per_token,
    reference_losses,
    run_train_task,
    train_steps,
)
from miniinferbench.workload import make_prompts

from .tiny_model import make_tiny_model

CONFIG = TrainConfig(
    model_name="tiny",
    batch_size=2,
    seq_len=16,
    precision="float32",
    learning_rate=1e-3,
    check_steps=3,
    warmup_steps=1,
    measure_runs=2,
    steps_per_run=3,
)
CPU = TrainOptions(require_cuda=False, cv_threshold=10.0)


def tiny_loader(config: TrainConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
    return make_tiny_model().to(device), {0}


def run(**changes: Any) -> dict[str, Any]:
    import dataclasses

    return run_train_task(dataclasses.replace(CONFIG, **changes), CPU, tiny_loader)


def test_config_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    CONFIG.save(path)
    assert TrainConfig.load(path) == CONFIG
    with pytest.raises(ConfigError, match="seq_len"):
        TrainConfig(seq_len=1)
    with pytest.raises(ConfigError, match="unknown fault"):
        TrainConfig(fault="typo")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="unknown config fields"):
        TrainConfig.from_dict({"output_len": 4})


def test_ok_run_has_gate_evidence_and_phase_timings() -> None:
    record = run()
    assert record["status"] == RunStatus.OK
    assert record["task"] == "train_step"
    gate = record["gate"]
    assert gate["kind"] in ("exact", "within_tolerance")
    assert gate["losses"] == pytest.approx(gate["reference_losses"], rel=1e-3)
    assert gate["losses"][-1] < gate["losses"][0]  # same batch every step: loss must fall
    assert all(norm > 0 for norm in gate["grad_norms"])
    raw = record["raw"]
    assert len(raw["e2e_wall_ms"]) == 2
    assert set(raw["phase_ms"][0]) == set(PHASES)
    assert all(len(raw["phase_ms"][0][phase]) == 3 for phase in PHASES)
    summary = record["summary"]
    assert summary["step_samples"] == 6
    assert summary["tokens_per_s"] > 0 and summary["achieved_tflops"] > 0
    assert "mfu" not in summary  # no peak given, no percentage reported


def test_mfu_is_reported_only_with_a_peak() -> None:
    summary = run(peak_tflops=100.0)["summary"]
    assert summary["mfu"] == pytest.approx(summary["achieved_tflops"] / 100.0)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("frozen_layers", "parameters are frozen"),
        ("zero_lr", "trainable parameters never changed"),
    ],
)
def test_loops_that_do_not_train_are_never_timed(fault: str, expected: str) -> None:
    record = run(fault=fault)
    assert record["status"] == RunStatus.WRONG_RESULT
    assert expected in record["reason"]
    assert "summary" not in record and "raw" not in record


def test_zero_lr_also_fails_the_loss_comparison() -> None:
    gate = run(fault="zero_lr")["gate"]
    assert gate["kind"] == "mismatch"
    assert len(set(gate["losses"])) == 1  # nothing moves
    assert gate["reference_losses"][-1] < gate["reference_losses"][0]


def test_model_is_restored_after_a_run() -> None:
    model = make_tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    record = run_train_task(CONFIG, CPU, lambda c, d: (model, {0}))
    assert record["status"] == RunStatus.OK
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not model.training
    assert all(p.requires_grad and p.grad is None for p in model.parameters())


def test_frozen_fault_is_undone() -> None:
    model = make_tiny_model()
    run_train_task(
        TrainConfig(**(CONFIG.to_dict() | {"fault": "frozen_layers"})),
        CPU,
        lambda c, d: (model, {0}),
    )
    assert all(p.requires_grad for p in model.parameters())


def test_probe_reports_a_parameter_without_gradient() -> None:
    model = make_tiny_model().train()
    ids = make_prompts(2, 16, vocab_size=128, seed=0)
    # An optimizer that only knows some parameters: the others get gradients
    # but never change.
    some = [p for name, p in model.named_parameters() if "layers.0" in name]
    optimizer = torch.optim.AdamW(some, lr=1e-3)
    probe = Probe(model)
    losses = train_steps(model, optimizer, ids, 2, "float32", probe=probe).tolist()
    report = evaluate_gate(losses, losses, probe, rtol=1e-3)
    assert not report.passed
    assert any("never changed" in failure for failure in report.failures)
    assert all("layers.0" not in name for name in probe.unchanged[0])


def test_gate_rejects_non_finite_loss() -> None:
    model = make_tiny_model().train()
    ids = make_prompts(1, 8, vocab_size=128, seed=0)
    probe = Probe(model)
    train_steps(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3), ids, 1, "float32", probe=probe
    )
    report = evaluate_gate([math.nan], [1.0], probe, rtol=1e-3)
    assert report.kind == "mismatch" and "loss is not finite" in report.failures


def test_reference_loop_is_independent_of_the_timed_one() -> None:
    model = make_tiny_model().train()
    ids = make_prompts(2, 16, vocab_size=128, seed=0)
    losses = reference_losses(model, ids, 3, CONFIG)
    assert losses[2] < losses[0]


def test_flops_estimate_counts_linear_layers_and_attention() -> None:
    model = make_tiny_model()
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert forward_flops_per_token(model, 16) == 2 * linear + 4 * 2 * 16 * 32
    assert forward_flops_per_token(model, 32) > forward_flops_per_token(model, 16)


def test_failures_are_classified() -> None:
    def broken(config: TrainConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
        raise torch.OutOfMemoryError("CUDA out of memory")

    record = run_train_task(CONFIG, CPU, broken)
    assert record["status"] == RunStatus.ENV_ERROR and "Traceback" in record["traceback"]


def test_train_command_appends_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import miniinferbench.training as training

    monkeypatch.setattr(
        training, "run_train_task", lambda c, o: run_train_task(c, CPU, tiny_loader)
    )
    config, results = tmp_path / "c.json", tmp_path / "train.jsonl"
    CONFIG.save(config)
    assert main(["train", "--config", str(config), "--results", str(results), "--allow-cpu"]) == 0
    assert "gate:" in capsys.readouterr().out
    (record,) = read_records(results)
    assert record["task"] == "train_step"


def test_shipped_train_configs_load() -> None:
    for path in Path("configs").glob("train_*.json"):
        TrainConfig.load(path)
