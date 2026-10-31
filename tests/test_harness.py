from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel

from miniinferbench.config import BenchmarkConfig
from miniinferbench.harness import HarnessOptions, compare_records, run_task
from miniinferbench.results import append_record, find_record, read_records
from miniinferbench.status import RunStatus

from .tiny_model import make_tiny_model

CONFIG = BenchmarkConfig(
    model_name="tiny", batch_size=2, input_len=8, output_len=4, warmup_runs=0, measure_runs=3
)
CPU = HarnessOptions(require_cuda=False, cv_threshold=10.0)


def tiny_loader(config: BenchmarkConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
    return make_tiny_model().to(device), {0}


def failing_loader(exc: BaseException) -> Any:
    def loader(config: BenchmarkConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
        raise exc

    return loader


def test_ok_run_has_evidence() -> None:
    record = run_task(CONFIG, CPU, tiny_loader)
    assert record["status"] == RunStatus.OK
    assert record["manifest"]["env"]["torch"] == torch.__version__
    assert len(record["raw"]["e2e_wall_ms"]) == 3
    assert len(record["raw"]["itl_ms"][0]) == 3  # output_len - 1 gaps per run
    assert record["input_fingerprint"]


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (torch.OutOfMemoryError("CUDA out of memory"), RunStatus.ENV_ERROR),
        (OSError("cannot download model"), RunStatus.ENV_ERROR),
        (RuntimeError("CUDA error: no kernel image is available"), RunStatus.ENV_ERROR),
        (ValueError("bug in our code"), RunStatus.PROGRAM_ERROR),
        (KeyError("missing"), RunStatus.PROGRAM_ERROR),
    ],
)
def test_failures_are_classified(exc: BaseException, expected: RunStatus) -> None:
    record = run_task(CONFIG, CPU, failing_loader(exc))
    assert record["status"] == expected
    assert type(exc).__name__ in record["reason"]
    assert "Traceback" in record["traceback"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA path")
def test_missing_cuda_is_env_error() -> None:
    record = run_task(CONFIG, HarnessOptions(require_cuda=True), tiny_loader)
    assert record["status"] == RunStatus.ENV_ERROR


def test_wrong_output_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    import miniinferbench.runner as runner

    def broken(model: Any, ids: torch.Tensor, n: int, t: Any = None, **kw: Any) -> torch.Tensor:
        return torch.zeros((ids.shape[0], n), dtype=torch.int64, device=ids.device)

    monkeypatch.setattr(runner, "greedy_decode", broken)
    record = run_task(CONFIG, CPU, tiny_loader)
    assert record["status"] == RunStatus.WRONG_RESULT
    assert record["correctness"]["kind"] == "mismatch"
    assert "logit margin" in record["reason"]
    assert "summary" not in record  # an incorrect loop is never timed


def test_noisy_run_is_unstable() -> None:
    record = run_task(CONFIG, HarnessOptions(require_cuda=False, cv_threshold=-1.0), tiny_loader)
    assert record["status"] == RunStatus.UNSTABLE


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    first = run_task(CONFIG, CPU, tiny_loader)
    second = run_task(CONFIG, CPU, failing_loader(ValueError("x")))
    append_record(path, first)
    append_record(path, second)
    assert [r["status"] for r in read_records(path)] == ["ok", "program_error"]
    assert find_record(path, second["run_id"])["reason"] == "ValueError: x"
    with pytest.raises(KeyError):
        find_record(path, "missing")


def _record(e2e: float, cv: float, fingerprint: str = "abc") -> dict[str, Any]:
    return {
        "run_id": f"r{e2e}",
        "status": "ok",
        "input_fingerprint": fingerprint,
        "summary": {"e2e_wall_ms_mean": e2e, "e2e_wall_ms_cv": cv, "ttft_ms_p50": 10.0},
    }


def test_compare_flags_only_changes_above_noise() -> None:
    small = {d.name: d for d in compare_records(_record(100.0, 0.01), _record(101.0, 0.01))}
    assert not small["e2e_wall_ms_mean"].significant  # +1% < 3% band
    big = {d.name: d for d in compare_records(_record(100.0, 0.01), _record(120.0, 0.01))}
    assert big["e2e_wall_ms_mean"].significant
    assert not big["ttft_ms_p50"].significant


def test_compare_refuses_different_inputs() -> None:
    with pytest.raises(ValueError, match="different inputs"):
        compare_records(_record(100.0, 0.01, "a"), _record(100.0, 0.01, "b"))
