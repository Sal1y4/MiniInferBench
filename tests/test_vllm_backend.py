import pytest

from miniinferbench.config import BenchmarkConfig
from miniinferbench.vllm_backend import (
    LockstepError,
    Step,
    VllmOptions,
    check_supported,
    samples_from_steps,
)

IDS = ("a", "b")


def lockstep(times: list[float], first_token: int = 10) -> list[Step]:
    return [(t, {rid: [first_token + i] for rid in IDS}) for i, t in enumerate(times)]


def test_ttft_itl_and_e2e_from_step_times() -> None:
    steps = lockstep([1.010, 1.014, 1.019])
    samples, tokens = samples_from_steps(1.000, steps, IDS, output_len=3)
    assert samples.ttft_ms == pytest.approx(10.0)
    assert samples.itl_ms == pytest.approx([4.0, 5.0])
    assert samples.e2e_wall_ms == pytest.approx(19.0)
    assert samples.peak_memory_bytes is None  # vLLM preallocates; not comparable
    assert tokens == [[10, 11, 12], [10, 11, 12]]


def test_empty_steps_count_towards_ttft() -> None:
    # Async scheduling: the first step() returns before the prefill output.
    steps: list[Step] = [(1.005, {}), *lockstep([1.010, 1.014])]
    samples, _ = samples_from_steps(1.000, steps, IDS, output_len=2)
    assert samples.ttft_ms == pytest.approx(10.0)
    assert samples.itl_ms == pytest.approx([4.0])


def test_split_prefill_is_rejected() -> None:
    # One request gets its first token a step later than the other.
    steps: list[Step] = [(1.01, {"a": [1]}), (1.02, {"a": [2], "b": [1]}), (1.03, {"b": [2]})]
    with pytest.raises(LockstepError, match="expected one token"):
        samples_from_steps(1.0, steps, IDS, output_len=2)


def test_several_tokens_in_one_step_are_rejected() -> None:
    steps: list[Step] = [(1.01, {"a": [1, 2], "b": [1, 2]})]
    with pytest.raises(LockstepError):
        samples_from_steps(1.0, steps, IDS, output_len=2)


def test_missing_tokens_are_rejected() -> None:
    with pytest.raises(LockstepError, match="expected 3"):
        samples_from_steps(1.0, lockstep([1.01, 1.02]), IDS, output_len=3)


def test_hf_only_knobs_are_rejected() -> None:
    check_supported(BenchmarkConfig())
    with pytest.raises(ValueError, match="decode_mode"):
        check_supported(BenchmarkConfig(decode_mode="cuda_graph"))
    with pytest.raises(ValueError, match="regression"):
        check_supported(BenchmarkConfig(regression="host_sync_each_step"))


def test_memory_fraction_is_validated() -> None:
    with pytest.raises(ValueError):
        VllmOptions(gpu_memory_utilization=1.5)
