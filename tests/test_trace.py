import json
from pathlib import Path

import pytest

from miniinferbench.trace import (
    _union_length,
    analyze,
    count_graph_launches,
    count_launches,
    step_stats,
    top_kernels,
)


def _span(ts: float, dur: float, name: str = "decode_step") -> dict[str, object]:
    return {"cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur}


def _kernel(ts: float, dur: float, name: str = "gemm") -> dict[str, object]:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur}


def test_union_merges_overlaps() -> None:
    assert _union_length([(0, 10), (5, 15), (20, 25)]) == 20
    assert _union_length([]) == 0


def test_step_busy_time_and_idle_gaps() -> None:
    # One 100 us step with 30 us of kernels and idle gaps between them.
    events = [_span(0, 100), _kernel(10, 10), _kernel(50, 10), _kernel(80, 10)]
    (step,) = step_stats(events)
    assert step.span_us == 100
    assert step.busy_us == 30
    assert step.kernels == 3
    assert step.busy_fraction == pytest.approx(0.3)


def test_kernels_are_clipped_to_the_step() -> None:
    events = [_span(0, 100), _span(100, 100), _kernel(90, 20)]
    first, second = step_stats(events)
    assert first.busy_us == 10 and second.busy_us == 10


def test_top_kernels_ranks_by_total_time() -> None:
    events = [_kernel(0, 5, "a"), _kernel(10, 5, "a"), _kernel(20, 20, "b")]
    assert top_kernels(events) == [("b", 20, 1), ("a", 10, 2)]


def test_analyze_reads_chrome_trace(tmp_path: Path) -> None:
    events = [_span(0, 100), _kernel(0, 50), _span(100, 100), _kernel(100, 25)]
    events.append({"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0, "dur": 1})
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    report = analyze(path)
    assert report["decode_steps"] == 2
    assert report["step_busy_fraction_median"] == pytest.approx(0.25)  # first step skipped
    assert report["kernel_launches_total"] == 1
    assert report["host_wait_calls_total"] == 0


def test_analyze_rejects_trace_without_steps(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"traceEvents": [_kernel(0, 1)]}))
    with pytest.raises(ValueError, match="decode_step"):
        analyze(path)


def test_capture_writes_a_chrome_trace(tmp_path: Path) -> None:
    from miniinferbench.trace import capture_trace
    from miniinferbench.workload import make_prompts

    from .tiny_model import make_tiny_model

    path = tmp_path / "t.trace.json"
    capture_trace(make_tiny_model(), make_prompts(1, 8, vocab_size=128, seed=0), 4, path)
    events = json.loads(path.read_text())["traceEvents"]
    names = {e.get("name") for e in events}
    assert {"prefill", "decode_step"} <= names


def test_launches_count_runtime_and_driver_api() -> None:
    # Triton kernels from torch.compile are launched through the driver API.
    events = [
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel"},
        {"cat": "cuda_driver", "name": "cuLaunchKernel"},
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch"},
        {"cat": "cpu_op", "name": "aten::mm"},
    ]
    assert count_launches(events) == 2
    assert count_graph_launches(events) == 1
