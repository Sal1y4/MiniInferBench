"""Does using the profiler once change later, untraced timings in the same process?

Times the same generation N times, captures one profiler trace, then times it
N times again. If the second block is slower or noisier than the first, timed
runs and traced runs must not share a process in that order.

Usage: python scripts/profiler_aftereffect.py [--config configs/regression_baseline.json]
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from pathlib import Path

import torch

from miniinferbench.bench import load_model
from miniinferbench.config import BenchmarkConfig
from miniinferbench.runner import measure_run
from miniinferbench.trace import capture_trace
from miniinferbench.workload import make_prompts


def block(label: str, values: list[float]) -> None:
    mean = statistics.fmean(values)
    cv = statistics.stdev(values) / mean
    print(
        f"{label}: median {statistics.median(values):8.1f} ms  CV {cv:.3f}  "
        f"min {min(values):.1f}  max {max(values):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/regression_baseline.json"))
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this demo needs a CUDA device")
    config = BenchmarkConfig.load(args.config)
    device = torch.device("cuda")
    model, special_ids = load_model(config, device)
    vocab = model.get_input_embeddings().weight.shape[0]
    prompts = make_prompts(config.batch_size, config.input_len, vocab, config.seed, special_ids)
    input_ids = prompts.to(device)

    def e2e(runs: int) -> list[float]:
        return [measure_run(model, input_ids, config.output_len).e2e_wall_ms for _ in range(runs)]

    e2e(2)  # warmup
    block("before the profiler ", e2e(args.runs))
    block("before, second block", e2e(args.runs))
    with tempfile.TemporaryDirectory() as tmp:
        capture_trace(model, input_ids, config.output_len, Path(tmp) / "t.trace.json")
    block("after the profiler  ", e2e(args.runs))
    block("after, second block ", e2e(args.runs))


if __name__ == "__main__":
    main()
