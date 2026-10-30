# MiniInferBench

A small, reproducible benchmark harness for LLM inference on a single GPU.

> **Status: work in progress (v0.1 under development).** No benchmark results are
> published yet. Every number that appears in this repository will come from a
> recorded run whose configuration and environment are stored next to it.

## Goals

- Measure prefill and decode separately with a hand-written generation loop
  (not `model.generate()`), so every step can be timed.
- Report each metric with an explicit measurement boundary (TTFT, ITL, TPOT,
  P50/P95/P99, prefill/decode throughput, peak memory).
- Treat every run as a reproducible task: saved config, environment snapshot,
  correctness check, and a status that separates program errors, environment
  failures, and unstable measurements.
- Back performance claims with profiler traces, not guesses.

## Setup

Requires Linux or WSL2, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"     # torch comes from the CUDA 12.8 wheel index
python scripts/check_env.py     # prints GPU, CUDA, and library versions
pytest -q
```

## Usage

```bash
mib run --config configs/smollm2_360m_smoke.json   # appends a record to results/runs.jsonl
mib rerun <run_id>                                  # repeat a run with its exact config
mib compare <baseline_run_id> <candidate_run_id>    # changes judged against run-to-run noise
python scripts/timing_pitfall.py                    # why GPU timing needs synchronization
```

Every run ends with exactly one status:

| Status | Meaning |
| --- | --- |
| `ok` | Finished, output matches `generate()`, repeated measurements stable |
| `wrong_result` | Output differs from `generate()`; timing is skipped |
| `unstable` | Correct, but end-to-end time varies more than the CV threshold |
| `env_error` | Out of memory, no CUDA, kernel/driver mismatch, download or IO failure |
| `program_error` | Any other exception, with its traceback stored |

Each record stores the config, git commit and dirty flag, library and GPU
versions, the model revision, a fingerprint of the input tokens, and every raw
timing sample.

## Development checks

```bash
ruff check . && ruff format --check .
mypy
pytest -q
```

CI runs the same checks on CPU; tests that need a GPU are marked `gpu` and
skipped automatically when no CUDA device is present.

## Methodology reference

The measurement methodology (warmup isolation, steady-state windows, separate
profiling passes, raw-quantity result logging) is adapted from the benchmark in
[DeepInfer](https://github.com/deepseek-ai/DeepInfer). The code here is an
independent implementation for a single consumer GPU and small dense models;
results are not comparable to DeepInfer's published numbers.

See [docs/PLAN.md](docs/PLAN.md) for the roadmap.
