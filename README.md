# MiniInferBench

A small, reproducible benchmark harness for LLM inference on a single GPU.

It drives a Hugging Face model with a hand-written prefill + decode loop, times
every step, checks the output token by token against `generate()`, and writes
one classified record per run (config, environment, raw samples, evidence).
Every number below comes from a record in [`results-published/`](results-published/).

## Results: SmolLM2-360M on an RTX 5070

Setup: `HuggingFaceTB/SmolLM2-360M-Instruct`, bfloat16, greedy, synthetic
prompts (seeded, fixed length), EOS ignored so every run does the same work.
RTX 5070 12 GB (SM120), WSL2, torch 2.11.0+cu128, transformers 5.22.1.
2 warmup runs + 5 measured runs per point; all records produced at commit
`00da9fe` with a clean tree.

Sweep `5070-360m-v3`: batch {1,2,4,8} x input {128,512,1024} x output {32,128,256}.
All 36 points match `generate()` exactly; 20 are `ok`, 16 `unstable`
(see [Limitations](#limitations)). Full table: `mib table --tag 5070-360m-v3
--results results-published/5070-360m-v3.jsonl`.

| Batch | Input | Output | Status | TTFT (ms) | ITL P50 (ms) | ITL P99 (ms) | Decode tok/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 128 | ok | 15.50 | 14.58 | 16.94 | 69.65 | 744 MiB |
| 2 | 512 | 128 | ok | 23.67 | 13.89 | 42.00 | 140.27 | 776 MiB |
| 4 | 512 | 128 | ok | 35.46 | 14.86 | 30.66 | 272.14 | 838 MiB |
| 8 | 512 | 128 | unstable | 61.81 | 14.12 | 26.95 | 568.62 | 965 MiB |
| 8 | 1024 | 256 | ok | 135.97 | 14.39 | 17.40 | 561.67 | 1215 MiB |

![Decode throughput vs batch size](docs/figures/throughput_vs_batch.png)

**Finding 1: decode time is set by the CPU launching kernels, not by the GPU.**

- ITL P50 stays between 13.9 and 14.9 ms at every point with output 128 or
  256, whatever the batch size or input length, so decode throughput grows almost linearly with batch size
  (about 70 → 140 → 272 → 560 tokens/s).
- Profiler traces (`results-published/traces/`) show about 1,300 kernel
  launches per decode step. The GPU is busy for only 3.6 ms (batch 1) to 4.5 ms
  (batch 8) of a ~14.5 ms step.
- Prefill of 128 tokens at batch 1 (TTFT 14.7 ms) costs about the same as one
  decode step: both are dominated by the same per-layer launch sequence.

The obvious next experiment is to cut launches with a static KV cache plus
CUDA Graphs or `torch.compile(mode="reduce-overhead")`. It is not done yet.

![Prefill vs one decode step](docs/figures/prefill_vs_decode.png)

More figures: [latency vs batch](docs/figures/latency_vs_batch.png),
[prefill vs input length](docs/figures/prefill_vs_input.png),
[ITL distribution](docs/figures/itl_distribution.png),
[memory: measured vs estimated](docs/figures/memory.png).

## Case study: can the harness catch a regression?

Two slowdowns are built in behind a config flag; both keep the output
identical, so only timing and traces can reveal them. Batch 4, input 512,
output 128 ([`results-published/regression.jsonl`](results-published/regression.jsonl)).

`mib compare` judges each change against a noise band of max(2%, 3 x the
larger end-to-end CV of the two runs):

| Metric | Baseline | `host_sync_each_step` | `no_kv_cache` |
| --- | --- | --- | --- |
| TTFT mean (ms) | 34.24 | 39.21 (+14.5%, band ±10.7%) | 35.23 (within noise) |
| ITL P50 (ms) | 14.82 | 14.50 (within noise) | 40.65 (+174%) |
| ITL P99 (ms) | 24.73 | **42.18 (+70.5%)** | 56.47 (+128%) |
| E2E mean (ms) | 1948.6 | 1961.4 (within noise) | 5368.3 (+176%) |
| Blocking host calls in trace | 2 | **256** | not traced |

- **No KV cache** recomputes the whole sequence every step: every decode metric
  moves, prefill does not. Easy to catch.
- **A host sync each step** (`token.cpu()`, a common bug when checking for EOS)
  does not move the median at all. Because decode is launch-bound, the GPU has
  usually finished by the time the CPU asks, so the sync is free most of the
  time and stalls only occasionally. It shows up only in the tail (ITL P99) and
  in the trace (blocking host calls 2 → 256, same kernels, same GPU busy time).
  A dashboard that tracks only P50 or end-to-end time would miss it. The TTFT
  change is outside its band, but this regression only touches the decode loop,
  so it is most likely noise.

## The bug the harness caught in itself

The first sweep reported 21 of 36 points as `wrong_result`. The correctness
check now records, for each diverging sequence, the logit margin between the
two tokens on their shared prefix. That evidence showed:

- our loop always picked token 2 (`<|im_end|>`), which the model ranked
  about 7 logits above the reference's choice;
- the divergence was identical in float32, so it was not a precision issue.

The reference was wrong: `GenerationConfig(eos_token_id=None, min_new_tokens=n)`
does not disable EOS. It falls back to the model's EOS, and `min_new_tokens`
then masks it, forcing the runner-up token. Fixed with `eos_token_id=[]`, with
a regression test that fails on the old config.

## How it works

```
config ──> preflight ──> load model ──> correctness check ──> warmup ──> measure
                                          (vs generate())                  │
record (status, manifest, raw samples, evidence) <── stability check <─────┘
```

- **Two clocks.** CUDA events give per-step gaps on the GPU timeline (TTFT,
  ITL); a synchronized wall clock gives end-to-end time. `scripts/timing_pitfall.py`
  shows why: without synchronization a 41 ms GPU workload "takes" 0.18 ms.
- **Correctness first.** A run that does not match `generate()` is never timed.
  Divergences are classified as exact, near tie (top-two tokens within a
  configurable logit tolerance) or mismatch, and the margin is stored.
- **One status per run.** Failures are classified rather than crashing the sweep.
- **Comparable by construction.** Each record stores a fingerprint of the input
  tokens; `mib compare` refuses runs with different inputs.
- **Resumable sweeps.** Points are keyed by tag; a rerun skips finished points.
  Points whose memory estimate exceeds 90% of free memory are recorded as
  `skipped` rather than attempted.
- **Profiling is a separate pass.** `mib trace` measures its own overhead
  (traced step span was 1.4–1.8x the untraced ITL), so traces are used for
  proportions and kernel breakdowns, not absolute times.

| Status | Meaning |
| --- | --- |
| `ok` | Output matches `generate()`, end-to-end CV within threshold (0.05) |
| `wrong_result` | Output differs from `generate()`; timing is skipped |
| `unstable` | Correct, but end-to-end time varies more than the CV threshold |
| `env_error` | Out of memory, no CUDA, kernel/driver mismatch, download or IO failure |
| `program_error` | Any other exception, with its traceback stored |
| `skipped` | Memory estimate exceeds the safe budget |

## Usage

Requires Linux or WSL2, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"      # torch comes from the CUDA 12.8 wheel index
python scripts/check_env.py      # GPU, CUDA, and library versions

mib run --config configs/smollm2_360m_smoke.json     # one run -> results/runs.jsonl
mib rerun <run_id>                                    # same config, new record
mib compare <baseline_run_id> <candidate_run_id>      # changes vs noise band
mib sweep --config configs/smollm2_360m_sweep.json --tag my-sweep
mib table --tag my-sweep && mib plot --tag my-sweep
mib trace --config configs/regression_baseline.json --out traces/baseline
```

Development checks (CI runs the same on CPU, with a tiny random Llama; GPU-only
tests are marked `gpu`):

```bash
ruff check . && ruff format --check . && mypy && pytest -q
```

## Limitations

- **Noise.** 16 of 36 sweep points exceed the 0.05 CV threshold, mostly short
  runs (output 32). The first sweep, on the same machine, had 3. During the
  final sweep other long-running processes were using several GB of the WSL
  VM's memory; host-side interference is the likely cause but is not proven.
  Unstable points are kept and labelled, not dropped. GPU clocks stayed at
  ~2880 MHz and 46–57 °C, so thermal throttling is ruled out.
- **One model, one GPU, one framework.** Results describe HF Transformers eager
  mode on a small model. They say nothing about vLLM, larger models, or other
  GPUs.
- **Synthetic prompts, equal length, no padding.** Real traffic is not like this.
- **Correctness is checked against `generate()`**, which is itself code with
  configuration pitfalls (see above). Matching it proves agreement, not truth.

## Methodology reference

The measurement methodology (separate prefill and decode, warmup isolation,
per-step timestamps, a separate profiling pass, raw-quantity result logging)
is adapted from the benchmark in [DeepInfer](https://github.com/deepseek-ai/DeepInfer).
DeepInfer's engine targets SM100 and Ascend, so it does not run on this GPU;
the code here is an independent implementation, and its numbers are not
comparable to DeepInfer's.

| DeepInfer idea | Here |
| --- | --- |
| Prefill and decode timed separately | TTFT vs ITL from per-step CUDA events |
| Warmup before measurement | `warmup_runs`, excluded from records |
| N+1 timestamps give N intervals | `EventTimeline.mark()` per token |
| Steps/s vs tokens/s kept apart | `decode_steps_per_s` and `decode_tokens_per_s` |
| Profiler in its own pass | `mib trace`, with measured overhead |
| Raw quantities in JSONL | `raw` field with every sample |

See [docs/PLAN.md](docs/PLAN.md) for the roadmap.
