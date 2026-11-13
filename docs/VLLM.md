# vLLM as a comparison backend

`mib vllm` runs the same grid as `mib sweep` on [vLLM](https://github.com/vllm-project/vllm)
and writes records in the same JSONL shape, so `mib table`, `mib compare` and
the plots work on both. This page says what is kept equal, what is not, and
which numbers can be compared.

## Setup

vLLM pins its own torch build (vLLM 0.32.0 needs torch 2.13.0, built for CUDA
13.0), while the main environment uses torch 2.11 for CUDA 12.8. Installing
both in one environment would replace the main torch, so vLLM gets its own:

```bash
uv venv --python 3.12 .venv-vllm
VIRTUAL_ENV=.venv-vllm uv pip install "vllm==0.32.0"
VIRTUAL_ENV=.venv-vllm uv pip install --no-deps -e .   # keep vLLM's torch
.venv-vllm/bin/mib vllm --config configs/vllm_360m_sweep.json --tag my-vllm \
    --input 128 512 1024 --output 128                  # add --enforce-eager for eager
```

Why 0.32.0: its build lists compute capability 12.0 (RTX 50 series) in
`CUDA_SUPPORTED_ARCHS`, and its wheel is built with CUDA 13.0, which the
installed driver supports (CUDA 13.3). 0.33.0 was five days old when this was
set up. `torch.cuda.get_arch_list()` in that environment includes `sm_120`.

Two environment variables are set by the backend before vLLM is imported:

| Variable | Value | Why |
| --- | --- | --- |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Engine core runs in the caller's process, so step timestamps contain no inter-process messaging. |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | FlashInfer's top-k/top-p sampler is JIT-compiled on first use and needs a full CUDA toolkit (it failed here on a missing `curand.h`). Greedy decoding does not use it. |

## What is kept equal

| | HF runner | vLLM backend |
| --- | --- | --- |
| Model and weights | `SmolLM2-360M-Instruct`, bf16 | same checkpoint, bf16 |
| Prompts | `make_prompts(batch, input_len, vocab, seed)` | same call; `input_fingerprint` matches the HF records |
| Decoding | greedy, EOS ignored, exactly `output_len` tokens | `temperature=0`, `ignore_eos`, `max_tokens = output_len` (no `min_tokens`, see below) |
| Batching | one static batch, all sequences in lockstep | all requests added before the first step; a run is rejected unless every request gets exactly one token per step |
| Detokenization | none | `detokenize=False` |
| Correctness | tokens vs HF `generate()`, divergences re-scored with HF logits | same check, same tolerance |
| Repeats | `warmup_runs`, then `measure_runs` | same |

Prefix caching is turned off: every run repeats the same prompts, and a cache
hit would skip most of the prefill.

## What is different

**Timing boundary.** The HF runner records CUDA events on the GPU stream; a
gap between two events is GPU time between two tokens. vLLM hides its stream,
so the backend takes a host timestamp each time `LLMEngine.step()` returns
new tokens to Python. Each gap therefore includes vLLM's scheduler, input
preparation, sampler and output processing for that step. When the GPU
finishes before the host asks, the two clocks agree; when the host is the
bottleneck, the host clock is the one a caller sees.

```
HF:    [event] prefill [event] step [event] step ... [event]     (GPU timeline)
vLLM:  t0 step() step() -> first tokens  step() -> tokens ...    (host clock)
```

TTFT for vLLM starts right after the requests are added (input processing
is outside it) and ends when the last request's first token is returned.
With async scheduling the first `step()` returns no tokens; it is part of TTFT.

**Execution.** vLLM uses its own model code, a paged KV cache, FlashAttention
and, unless `--enforce-eager`, torch.compile with piecewise and full CUDA
graphs (`cudagraph_mode` is stored in each record). Its eager mode is not
the same program as our `eager` mode, and its graph mode is not our
`cuda_graph` mode.

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU (0.4 here) up
front and fills it with KV cache blocks. Peak allocated memory then measures
that setting, not the run, so `peak_memory_bytes` is not recorded for vLLM.

**Engine size.** One engine serves the whole sweep, sized for its largest
point (`max_num_seqs` = largest batch, `max_model_len` = longest input plus
longest output, and a token budget large enough for the whole batch's prefill
in one step). These are stored under `backend` in each record.

**Scheduling.** Offline, a static batch with every request present from the
start is the only case where the two sides do the same work. vLLM's
continuous batching, preemption and chunked prefill are exactly what this
setup avoids, so these numbers say nothing about vLLM as a server under load.

## Which numbers compare

| Metric | Comparable? |
| --- | --- |
| E2E wall time, output tokens/s | Yes: both are synchronized host wall clocks over the same work. |
| ITL, decode tokens/s | Mostly: different clocks (GPU events vs host), same definition (one gap per step, every sequence advances). |
| TTFT, prefill tokens/s | Loosely: vLLM's TTFT contains a scheduling step and output handling that HF's does not. |
| Peak memory | No. |
| CV / status | Yes: same thresholds. |

## A bug the correctness check caught

The first vLLM sweep set `min_tokens = max_tokens = output_len`, mirroring
the HF reference's `min_new_tokens`. Every input-1024 point came back
`wrong_result`: vLLM chose a token about 7 logits below the best one, and
the best one was EOS (token 2). vLLM's min-tokens logits processor masks
stop tokens until the minimum length is reached, even with `ignore_eos`,
which is the same trap the HF reference fell into with
`GenerationConfig(eos_token_id=None, min_new_tokens=n)`. `ignore_eos` with
`max_tokens` already gives exactly `output_len` tokens (the backend checks
the count on every run), so `min_tokens` was removed and a regression test
pins that. Those records were discarded, not published.

## Published run

`results-published/vllm.jsonl`: tags `5070-360m-vllm-graph` and
`5070-360m-vllm-eager`, batch {1,2,4,8} x input {128,512,1024}, output 128,
6 warmup + 5 measured runs, commit `6437ab9`, clean tree. Run on an idle
machine (1-minute load average at run start 0.35-2.0, recorded per record in
`manifest.host`; the higher values are the previous point's own load). An
earlier attempt overlapped with an unrelated CPU-heavy job on the same machine
and was discarded.

Table and figures:

```bash
mib table --tag 5070-360m-vllm-graph --results results-published/vllm.jsonl
mib plot-modes --prefix backends \
    --results results-published/5070-360m-modes.jsonl results-published/vllm.jsonl \
    --tag "HF eager=5070-360m-eager-v6" --tag "HF cuda_graph=5070-360m-cudagraph-v3" \
    --tag "vLLM eager=5070-360m-vllm-eager" --tag "vLLM graph=5070-360m-vllm-graph"
```

## Cross-check against `vllm bench latency`

Our vLLM timing loop is our own code (host timestamps around
`LLMEngine.step()`), so its E2E was checked against vLLM's own offline
latency benchmark. `scripts/vllm_bench_crosscheck.sh` runs
`vllm bench latency` at the same batch, input and output lengths, with the
same engine settings (`max_num_seqs` 8, `max_model_len` 1152,
`max_num_batched_tokens` 9216, `gpu_memory_utilization` 0.4, prefix caching
off, no detokenization, the same two environment variables), 6 warmup and
20 measured iterations per point.

`vllm bench latency` times one `LLM.generate()` call with `perf_counter()`.
That is the same boundary as our E2E except for three things: it includes
adding the requests (input processing), it samples with `temperature=1.0`
instead of greedy, and its prompts are random token ids below 10000, not our
`make_prompts` output. With `ignore_eos` the work per step is the same, so
the two numbers should agree if our loop adds no hidden cost.

Graph mode, output 128, against the published records (run on 2026-11-13,
same machine, idle, 1-minute load average 0.1-1.5):

| Batch | Input | E2E ours (published, 5 runs) | `vllm bench latency` (20 runs) | Diff |
| --- | --- | --- | --- | --- |
| 1 | 128 | 336.9 ms | 329.9 ms | +2.1% |
| 1 | 512 | 334.1 ms | 346.5 ms | -3.6% |
| 1 | 1024 | 342.2 ms | 347.4 ms | -1.5% |
| 2 | 128 | 345.0 ms | 342.9 ms | +0.6% |
| 2 | 512 | 337.9 ms | 350.0 ms | -3.5% |
| 2 | 1024 | 373.8 ms | 364.1 ms | +2.7% |
| 4 | 128 | 351.7 ms | 337.9 ms | +4.1% |
| 4 | 512 | 374.1 ms | 375.8 ms | -0.4% |
| 4 | 1024 | 414.3 ms | 411.4 ms | +0.7% |
| 8 | 128 | 344.1 ms | 348.4 ms | -1.2% |
| 8 | 512 | 420.5 ms | 420.1 ms | +0.1% |
| 8 | 1024 | 508.6 ms | 506.6 ms | +0.4% |

Every point is within 4.1%, with no consistent sign, and the run-to-run CV of
`vllm bench latency` at the small points is 0.03-0.06, so the differences are
within noise. The longest points, where GPU work dominates, agree within
0.7%. Our E2E measurement adds no visible cost to graph-mode vLLM.

Eager mode was checked at the four corners. Against the published eager
records two points differed by 13-16% (batch 1 input 128: 1606 vs 1389 ms;
batch 8 input 1024: 1834 vs 1623 ms). Re-running our harness on the same day
as the benchmark (`--enforce-eager --batch 1 8 --input 128 1024`) gave:

| Batch | Input | E2E ours (same day, 5 runs) | `vllm bench latency` (20 runs) | Diff |
| --- | --- | --- | --- | --- |
| 1 | 128 | 1423.8 ms | 1388.5 ms | +2.5% |
| 1 | 1024 | 1638.2 ms | 1575.1 ms | +4.0% |
| 8 | 128 | 1683.5 ms | 1631.1 ms | +3.2% |
| 8 | 1024 | 1665.4 ms | 1622.6 ms | +2.6% |

So the large gap was between days, not between tools: the published eager
batch 1 input 128 point (1606 ms) and today's run of the same point
(1424 ms) differ by 13% with the same code. Eager vLLM is host-bound and its
CV is 0.06-0.09 on both tools (3 of these 4 rerun points are `unstable`), so
eager absolute numbers should not be compared across days. Same-day, ours
is 2.5-4.0% slower on all four points. That consistent sign may be our
per-step Python bookkeeping (one timestamp and one dict per `step()`), which
matters more when each step is host-bound, but with 5 runs and CV above 0.05
it was not separated from noise.

These raw files are in `results/` (not committed): the benchmark JSONs in
`results/vllm-bench-latency/`, the eager rerun in
`results/xcheck-eager.jsonl` (tag `xcheck-eager-rerun`).
