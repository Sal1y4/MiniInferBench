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
