# Plan

## v0.1 (one week)

| Day | Scope |
| --- | --- |
| 1 | Environment check, project skeleton, config, ruff + mypy + pytest CI |
| 2 | Inference task: hand-written prefill/decode loop, CUDA event and wall-clock timing, token-level correctness check against `generate()` |
| 3 | Thin harness: task spec, run status classes, run manifest, JSONL results, fault-injection tests |
| 4 | Parameter sweep (batch 1/2/4/8 x input 128/512/1024 x output 32/128/256) with repeats, variance check, and memory-based skipping |
| 5 | torch.profiler traces for representative points and a small trace analyzer |
| 6 | Injected-regression case study, plots, README results |
| 7 | Buffer |

## Run status classes

| Status | Meaning |
| --- | --- |
| `ok` | Finished, output correct, measurements stable |
| `wrong_result` | Finished, but output does not match the reference |
| `program_error` | Exception raised by benchmark code |
| `env_error` | Environment problem: CUDA unavailable, out of memory, download failure |
| `unstable` | Repeated measurements vary more than the configured threshold |
| `skipped` | Not run because the memory estimate exceeds the safe budget |

## Next (in this order)

1. **Measurement trust.** A/A runs for the false-positive rate of each
   comparison rule, calibrated injected slowdowns for the smallest detectable
   regression (`mib repeat`, `mib noise`). Code is in; results from the RTX 5070
   are pending.
2. **Verified training-step task.** Forward + backward + optimizer with
   per-phase timing; timed only if the loss matches a reference loop and
   gradients actually flow.
3. **Diagnosis task set.** Output-preserving slowdowns with known root causes,
   taken from published production cases, plus unchanged and legitimately
   changed controls; scores a diagnoser on detection, root cause and false alarms.

## Later

- Training-step task (forward + backward + optimizer) to show the harness is not inference-only
- Static KV cache + CUDA Graphs / `torch.compile` before-and-after study
- vLLM as a reference backend (after checking SM120 support)
- A small set of performance-diagnosis tasks with known root causes

## Hardware notes

- Primary results: RTX 5070 desktop, 12 GB, SM120.
- Development: RTX 3060 Laptop, 6 GB, SM86. Laptop power and clock behavior make
  these numbers noisier; they are used for debugging, not for published results.
