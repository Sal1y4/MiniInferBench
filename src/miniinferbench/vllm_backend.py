"""vLLM as a comparison backend: same model, same prompt token ids, greedy.

vLLM pins its own torch build, so this module runs from a separate virtual
environment (`.venv-vllm`, see docs/VLLM.md). The vLLM import is deferred so
that the pure functions here can be tested without vLLM installed.

What is kept equal to the Hugging Face runner:

- the prompts: the same `make_prompts` call, so `input_fingerprint` matches
  and `mib compare` accepts an HF record against a vLLM record;
- the work: greedy, EOS ignored, exactly `output_len` tokens per sequence,
  no detokenization, all sequences submitted together (a static batch);
- the correctness check: tokens are compared with HF `generate()` and every
  divergence is re-scored with HF logits before anything is timed.

What is not equal, and cannot be made equal from outside the engine:

- timestamps are host wall-clock times at which `LLMEngine.step()` hands new
  tokens back to Python, not CUDA events on the GPU stream. They include
  vLLM's scheduler, sampler and output processing on every step;
- vLLM runs its own kernels (FlashAttention, paged KV cache, torch.compile
  with piecewise and full CUDA graphs unless `enforce_eager`);
- vLLM preallocates its KV cache from `gpu_memory_utilization`, so peak
  allocated memory says nothing about what a run needs; it is not reported.

Measurement boundaries for one run (requests already added to the engine):

    first step() ── step() ... ── all first tokens ── step() ... ── last token
      |<---------------- TTFT ---------------->|<- ITL ->|       |<- ITL ->|
      |<------------------------------ E2E ------------------------------->|
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from miniinferbench.bench import BenchmarkResult, load_model
from miniinferbench.config import BenchmarkConfig
from miniinferbench.harness import (
    HarnessOptions,
    classify_result,
    correctness_evidence,
    raw_samples,
)
from miniinferbench.manifest import build_manifest, new_run_id
from miniinferbench.results import append_record
from miniinferbench.runner import RunSamples, judge_tokens
from miniinferbench.status import EnvironmentProblem, classify_exception
from miniinferbench.sweep import SweepGrid, completed_points, point_key
from miniinferbench.workload import fingerprint, make_prompts


@dataclass(frozen=True, slots=True)
class VllmOptions:
    """Engine settings that change what vLLM executes.

    Attributes:
        enforce_eager: True runs every kernel eagerly (no torch.compile, no
            CUDA graphs); False is vLLM's default compiled + CUDA graph path.
        gpu_memory_utilization: Fraction of total GPU memory vLLM may take,
            including its preallocated KV cache. Kept low so the HF reference
            model fits next to it.
    """

    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1), got {self.gpu_memory_utilization}"
            )


class LockstepError(RuntimeError):
    """The engine did not advance every sequence by one token per step.

    The HF runner decodes a static batch in lockstep, so ITL is one gap per
    step. If vLLM splits the prefill across steps or skips a sequence in a
    step, per-step gaps no longer mean the same thing and are not reported.
    """


# One entry per `step()` call: host time (perf_counter seconds) when it
# returned, and the new token ids it delivered for each request.
Step = tuple[float, dict[str, list[int]]]


def samples_from_steps(
    start: float, steps: Sequence[Step], request_ids: Sequence[str], output_len: int
) -> tuple[RunSamples, list[list[int]]]:
    """Turn a step log into raw samples plus the tokens of each request.

    Steps that deliver nothing (vLLM's async scheduling returns the first
    prefill output one call late) are part of TTFT, not separate gaps.
    """
    tokens: dict[str, list[int]] = {rid: [] for rid in request_ids}
    times: list[float] = []
    for t, delivered in steps:
        if not delivered:
            continue
        if set(delivered) != set(request_ids) or any(len(v) != 1 for v in delivered.values()):
            counts = {rid: len(v) for rid, v in delivered.items()}
            raise LockstepError(
                f"step {len(times)} delivered {counts}; expected one token for each of "
                f"{len(request_ids)} requests"
            )
        for rid, new in delivered.items():
            tokens[rid].extend(new)
        times.append(t)
    if len(times) != output_len:
        raise LockstepError(f"got {len(times)} token steps, expected {output_len}")
    gaps = [(b - a) * 1000.0 for a, b in itertools.pairwise(times)]
    samples = RunSamples(
        ttft_ms=(times[0] - start) * 1000.0,
        itl_ms=gaps,
        e2e_wall_ms=(times[-1] - start) * 1000.0,
        peak_memory_bytes=None,
    )
    return samples, [tokens[rid] for rid in request_ids]


def sampling_kwargs(output_len: int) -> dict[str, Any]:
    """Greedy, EOS ignored, exactly `output_len` tokens, no detokenization."""
    # No min_tokens: vLLM's min-tokens processor masks EOS to -inf until the
    # minimum is reached, even with ignore_eos, so wherever EOS is the true
    # argmax it forces the runner-up. Same trap as min_new_tokens in HF
    # generate(); seen on SmolLM2 with a margin of ~7 logits. ignore_eos with
    # max_tokens already yields exactly output_len tokens (checked per run).
    return {
        "temperature": 0.0,
        "max_tokens": output_len,
        "ignore_eos": True,
        "detokenize": False,
    }


def _import_vllm() -> Any:
    # In-process engine: step() then runs scheduling, the forward pass and
    # output processing in this process, so host timestamps contain no
    # inter-process messaging. Must be set before vLLM is imported.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # FlashInfer's top-k/top-p sampler is JIT-compiled on first use and needs
    # a full CUDA toolkit (curand headers). Greedy decoding never uses it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        import vllm
    except ImportError as exc:
        raise EnvironmentProblem(
            "vLLM is not installed in this environment; run from .venv-vllm (docs/VLLM.md)"
        ) from exc
    return vllm


class VllmEngine:
    """One vLLM engine sized for the largest point of a sweep, reused across points."""

    def __init__(
        self, model_name: str, dtype: str, max_batch: int, max_len: int, options: VllmOptions
    ) -> None:
        vllm = _import_vllm()
        self.version: str = vllm.__version__
        self.options = options
        self._next_id = 0
        self._llm = vllm.LLM(
            model=model_name,
            dtype=dtype,
            enforce_eager=options.enforce_eager,
            gpu_memory_utilization=options.gpu_memory_utilization,
            max_model_len=max_len,
            max_num_seqs=max_batch,
            # The whole batch's prefill must fit in one step, or sequences
            # would get their first token in different steps (see LockstepError).
            max_num_batched_tokens=max_batch * max_len,
            # Every run repeats the same prompts; a prefix cache hit would
            # skip prefill and make TTFT meaningless.
            enable_prefix_caching=False,
            seed=0,
        )
        self._engine = self._llm.llm_engine

    def run(self, prompts: torch.Tensor, output_len: int) -> tuple[RunSamples, torch.Tensor]:
        """Generate `output_len` tokens for every row of `prompts`, timing each step."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        params = SamplingParams(**sampling_kwargs(output_len), output_kind=RequestOutputKind.DELTA)
        request_ids = []
        for row in prompts.tolist():
            request_id = f"req-{self._next_id}"
            self._next_id += 1
            self._engine.add_request(request_id, {"prompt_token_ids": row}, params)
            request_ids.append(request_id)
        steps: list[Step] = []
        start = time.perf_counter()
        while self._engine.has_unfinished_requests():
            outputs = self._engine.step()
            now = time.perf_counter()
            steps.append((now, {o.request_id: list(o.outputs[0].token_ids) for o in outputs}))
        samples, tokens = samples_from_steps(start, steps, request_ids, output_len)
        return samples, torch.tensor(tokens, dtype=torch.int64)

    def describe(self) -> dict[str, Any]:
        """Settings recorded with every result, so the engine can be rebuilt."""
        cfg = self._engine.vllm_config
        return {
            "name": "vllm",
            "version": self.version,
            **dataclasses.asdict(self.options),
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "max_model_len": cfg.model_config.max_model_len,
            "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
            "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
            "env": {
                k: os.environ.get(k)
                for k in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER")
            },
        }


def check_supported(config: BenchmarkConfig) -> None:
    """The decode-mode and regression knobs belong to the HF loop only."""
    if config.decode_mode != "eager" or config.regression != "none":
        raise ValueError(
            "vLLM runs need decode_mode 'eager' and regression 'none' "
            "(use VllmOptions.enforce_eager to choose vLLM's execution mode)"
        )


def run_vllm_task(
    config: BenchmarkConfig,
    engine: VllmEngine,
    reference: PreTrainedModel,
    special_ids: set[int],
    options: HarnessOptions | None = None,
) -> dict[str, Any]:
    """One configuration on vLLM, recorded in the same shape as `harness.run_task`."""
    options = options or HarnessOptions()
    run_id = new_run_id()
    record: dict[str, Any] = {
        "run_id": run_id,
        "config": config.to_dict(),
        "manifest": build_manifest(run_id),
        "options": dataclasses.asdict(options),
        "backend": engine.describe(),
        "model_revision": getattr(reference.config, "_commit_hash", None),
    }
    try:
        check_supported(config)
        vocab_size = reference.get_input_embeddings().weight.shape[0]
        prompts = make_prompts(
            config.batch_size, config.input_len, vocab_size, config.seed, special_ids
        )
        _, tokens = engine.run(prompts, config.output_len)
        device = next(reference.parameters()).device
        result = BenchmarkResult(
            config=config,
            input_fingerprint=fingerprint(prompts),
            correctness=judge_tokens(reference, prompts.to(device), tokens, options.tie_tolerance),
        )
        if result.correctness.matches:
            for _ in range(config.warmup_runs):
                engine.run(prompts, config.output_len)
            for _ in range(config.measure_runs):
                result.runs.append(engine.run(prompts, config.output_len)[0])
    except Exception as exc:
        record |= {
            "status": classify_exception(exc).value,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return record
    status, reason = classify_result(result, options.cv_threshold)
    record |= {
        "status": status.value,
        "reason": reason,
        "input_fingerprint": result.input_fingerprint,
        "correctness": correctness_evidence(result),
    }
    if result.runs:
        record |= {"summary": result.summary(), "raw": raw_samples(result)}
    return record


def run_vllm_sweep(
    base: BenchmarkConfig,
    grid: SweepGrid,
    results: Path,
    tag: str,
    vllm_options: VllmOptions,
    options: HarnessOptions | None = None,
) -> list[dict[str, Any]]:
    """Like `sweep.run_sweep`, on one vLLM engine sized for the largest point.

    There is no per-point memory estimate: vLLM reserves its KV cache up
    front and refuses to start if the largest point cannot fit.
    """
    check_supported(base)
    if not torch.cuda.is_available():
        raise EnvironmentProblem("CUDA is not available; vLLM needs a GPU")
    max_len = max(grid.input_lens) + max(grid.output_lens)
    # The engine goes first: it sizes its KV cache from the free memory it
    # sees at startup, and the small HF reference model fits in what is left.
    engine = VllmEngine(base.model_name, base.dtype, max(grid.batch_sizes), max_len, vllm_options)
    reference, special_ids = load_model(base, torch.device("cuda"))
    done = completed_points(results, tag)
    records = []
    for config in grid.configs(base):
        point = f"bs={config.batch_size} in={config.input_len} out={config.output_len}"
        if point_key(config.to_dict()) in done:
            print(f"[already done] {point}")
            continue
        record = run_vllm_task(config, engine, reference, special_ids, options)
        record["sweep_tag"] = tag
        append_record(results, record)
        records.append(record)
        itl = record.get("summary", {}).get("itl_ms_p50", float("nan"))
        print(f"[{record['status']}] {point} itl_p50={itl:.2f}ms {record.get('reason', '')}")
    return records
