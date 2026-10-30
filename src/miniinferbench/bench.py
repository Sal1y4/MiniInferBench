"""One benchmark: load the model, check correctness, warm up, measure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from miniinferbench import metrics
from miniinferbench.config import BenchmarkConfig
from miniinferbench.runner import CorrectnessReport, RunSamples, check_correctness, measure_run
from miniinferbench.workload import fingerprint, make_prompts

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config: BenchmarkConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
    """Load weights in eval mode and return the model plus its special token ids."""
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=DTYPES[config.dtype])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, set(tokenizer.all_special_ids)


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    input_fingerprint: str
    correctness: CorrectnessReport
    runs: list[RunSamples] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Aggregate metrics over all measured runs."""
        cfg = self.config
        ttft = [r.ttft_ms for r in self.runs]
        itl = [gap for r in self.runs for gap in r.itl_ms]
        e2e = [r.e2e_wall_ms for r in self.runs]
        out: dict[str, Any] = {
            "correct": self.correctness.matches,
            "num_runs": len(self.runs),
            "ttft_ms_mean": metrics.mean(ttft),
            "e2e_wall_ms_mean": metrics.mean(e2e),
            "e2e_wall_ms_cv": metrics.coefficient_of_variation(e2e),
            "prefill_tokens_per_s": metrics.prefill_tokens_per_s(
                cfg.batch_size, cfg.input_len, metrics.mean(ttft)
            ),
            "output_tokens_per_s": metrics.output_tokens_per_s(
                cfg.batch_size, cfg.output_len, metrics.mean(e2e)
            ),
            "peak_memory_bytes": max((r.peak_memory_bytes or 0) for r in self.runs) or None,
        }
        if itl:
            out |= {
                "itl_samples": len(itl),
                "itl_ms_p50": metrics.percentile(itl, 50),
                "itl_ms_p95": metrics.percentile(itl, 95),
                "itl_ms_p99": metrics.percentile(itl, 99),
                "decode_steps_per_s": metrics.decode_steps_per_s(itl),
                "decode_tokens_per_s": metrics.decode_tokens_per_s(itl, cfg.batch_size),
                "tpot_ms_mean": metrics.mean(
                    [metrics.tpot_ms(r.e2e_event_ms, r.ttft_ms, cfg.output_len) for r in self.runs]
                ),
            }
        return out


def run_benchmark(
    config: BenchmarkConfig,
    model: PreTrainedModel,
    special_ids: set[int],
    device: torch.device,
) -> BenchmarkResult:
    vocab_size = model.get_input_embeddings().weight.shape[0]
    prompts = make_prompts(
        config.batch_size, config.input_len, vocab_size, config.seed, excluded_ids=special_ids
    )
    input_ids = prompts.to(device)
    result = BenchmarkResult(
        config=config,
        input_fingerprint=fingerprint(prompts),
        correctness=check_correctness(model, input_ids, config.output_len, config.regression),
    )
    if not result.correctness.matches:
        return result  # timing an incorrect implementation proves nothing
    for _ in range(config.warmup_runs):
        measure_run(model, input_ids, config.output_len, config.regression)
    for _ in range(config.measure_runs):
        result.runs.append(measure_run(model, input_ids, config.output_len, config.regression))
    return result
