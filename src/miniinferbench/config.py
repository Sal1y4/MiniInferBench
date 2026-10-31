"""Benchmark configuration: one immutable record that fully describes a run.

Every experiment parameter lives here so that a run can be saved, reloaded,
and repeated exactly. Unknown or missing fields are rejected on load instead
of being silently ignored.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

Dtype = Literal["bfloat16", "float16", "float32"]
Regression = Literal["none", "host_sync_each_step", "no_kv_cache"]
DecodeMode = Literal["eager", "cuda_graph"]


class ConfigError(ValueError):
    """Raised when a configuration is invalid or cannot be loaded."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Parameters of one benchmark run.

    Attributes:
        model_name: Hugging Face model id or local checkpoint directory.
        batch_size: Number of sequences decoded together (static batch).
        input_len: Prompt length in tokens, identical for every sequence.
        output_len: Number of tokens generated per sequence (EOS is ignored).
        seed: Seed for the synthetic workload.
        dtype: Model weight and activation dtype.
        warmup_runs: Full prefill+decode runs executed before measuring.
        measure_runs: Measured runs; each one contributes one set of samples.
        regression: Deliberately injected slowdown for case studies; "none" normally.
        decode_mode: "eager" (dynamic KV cache, one launch per kernel) or
            "cuda_graph" (static KV cache, each decode step replayed as a CUDA graph).
    """

    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    batch_size: int = 1
    input_len: int = 128
    output_len: int = 32
    seed: int = 0
    dtype: Dtype = "bfloat16"
    warmup_runs: int = 2
    measure_runs: int = 5
    regression: Regression = "none"
    decode_mode: DecodeMode = "eager"

    def __post_init__(self) -> None:
        positive = ("batch_size", "input_len", "output_len", "measure_runs")
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.warmup_runs, int) or self.warmup_runs < 0:
            raise ConfigError(f"warmup_runs must be >= 0, got {self.warmup_runs!r}")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ConfigError(f"unsupported dtype {self.dtype!r}")
        if self.regression not in ("none", "host_sync_each_step", "no_kv_cache"):
            raise ConfigError(f"unknown regression {self.regression!r}")
        if self.decode_mode not in ("eager", "cuda_graph"):
            raise ConfigError(f"unknown decode_mode {self.decode_mode!r}")
        if self.decode_mode == "cuda_graph" and self.regression == "no_kv_cache":
            raise ConfigError("no_kv_cache cannot be combined with cuda_graph decoding")
        if not self.model_name:
            raise ConfigError("model_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown config fields: {', '.join(unknown)}")
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config {path} must contain a JSON object")
        return cls.from_dict(data)
