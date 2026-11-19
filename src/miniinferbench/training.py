"""Training-step task: time forward, backward and optimizer, but only after
checking that the loop really trains.

A throughput number for a loop that does not update the model is worthless,
and nothing in a tokens/s figure reveals it. So before any timing the loop
must pass a gate:

1. Its loss trajectory matches a deliberately plain reference loop (Hugging
   Face's own loss, default AdamW), run from the same weights on the same batch.
2. Structural checks hold on every checked step: the loss is finite, every
   parameter is trainable, received a non-zero gradient, and changed.

Measurement boundaries for one step (the batch is already on the device):

    start ── forward + loss ── backward ── optimizer.step + zero_grad ── next start
      |<----- forward ----->|<- backward ->|<--------- optimizer ------->|<- other ->|

Phase times are gaps between CUDA events, i.e. time on the device timeline.
Losses stay on the device until the run ends: reading one back each step is a
host sync, the bug the inference case study injects on purpose.
"""

from __future__ import annotations

import dataclasses
import math
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from miniinferbench import metrics
from miniinferbench.config import ConfigError, JsonConfig
from miniinferbench.manifest import build_manifest, new_run_id
from miniinferbench.status import EnvironmentProblem, RunStatus, classify_exception
from miniinferbench.timing import EventTimeline, WallClock
from miniinferbench.workload import fingerprint, make_prompts

Precision = Literal["float32", "bf16_autocast"]
# Loops that run at full speed (or faster) without training the model. Used to
# show that the gate catches them; "none" in normal use.
#   frozen_layers: only the input embedding is trainable, so backward is cheap.
#   zero_lr: the optimizer runs but never changes a weight.
TrainFault = Literal["none", "frozen_layers", "zero_lr"]
PHASES = ("forward", "backward", "optimizer", "other")
DEFAULT_LOSS_RTOL = 1e-3
DEFAULT_CV_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class TrainConfig(JsonConfig):
    """Parameters of one training-step benchmark.

    Attributes:
        model_name: Hugging Face model id or local checkpoint directory.
        batch_size: Sequences per step.
        seq_len: Tokens per sequence; every step sees batch_size x seq_len tokens.
        seed: Seed for the synthetic batch.
        precision: "float32", or "bf16_autocast" (float32 weights, bfloat16
            autocast in the forward pass).
        fused_optimizer: Use AdamW's fused implementation (CUDA only).
        compile: Wrap the model in `torch.compile` (CUDA only).
        learning_rate: AdamW learning rate.
        check_steps: Steps compared against the reference loop before timing.
        warmup_steps: Untimed steps before the measured runs.
        measure_runs: Measured windows; each one contributes one wall-clock sample.
        steps_per_run: Training steps in each measured window.
        fault: A deliberately broken loop for demonstrating the gate; "none" normally.
        peak_tflops: Vendor peak TFLOPS for this GPU and precision. MFU is
            reported only when this is given.
    """

    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    batch_size: int = 4
    seq_len: int = 256
    seed: int = 0
    precision: Precision = "bf16_autocast"
    fused_optimizer: bool = True
    compile: bool = False
    learning_rate: float = 1e-5
    check_steps: int = 5
    warmup_steps: int = 5
    measure_runs: int = 5
    steps_per_run: int = 20
    fault: TrainFault = "none"
    peak_tflops: float | None = None

    def __post_init__(self) -> None:
        for name in ("batch_size", "check_steps", "measure_runs", "steps_per_run"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.seq_len, int) or isinstance(self.seq_len, bool) or self.seq_len < 2:
            raise ConfigError(f"seq_len must be an integer >= 2, got {self.seq_len!r}")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ConfigError(f"warmup_steps must be >= 0, got {self.warmup_steps!r}")
        if self.precision not in ("float32", "bf16_autocast"):
            raise ConfigError(f"unsupported precision {self.precision!r}")
        if self.fault not in ("none", "frozen_layers", "zero_lr"):
            raise ConfigError(f"unknown fault {self.fault!r}")
        if not self.learning_rate > 0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if self.peak_tflops is not None and not self.peak_tflops > 0:
            raise ConfigError(f"peak_tflops must be > 0, got {self.peak_tflops!r}")
        if not self.model_name:
            raise ConfigError("model_name must not be empty")


def load_train_model(config: TrainConfig, device: torch.device) -> tuple[PreTrainedModel, set[int]]:
    """Load float32 weights and return the model plus its special token ids."""
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float32)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, set(tokenizer.all_special_ids)


@contextmanager
def _autocast(precision: Precision, device: torch.device) -> Iterator[None]:
    if precision == "bf16_autocast":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        yield


def _make_optimizer(
    model: torch.nn.Module, learning_rate: float, fused: bool
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if fused:
        return torch.optim.AdamW(params, lr=learning_rate, fused=True)
    return torch.optim.AdamW(params, lr=learning_rate)


class Probe:
    """Structural evidence collected on the checked steps, one entry per step."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._named = list(model.named_parameters())
        self.frozen = [name for name, p in self._named if not p.requires_grad]
        self.grad_norms: list[float] = []
        self.zero_grad: list[list[str]] = []
        self.unchanged: list[list[str]] = []
        self._before: torch.Tensor | None = None

    def _checksums(self) -> torch.Tensor:
        return torch.stack([p.detach().double().sum() for _, p in self._named])

    def after_backward(self) -> None:
        squares = []
        zero = []
        for name, p in self._named:
            if p.grad is None:
                zero.append(name)
                continue
            square = p.grad.detach().double().pow(2).sum()
            squares.append(square)
            if float(square) == 0.0:
                zero.append(name)
        self.grad_norms.append(math.sqrt(float(torch.stack(squares).sum())) if squares else 0.0)
        self.zero_grad.append(zero)
        self._before = self._checksums()

    def after_optimizer(self) -> None:
        assert self._before is not None
        same = (self._checksums() == self._before).tolist()
        self.unchanged.append([name for (name, _), s in zip(self._named, same, strict=True) if s])


def train_steps(
    forward: Callable[..., Any],
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    steps: int,
    precision: Precision,
    timeline: EventTimeline | None = None,
    probe: Probe | None = None,
) -> torch.Tensor:
    """Run `steps` optimizer steps on one batch; returns the per-step losses
    as a device tensor. The timeline is marked four times per step (see module
    docstring). The same function serves the gate, warmup and measured runs."""
    device = input_ids.device
    targets = input_ids[:, 1:].reshape(-1)
    losses = []
    for _ in range(steps):
        if timeline is not None:
            timeline.mark()
        with _autocast(precision, device):
            logits = forward(input_ids=input_ids).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets)
        if timeline is not None:
            timeline.mark()
        loss.backward()  # type: ignore[no-untyped-call]
        if timeline is not None:
            timeline.mark()
        if probe is not None:
            probe.after_backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if timeline is not None:
            timeline.mark()
        if probe is not None:
            probe.after_optimizer()
        losses.append(loss.detach())
    return torch.stack(losses).float()


def reference_losses(
    model: PreTrainedModel, input_ids: torch.Tensor, steps: int, config: TrainConfig
) -> list[float]:
    """The plain loop the timed one is compared with: the model's own loss
    (labels = inputs), default AdamW, nothing else."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses = []
    for _ in range(steps):
        with _autocast(config.precision, input_ids.device):
            loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))
    return losses


GateKind = Literal["exact", "within_tolerance", "mismatch"]


@dataclass(frozen=True, slots=True)
class GateReport:
    """Whether the loop trains, with the evidence behind the verdict."""

    kind: GateKind
    losses: list[float]
    reference_losses: list[float]
    max_relative_gap: float
    grad_norms: list[float]
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.kind != "mismatch" and not self.failures


def _examples(names: list[str]) -> str:
    return ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def evaluate_gate(
    losses: list[float], reference: list[float], probe: Probe, rtol: float
) -> GateReport:
    failures = []
    if not all(math.isfinite(v) for v in losses):
        failures.append("loss is not finite")
    if probe.frozen:
        failures.append(f"{len(probe.frozen)} parameters are frozen ({_examples(probe.frozen)})")
    never_grad = set(probe.zero_grad[0]).intersection(*probe.zero_grad[1:]) - set(probe.frozen)
    if never_grad:
        failures.append(
            f"{len(never_grad)} trainable parameters never received a gradient "
            f"({_examples(sorted(never_grad))})"
        )
    never_changed = set(probe.unchanged[0]).intersection(*probe.unchanged[1:]) - set(probe.frozen)
    if never_changed:
        failures.append(
            f"{len(never_changed)} trainable parameters never changed "
            f"({_examples(sorted(never_changed))})"
        )
    gaps = [abs(a - b) / max(abs(b), 1e-12) for a, b in zip(losses, reference, strict=True)]
    worst = max(gaps) if all(math.isfinite(g) for g in gaps) else float("inf")
    kind: GateKind = (
        "exact" if worst == 0.0 else "within_tolerance" if worst <= rtol else "mismatch"
    )
    return GateReport(kind, losses, reference, worst, probe.grad_norms, failures)


@contextmanager
def _fault(model: PreTrainedModel, fault: TrainFault) -> Iterator[None]:
    """Apply a structural fault to the model for the duration of the block."""
    frozen = []
    if fault == "frozen_layers":
        keep = model.get_input_embeddings().weight
        for p in model.parameters():
            if p is not keep and p.requires_grad:
                p.requires_grad_(False)
                frozen.append(p)
    try:
        yield
    finally:
        for p in frozen:
            p.requires_grad_(True)


@dataclass(frozen=True, slots=True)
class TrainRunSamples:
    """Raw timings of one measured window of `steps_per_run` steps."""

    phase_ms: dict[str, list[float]]  # phase -> one value per step
    e2e_wall_ms: float
    peak_memory_bytes: int | None

    @property
    def step_ms(self) -> list[float]:
        return [sum(values) for values in zip(*(self.phase_ms[p] for p in PHASES), strict=True)]


def forward_flops_per_token(model: torch.nn.Module, seq_len: int) -> float:
    """Analytic forward FLOPs per token: 2 per weight of every linear layer,
    plus the two attention matmuls (scores and weighted sum), each
    2 x seq_len x hidden per layer and token. Norms, activations and
    embedding lookups are ignored. Training is counted as 3x this."""
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    config: Any = model.config
    attention = 4 * config.num_hidden_layers * seq_len * config.hidden_size
    return float(2 * linear + attention)


@dataclass
class TrainResult:
    config: TrainConfig
    input_fingerprint: str
    gate: GateReport
    forward_flops_per_token: float
    runs: list[TrainRunSamples] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        cfg = self.config
        steps = [ms for r in self.runs for ms in r.step_ms]
        e2e = [r.e2e_wall_ms for r in self.runs]
        tokens_per_s = cfg.batch_size * cfg.seq_len / (metrics.mean(steps) / 1000.0)
        achieved_tflops = 3 * self.forward_flops_per_token * tokens_per_s / 1e12
        out: dict[str, Any] = {
            "gate": self.gate.kind,
            "num_runs": len(self.runs),
            "step_samples": len(steps),
            "step_ms_p50": metrics.percentile(steps, 50),
            "step_ms_p95": metrics.percentile(steps, 95),
            "step_ms_p99": metrics.percentile(steps, 99),
            "e2e_wall_ms_mean": metrics.mean(e2e),
            "e2e_wall_ms_cv": metrics.coefficient_of_variation(e2e),
            "tokens_per_s": tokens_per_s,
            "achieved_tflops": achieved_tflops,
            "peak_memory_bytes": max((r.peak_memory_bytes or 0) for r in self.runs) or None,
        }
        for phase in PHASES:
            values = [ms for r in self.runs for ms in r.phase_ms[phase]]
            out[f"{phase}_ms_p50"] = metrics.percentile(values, 50)
        if cfg.peak_tflops is not None:
            out["mfu"] = achieved_tflops / cfg.peak_tflops
        return out


def _measure_window(
    forward: Callable[..., Any],
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    config: TrainConfig,
) -> TrainRunSamples:
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timeline = EventTimeline(device)
    wall = WallClock(device)
    wall.start()
    train_steps(forward, optimizer, input_ids, config.steps_per_run, config.precision, timeline)
    timeline.mark()  # closes the last step's "other" gap
    e2e_wall_ms = wall.stop()
    gaps = timeline.finish()
    phase_ms = {phase: gaps[i :: len(PHASES)] for i, phase in enumerate(PHASES)}
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return TrainRunSamples(phase_ms, e2e_wall_ms, peak)


def run_train_benchmark(
    config: TrainConfig,
    model: PreTrainedModel,
    special_ids: set[int],
    device: torch.device,
    loss_rtol: float = DEFAULT_LOSS_RTOL,
) -> TrainResult:
    """Gate, warm up, measure. The model's weights are restored afterwards."""
    vocab_size = model.get_input_embeddings().weight.shape[0]
    batch = make_prompts(config.batch_size, config.seq_len, vocab_size, config.seed, special_ids)
    input_ids = batch.to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    model.train()
    try:
        reference = reference_losses(model, input_ids, config.check_steps, config)
        model.load_state_dict(initial)
        with _fault(model, config.fault):
            lr = 0.0 if config.fault == "zero_lr" else config.learning_rate
            fused = config.fused_optimizer and device.type == "cuda"
            optimizer = _make_optimizer(model, lr, fused)
            forward: Callable[..., Any] = model
            if config.compile and device.type == "cuda":
                forward = torch.compile(model)
            probe = Probe(model)
            losses = train_steps(
                forward, optimizer, input_ids, config.check_steps, config.precision, probe=probe
            ).tolist()
            result = TrainResult(
                config=config,
                input_fingerprint=fingerprint(batch),
                gate=evaluate_gate(losses, reference, probe, loss_rtol),
                forward_flops_per_token=forward_flops_per_token(model, config.seq_len),
            )
            if not result.gate.passed:
                return result  # a loop that does not train is never timed
            train_steps(forward, optimizer, input_ids, config.warmup_steps, config.precision)
            for _ in range(config.measure_runs):
                result.runs.append(_measure_window(forward, optimizer, input_ids, config))
            return result
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        model.train(was_training)


TrainLoader = Callable[[TrainConfig, torch.device], tuple[PreTrainedModel, set[int]]]


@dataclasses.dataclass(frozen=True, slots=True)
class TrainOptions:
    require_cuda: bool = True
    cv_threshold: float = DEFAULT_CV_THRESHOLD
    loss_rtol: float = DEFAULT_LOSS_RTOL


def classify_train_result(result: TrainResult, cv_threshold: float) -> tuple[RunStatus, str]:
    gate = result.gate
    if not gate.passed:
        reasons = list(gate.failures)
        if gate.kind == "mismatch":
            reasons.append(
                "loss differs from the reference loop by up to "
                f"{gate.max_relative_gap:.2e} (relative)"
            )
        return RunStatus.WRONG_RESULT, "; ".join(reasons)
    cv = result.summary()["e2e_wall_ms_cv"]
    if cv > cv_threshold:
        return RunStatus.UNSTABLE, f"e2e CV {cv:.3f} exceeds threshold {cv_threshold:.3f}"
    return RunStatus.OK, ""


def run_train_task(
    config: TrainConfig,
    options: TrainOptions | None = None,
    loader: TrainLoader = load_train_model,
) -> dict[str, Any]:
    """Run one training configuration and return its record (never raises for
    run failures). Same record shape and status classes as inference runs."""
    options = options or TrainOptions()
    run_id = new_run_id()
    record: dict[str, Any] = {
        "run_id": run_id,
        "task": "train_step",
        "config": config.to_dict(),
        "manifest": build_manifest(run_id),
        "options": dataclasses.asdict(options),
    }
    try:
        if options.require_cuda and not torch.cuda.is_available():
            raise EnvironmentProblem("CUDA is not available (use --allow-cpu for a CPU smoke run)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, special_ids = loader(config, device)
        record["model_revision"] = getattr(model.config, "_commit_hash", None)
        result = run_train_benchmark(config, model, special_ids, device, options.loss_rtol)
    except Exception as exc:
        record |= {
            "status": classify_exception(exc).value,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return record
    status, reason = classify_train_result(result, options.cv_threshold)
    record |= {
        "status": status.value,
        "reason": reason,
        "input_fingerprint": result.input_fingerprint,
        "gate": dataclasses.asdict(result.gate),
        "forward_flops_per_token": result.forward_flops_per_token,
    }
    if result.runs:
        record |= {
            "summary": result.summary(),
            "raw": {
                "phase_ms": [r.phase_ms for r in result.runs],
                "e2e_wall_ms": [r.e2e_wall_ms for r in result.runs],
                "peak_memory_bytes": [r.peak_memory_bytes for r in result.runs],
            },
        }
    return record
