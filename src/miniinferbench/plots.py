"""Static figures for one sweep tag, written as PNGs for the README.

Every figure is drawn from recorded runs with status `ok` (or `unstable`,
marked as such); skipped and failed points are left out rather than guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from miniinferbench.results import read_records

# Validated categorical order (blue, orange, aqua, yellow) and recessive inks.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

Record = dict[str, Any]


def load_points(results: Path, tag: str) -> list[Record]:
    points = [
        r
        for r in read_records(results)
        if r.get("sweep_tag") == tag and r.get("status") in ("ok", "unstable")
    ]
    if not points:
        raise ValueError(f"no usable runs for tag {tag!r} in {results}")
    return points


def _style(ax: Axes, title: str, xlabel: str, ylabel: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY)


def _lines(
    ax: Axes,
    points: Sequence[Record],
    x_key: str,
    series_key: str,
    y: Callable[[Record], float],
    fixed: dict[str, int],
    label: Callable[[int], str],
) -> None:
    chosen = [p for p in points if all(p["config"][k] == v for k, v in fixed.items())]
    for color, value in zip(
        SERIES, sorted({p["config"][series_key] for p in chosen}), strict=False
    ):
        rows = sorted(
            (p for p in chosen if p["config"][series_key] == value),
            key=lambda p: p["config"][x_key],
        )
        xs = [p["config"][x_key] for p in rows]
        ys = [y(p) for p in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label(value))
    ticks = sorted({p["config"][x_key] for p in chosen})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)


def _save(fig: Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_all(results: Path, tag: str, out_dir: Path, output_len: int = 128) -> list[Path]:
    points = load_points(results, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = points[0]
    gpu = head["manifest"]["env"].get("gpu_name", "CPU")
    model = head["config"]["model_name"].split("/")[-1]
    sub = f"{model}, {head['config']['dtype']}, {gpu}, output {output_len} tokens"
    written = []

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _lines(
        ax,
        points,
        "batch_size",
        "input_len",
        lambda p: p["summary"]["decode_tokens_per_s"],
        {"output_len": output_len},
        lambda v: f"input {v}",
    )
    _style(ax, "Decode throughput vs batch size", "Batch size", "Decode tokens/s", sub)
    written.append(_save(fig, out_dir, "throughput_vs_batch.png"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _lines(
        ax,
        points,
        "batch_size",
        "input_len",
        lambda p: p["summary"]["itl_ms_p50"],
        {"output_len": output_len},
        lambda v: f"input {v}",
    )
    _style(ax, "Inter-token latency (P50) vs batch size", "Batch size", "ITL P50 (ms)", sub)
    ax.set_ylim(bottom=0)
    written.append(_save(fig, out_dir, "latency_vs_batch.png"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _lines(
        ax,
        points,
        "input_len",
        "batch_size",
        lambda p: p["summary"]["prefill_tokens_per_s"],
        {"output_len": output_len},
        lambda v: f"batch {v}",
    )
    _style(
        ax, "Prefill throughput vs input length", "Input length (tokens)", "Prefill tokens/s", sub
    )
    written.append(_save(fig, out_dir, "prefill_vs_input.png"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fixed = [
        p
        for p in points
        if p["config"]["output_len"] == output_len and p["config"]["input_len"] == 512
    ]
    fixed.sort(key=lambda p: p["config"]["batch_size"])
    xs = range(len(fixed))
    width = 0.38
    ax.bar(
        [x - width / 2 for x in xs],
        [p["summary"]["ttft_ms_mean"] for p in fixed],
        width=width,
        color=SERIES[0],
        label="Prefill (TTFT, 512 tokens/seq)",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        [p["summary"]["itl_ms_p50"] for p in fixed],
        width=width,
        color=SERIES[1],
        label="One decode step (ITL P50)",
    )
    ax.set_xticks(list(xs), [str(p["config"]["batch_size"]) for p in fixed])
    _style(ax, "Prefill vs one decode step", "Batch size", "Time (ms)", sub + ", input 512")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    written.append(_save(fig, out_dir, "prefill_vs_decode.png"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    longest = max(p["config"]["output_len"] for p in points)
    picks = [p for p in points if p["config"]["output_len"] == longest]
    picks = [
        min(picks, key=lambda p: (p["config"]["batch_size"], p["config"]["input_len"])),
        max(picks, key=lambda p: (p["config"]["batch_size"], p["config"]["input_len"])),
    ]
    all_samples = [[g for run in p["raw"]["itl_ms"] for g in run] for p in picks]
    lo = int(min(min(s) for s in all_samples))
    hi = max(max(s) for s in all_samples)
    bins = [lo + i * 0.5 for i in range(int((hi - lo) / 0.5) + 2)]  # shared 0.5 ms bins
    for color, p, samples in zip(SERIES, picks, all_samples, strict=False):
        c = p["config"]
        ax.hist(
            samples,
            bins=bins,
            color=color,
            alpha=0.75,
            label=f"batch {c['batch_size']}, input {c['input_len']} (n={len(samples)})",
        )
    _style(
        ax,
        "Inter-token latency distribution",
        "ITL (ms)",
        "Decode steps",
        f"{model}, {gpu}, output {longest} tokens, all measured runs",
    )
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    written.append(_save(fig, out_dir, "itl_distribution.png"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    tokens = [
        p["config"]["batch_size"] * (p["config"]["input_len"] + p["config"]["output_len"])
        for p in points
    ]
    measured = [(p["summary"]["peak_memory_bytes"] or 0) / 2**30 for p in points]
    estimated = [p.get("memory_estimate_bytes", 0) / 2**30 for p in points]
    ax.scatter(tokens, measured, s=36, color=SERIES[0], label="Measured peak (allocator)")
    ax.scatter(tokens, estimated, s=36, color=SERIES[1], marker="s", label="Estimate used to skip")
    _style(
        ax,
        "GPU memory vs tokens held in the KV cache",
        "batch x (input + output) tokens",
        "GiB",
        sub.replace(f", output {output_len} tokens", ", all points"),
    )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    written.append(_save(fig, out_dir, "memory.png"))
    return written
