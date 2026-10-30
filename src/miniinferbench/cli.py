"""Command-line entry point (`mib`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from miniinferbench import __version__
from miniinferbench.config import BenchmarkConfig

DEFAULT_RESULTS = Path("results/runs.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mib", description="MiniInferBench")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run one configuration and append a record")
    run.add_argument("--config", type=Path, help="JSON config file (defaults if omitted)")
    run.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    run.add_argument("--allow-cpu", action="store_true", help="allow a CPU smoke run")

    rerun = sub.add_parser("rerun", help="repeat a recorded run with its exact config")
    rerun.add_argument("run_id")
    rerun.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    rerun.add_argument("--allow-cpu", action="store_true")

    compare = sub.add_parser("compare", help="compare two recorded runs against noise")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    sweep = sub.add_parser("sweep", help="run a grid of configurations (resumable by --tag)")
    sweep.add_argument("--config", type=Path, help="base config; grid values override it")
    sweep.add_argument("--tag", required=True, help="names one curve; reuse it to resume")
    sweep.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    sweep.add_argument("--input", type=int, nargs="+", default=[128, 512, 1024])
    sweep.add_argument("--output", type=int, nargs="+", default=[32, 128, 256])
    sweep.add_argument("--results", type=Path, default=Path("results/sweep.jsonl"))

    table = sub.add_parser("table", help="print one sweep tag as a Markdown table")
    table.add_argument("--tag", required=True)
    table.add_argument("--results", type=Path, default=Path("results/sweep.jsonl"))
    return parser


def _print_table(results: Path, tag: str) -> None:
    from miniinferbench.results import read_records

    cols = (
        "ttft_ms_mean",
        "itl_ms_p50",
        "itl_ms_p99",
        "decode_tokens_per_s",
        "prefill_tokens_per_s",
        "peak_memory_bytes",
    )
    print("| bs | in | out | status | " + " | ".join(cols) + " |")
    print("|" + " --- |" * (4 + len(cols)))
    rows = [r for r in read_records(results) if r.get("sweep_tag") == tag]
    rows.sort(
        key=lambda r: (
            r["config"]["batch_size"],
            r["config"]["input_len"],
            r["config"]["output_len"],
        )
    )
    for r in rows:
        c, summary = r["config"], r.get("summary", {})
        cells = []
        for col in cols:
            value = summary.get(col)
            if value is None:
                cells.append("")
            elif col == "peak_memory_bytes":
                cells.append(f"{value / 2**20:.0f} MiB")
            else:
                cells.append(f"{value:.2f}")
        head = f"| {c['batch_size']} | {c['input_len']} | {c['output_len']} | {r['status']} | "
        print(head + " | ".join(cells) + " |")


def _execute(config: BenchmarkConfig, results: Path, allow_cpu: bool) -> dict[str, Any]:
    from miniinferbench.harness import HarnessOptions, run_task
    from miniinferbench.results import append_record

    record = run_task(config, HarnessOptions(require_cuda=not allow_cpu))
    append_record(results, record)
    print(f"run {record['run_id']}: {record['status']} {record.get('reason', '')}".rstrip())
    if "summary" in record:
        print(json.dumps(record["summary"], indent=2))
    return record


def _print_comparison(results: Path, baseline_id: str, candidate_id: str) -> None:
    from miniinferbench.harness import compare_records
    from miniinferbench.results import find_record

    diffs = compare_records(find_record(results, baseline_id), find_record(results, candidate_id))
    for d in diffs:
        verdict = "CHANGED" if d.significant else "within noise"
        print(
            f"{d.name:18s} {d.baseline:10.3f} -> {d.candidate:10.3f} "
            f"({d.relative_change:+.1%}, band ±{d.noise_band:.1%}) {verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = BenchmarkConfig.load(args.config) if args.config else BenchmarkConfig()
        record = _execute(config, args.results, args.allow_cpu)
        return 0 if record["status"] == "ok" else 1
    if args.command == "rerun":
        from miniinferbench.results import find_record

        original = find_record(args.results, args.run_id)
        record = _execute(
            BenchmarkConfig.from_dict(original["config"]), args.results, args.allow_cpu
        )
        if "summary" in record and "summary" in original:
            _print_comparison(args.results, args.run_id, record["run_id"])
        return 0 if record["status"] == "ok" else 1
    if args.command == "sweep":
        from miniinferbench.sweep import SweepGrid, run_sweep

        base = BenchmarkConfig.load(args.config) if args.config else BenchmarkConfig()
        grid = SweepGrid(args.batch, args.input, args.output)
        records = run_sweep(base, grid, args.results, args.tag)
        statuses = [r["status"] for r in records]
        print({s: statuses.count(s) for s in sorted(set(statuses))})
        return 0
    if args.command == "table":
        _print_table(args.results, args.tag)
        return 0
    if args.command == "compare":
        _print_comparison(args.results, args.baseline, args.candidate)
        return 0
    build_parser().print_help()
    return 0
