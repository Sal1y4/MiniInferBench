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
    return parser


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
    if args.command == "compare":
        _print_comparison(args.results, args.baseline, args.candidate)
        return 0
    build_parser().print_help()
    return 0
