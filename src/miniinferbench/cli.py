"""Command-line entry point (`mib`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniinferbench import __version__
from miniinferbench.config import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mib", description="MiniInferBench")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run one benchmark configuration and print a summary")
    run.add_argument("--config", type=Path, help="JSON config file (defaults are used if omitted)")
    return parser


def cmd_run(config_path: Path | None) -> int:
    # Heavy imports stay inside the command so `mib --version` starts instantly.
    from miniinferbench.bench import default_device, load_model, run_benchmark

    config = BenchmarkConfig.load(config_path) if config_path else BenchmarkConfig()
    device = default_device()
    model, special_ids = load_model(config, device)
    result = run_benchmark(config, model, special_ids, device)
    print(json.dumps({"config": config.to_dict(), **result.summary()}, indent=2))
    return 0 if result.correctness.matches else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.config)
    build_parser().print_help()
    return 0
