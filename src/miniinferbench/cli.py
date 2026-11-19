"""Command-line entry point (`mib`)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
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

    train = sub.add_parser("train", help="run one training-step config and append a record")
    train.add_argument("--config", type=Path, help="JSON TrainConfig file (defaults if omitted)")
    train.add_argument("--results", type=Path, default=Path("results/train.jsonl"))
    train.add_argument("--allow-cpu", action="store_true", help="allow a CPU smoke run")

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

    trace = sub.add_parser("trace", help="capture and analyze a profiler trace for one config")
    trace.add_argument("--config", type=Path, help="JSON config file (defaults if omitted)")
    trace.add_argument("--out", type=Path, default=Path("traces"))

    plot = sub.add_parser("plot", help="write the figures for one sweep tag")
    plot.add_argument("--tag", required=True)
    plot.add_argument("--results", type=Path, default=Path("results/sweep.jsonl"))
    plot.add_argument("--out", type=Path, default=Path("docs/figures"))

    modes = sub.add_parser("plot-modes", help="compare decode modes, one sweep tag each")
    modes.add_argument(
        "--tag",
        action="append",
        required=True,
        metavar="LABEL=TAG",
        help="e.g. --tag eager=my-eager-sweep --tag cuda_graph=my-graph-sweep",
    )
    modes.add_argument("--results", type=Path, default=Path("results/sweep.jsonl"))
    modes.add_argument("--out", type=Path, default=Path("docs/figures"))

    repeat = sub.add_parser(
        "repeat", help="run one config many times, optionally with known slowdowns"
    )
    repeat.add_argument("--config", type=Path, help="JSON config file (defaults if omitted)")
    repeat.add_argument("--tag", required=True, help="names the experiment; reuse it to resume")
    repeat.add_argument("--repeats", type=int, default=20, help="records per delay")
    repeat.add_argument(
        "--delay-us",
        type=int,
        nargs="+",
        default=[0],
        help="busy-wait per decode step; 0 is the unchanged baseline",
    )
    repeat.add_argument("--results", type=Path, default=Path("results/repeat.jsonl"))
    repeat.add_argument("--allow-cpu", action="store_true")

    noise = sub.add_parser(
        "noise", help="false-positive and detection rates of the comparison rules"
    )
    noise.add_argument("--tag", required=True)
    noise.add_argument("--results", type=Path, default=Path("results/repeat.jsonl"))
    noise.add_argument("--json", type=Path, help="also write the full report as JSON")
    noise.add_argument("--figures", type=Path, help="also write detection figures here")

    cases = sub.add_parser(
        "cases", help="run a diagnosis suite: baseline + candidate evidence per case"
    )
    cases.add_argument("--suite", type=Path, default=Path("configs/diagnosis_suite.json"))
    cases.add_argument("--tag", required=True)
    cases.add_argument("--results", type=Path, default=Path("results/diagnosis.jsonl"))
    cases.add_argument("--traces", type=Path, default=Path("traces/diagnosis"))
    cases.add_argument("--keep-traces", action="store_true")
    cases.add_argument("--allow-cpu", action="store_true")
    diagnose = sub.add_parser("diagnose", help="score the reference diagnosers on collected cases")
    diagnose.add_argument("--tag", required=True)
    diagnose.add_argument("--results", type=Path, default=Path("results/diagnosis.jsonl"))
    diagnose.add_argument("--json", type=Path, help="also write the full report as JSON")
    gen = sub.add_parser(
        "gen", help="time and trace one generation path: real generate() or the minimal loop"
    )
    gen.add_argument("--config", type=Path, help="JSON config file (defaults if omitted)")
    gen.add_argument("--path", choices=("generate", "loop"), default="generate")
    gen.add_argument("--label", help="group name for `mib check` (default: path@transformers)")
    gen.add_argument("--results", type=Path, default=Path("results/gen.jsonl"))
    gen.add_argument("--traces", type=Path, default=Path("traces/gen"))
    gen.add_argument("--keep-trace", action="store_true")
    gen.add_argument("--allow-cpu", action="store_true")
    check = sub.add_parser(
        "check", help="is one group of `mib gen` records slower than another, and why"
    )
    check.add_argument(
        "labels", nargs="+", help="two or more labels, oldest first; neighbours are compared"
    )
    check.add_argument("--results", type=Path, default=Path("results/gen.jsonl"))
    check.add_argument("--json", type=Path, help="also write the comparisons as JSON")
    table = sub.add_parser("table", help="print one sweep tag as a Markdown table")
    table.add_argument("--tag", required=True)
    table.add_argument("--results", type=Path, default=Path("results/sweep.jsonl"))
    return parser


def _print_table(results: Path, tag: str) -> None:
    from miniinferbench.results import read_records

    cols = (
        "ttft_ms_p50",
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


def cmd_check(results: Path, labels: list[str], json_path: Path | None) -> int:
    from miniinferbench.genbench import compare, format_comparison, load_group

    if len(labels) < 2:
        print("give at least two labels", file=sys.stderr)
        return 2
    groups = {label: load_group(results, label) for label in labels}
    missing = [label for label, group in groups.items() if not group]
    if missing:
        print(f"no usable records in {results} for: {', '.join(missing)}", file=sys.stderr)
        return 1
    pairs = list(itertools.pairwise(labels))
    if len(labels) > 2:
        pairs.append((labels[0], labels[-1]))
    report = []
    for old, new in pairs:
        result = compare(groups[old], groups[new])
        report.append({"baseline": old, "candidate": new} | result)
        print(format_comparison(result, old, new), end="\n\n")
    if json_path:
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_trace(config_path: Path | None, out_dir: Path) -> int:
    from miniinferbench import metrics
    from miniinferbench.bench import default_device, load_model
    from miniinferbench.runner import measure_run
    from miniinferbench.trace import analyze, capture_trace
    from miniinferbench.workload import fingerprint, make_prompts

    config = BenchmarkConfig.load(config_path) if config_path else BenchmarkConfig()
    device = default_device()
    model, special_ids = load_model(config, device)
    vocab = model.get_input_embeddings().weight.shape[0]
    prompts = make_prompts(config.batch_size, config.input_len, vocab, config.seed, special_ids)
    input_ids = prompts.to(device)
    for _ in range(max(config.warmup_runs, 1)):
        measure_run(model, input_ids, config.output_len, config.regression, config.decode_mode)
    # Untraced baseline right before the traced pass, to measure profiler overhead.
    baseline = measure_run(
        model, input_ids, config.output_len, config.regression, config.decode_mode
    )
    name = f"bs{config.batch_size}-in{config.input_len}-out{config.output_len}"
    if config.decode_mode != "eager":
        name += f"-{config.decode_mode}"
    trace_path = out_dir / f"{name}.trace.json"
    capture_trace(
        model,
        input_ids,
        config.output_len,
        trace_path,
        config.regression,
        config.decode_mode,
        config.step_delay_us,
        config.regression_level,
        config.step_delay_sync,
    )
    report = analyze(trace_path)
    untraced_us = metrics.percentile(baseline.itl_ms, 50) * 1000
    report |= {
        "config": config.to_dict(),
        "input_fingerprint": fingerprint(prompts),
        "untraced_itl_us_p50": untraced_us,
        "profiler_overhead": report["step_span_us_median"] / untraced_us - 1,
    }
    (out_dir / f"{name}.analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"trace: {trace_path}")
    keys = (
        "step_span_us_median",
        "step_busy_us_median",
        "step_busy_fraction_median",
        "kernels_per_step_median",
        "untraced_itl_us_p50",
        "profiler_overhead",
        "kernel_launches_total",
        "graph_launches_total",
        "host_wait_calls_total",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print("top kernels:")
    for k in report["top_kernels"][:5]:
        print(f"  {k['total_us']:10.0f} us  x{k['count']:<5d} {k['name'][:90]}")
    return 0


def cmd_noise(results: Path, tag: str, json_path: Path | None, figures: Path | None) -> int:
    from miniinferbench.noise import build_report, format_report, load_groups

    groups = load_groups(results, tag)
    report = build_report(groups)
    print(format_report(report))
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if figures:
        from miniinferbench.plots import plot_detection

        head = groups[0][0]
        config = head["config"]
        subtitle = (
            f"{config['model_name'].split('/')[-1]}, batch {config['batch_size']}, "
            f"input {config['input_len']}, output {config['output_len']}, "
            f"{report['baseline_records']} records x {report['runs_per_record']} runs"
        )
        for metric in report["metrics"]:
            print(plot_detection(report, metric, figures, subtitle))
    return 0


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
    if args.command == "train":
        from miniinferbench.results import append_record
        from miniinferbench.training import TrainConfig, TrainOptions, run_train_task

        train_config = TrainConfig.load(args.config) if args.config else TrainConfig()
        record = run_train_task(train_config, TrainOptions(require_cuda=not args.allow_cpu))
        append_record(args.results, record)
        print(f"run {record['run_id']}: {record['status']} {record.get('reason', '')}".rstrip())
        if "gate" in record:
            gate = record["gate"]
            print(f"gate: {gate['kind']}, max relative loss gap {gate['max_relative_gap']:.2e}")
        if "summary" in record:
            print(json.dumps(record["summary"], indent=2))
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
    if args.command == "repeat":
        from miniinferbench.noise import run_repeats

        base = BenchmarkConfig.load(args.config) if args.config else BenchmarkConfig()
        records = run_repeats(
            base, args.delay_us, args.repeats, args.results, args.tag, not args.allow_cpu
        )
        statuses = [r["status"] for r in records]
        print({s: statuses.count(s) for s in sorted(set(statuses))})
        return 0
    if args.command == "noise":
        return cmd_noise(args.results, args.tag, args.json, args.figures)
    if args.command == "cases":
        from miniinferbench.diagnosis import Suite, collect

        written = collect(
            Suite.load(args.suite),
            args.results,
            args.tag,
            args.traces,
            require_cuda=not args.allow_cpu,
            keep_traces=args.keep_traces,
        )
        print(f"{written} cases written to {args.results}")
        return 0
    if args.command == "diagnose":
        from miniinferbench.diagnosis import DIAGNOSERS, format_scores, load_cases, score

        collected = load_cases(args.results, args.tag)
        if not collected:
            print(f"no cases with tag {args.tag!r} in {args.results}", file=sys.stderr)
            return 1
        report = {name: score(collected, diagnoser) for name, diagnoser in DIAGNOSERS.items()}
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(format_scores(report))
        return 0
    if args.command == "gen":
        from miniinferbench.genbench import default_label, run_generation_task
        from miniinferbench.results import append_record

        config = BenchmarkConfig.load(args.config) if args.config else BenchmarkConfig()
        record = run_generation_task(
            config,
            args.path,
            args.traces / "gen.trace.json",
            require_cuda=not args.allow_cpu,
            keep_trace=args.keep_trace,
        )
        record["label"] = args.label or default_label(record)
        append_record(args.results, record)
        print(f"[{record['status']}] {record['label']} -> {args.results}  {record['reason']}")
        return 0 if record["status"] == "ok" else 1
    if args.command == "check":
        return cmd_check(args.results, args.labels, args.json)
    if args.command == "trace":
        return cmd_trace(args.config, args.out)
    if args.command == "plot":
        from miniinferbench.plots import plot_all

        for path in plot_all(args.results, args.tag, args.out):
            print(path)
        return 0
    if args.command == "plot-modes":
        from miniinferbench.plots import plot_modes

        tags = dict(item.split("=", 1) for item in args.tag)
        for path in plot_modes(args.results, tags, args.out):
            print(path)
        return 0
    if args.command == "table":
        _print_table(args.results, args.tag)
        return 0
    if args.command == "compare":
        _print_comparison(args.results, args.baseline, args.candidate)
        return 0
    build_parser().print_help()
    return 0
