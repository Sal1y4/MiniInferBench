"""Run `mib gen` under several transformers versions, interleaved.

Each version gets its own virtual environment under `.venvs/` (created with
uv, so the large torch files are shared through uv's cache, not copied). The
versions are then run round-robin, `--rounds` times, so slow drift of the
machine hits every version alike. Afterwards:

    mib check generate@<v1> generate@<v2> ... --results results/versions.jsonl

Usage: python scripts/version_sweep.py --versions 4.57.1 5.2.0 5.4.0 \\
           --torch 2.11.0 --index https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, check=False).returncode


def ensure_env(version: str, torch_version: str, index: str, python: str) -> Path | None:
    env = ROOT / ".venvs" / f"tf-{version}"
    mib = env / "bin" / "mib"
    if mib.exists():
        return mib
    py = str(env / "bin" / "python")
    steps = [
        ("uv", "venv", "--python", python, str(env)),
        ("uv", "pip", "install", "--python", py, f"torch=={torch_version}", "--index-url", index),
        ("uv", "pip", "install", "--python", py, f"transformers=={version}", "numpy"),
        ("uv", "pip", "install", "--python", py, "--no-deps", "-e", "."),
    ]
    for step in steps:
        if run(*step) != 0:
            print(f"!! could not build the environment for transformers {version}", flush=True)
            return None
    return mib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", required=True, help="oldest first")
    parser.add_argument("--torch", required=True, help="torch version, the same in every env")
    parser.add_argument("--index", required=True, help="wheel index URL for torch")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--config", default="configs/regression_baseline.json")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--results", default="results/versions.jsonl")
    args = parser.parse_args()

    envs = {}
    for version in args.versions:
        mib = ensure_env(version, args.torch, args.index, args.python)
        if mib is not None:
            envs[version] = mib
    if len(envs) < 2:
        print("need at least two working environments", file=sys.stderr)
        return 1
    failed: dict[str, int] = {}
    for round_index in range(args.rounds):
        for version, mib in envs.items():
            if failed.get(version, 0) >= 2:
                continue  # broken under this version; the records say why
            print(f"== round {round_index}, transformers {version}", flush=True)
            code = run(
                str(mib), "gen", "--config", args.config, "--results", args.results,
                "--label", f"generate@{version}",
            )  # fmt: skip
            if code != 0:
                failed[version] = failed.get(version, 0) + 1
    labels = " ".join(f"generate@{v}" for v in envs)
    print(f"done. next: mib check {labels} --results {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
