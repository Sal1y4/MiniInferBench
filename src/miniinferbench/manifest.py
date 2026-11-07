"""Run manifest: everything needed to explain and repeat a result."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit hash and whether uncommitted changes were present."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def environment() -> dict[str, Any]:
    try:
        import transformers

        transformers_version: str | None = transformers.__version__
    except ImportError:
        transformers_version = None
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers_version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        env |= {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{major}.{minor}",
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    return env


def host_load() -> dict[str, Any]:
    """1-minute load average and CPU count when the run started.

    Decode here is bound by the host launching work, so other busy processes
    on the machine move the numbers; this is the evidence for or against it.
    """
    try:
        load_1m: float | None = os.getloadavg()[0]
    except OSError:
        load_1m = None
    return {"load_avg_1m": load_1m, "cpu_count": os.cpu_count()}


def build_manifest(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **git_state(),
        "env": environment(),
        "host": host_load(),
    }
