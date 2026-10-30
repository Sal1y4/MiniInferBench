"""Run status classes: every run ends in exactly one of these.

The point is to separate *why* a run produced no trustworthy number. A crash
in our code, a machine without enough memory, a wrong answer, and a noisy
measurement need different fixes, so they must not collapse into "failed".
"""

from __future__ import annotations

from enum import StrEnum

import torch


class RunStatus(StrEnum):
    OK = "ok"  # finished, output correct, measurements stable
    WRONG_RESULT = "wrong_result"  # finished, output differs from the reference
    UNSTABLE = "unstable"  # finished and correct, but repeats vary too much
    PROGRAM_ERROR = "program_error"  # exception raised by benchmark code
    ENV_ERROR = "env_error"  # out of memory, no CUDA, download/IO failure
    SKIPPED = "skipped"  # not run, e.g. memory estimate over budget


class EnvironmentProblem(RuntimeError):
    """Raised by harness preflight checks for problems outside our code."""


# Substrings of RuntimeError messages that point at the CUDA setup rather than
# at our code, e.g. a wheel built without kernels for this GPU.
_CUDA_ENV_MESSAGES = (
    "no kernel image is available",
    "CUDA driver version is insufficient",
    "CUDA error: out of memory",
    "Found no NVIDIA driver",
)


def classify_exception(exc: BaseException) -> RunStatus:
    """Map an exception to ENV_ERROR or PROGRAM_ERROR."""
    if isinstance(exc, torch.OutOfMemoryError | EnvironmentProblem | OSError | ImportError):
        return RunStatus.ENV_ERROR
    if isinstance(exc, RuntimeError) and any(m in str(exc) for m in _CUDA_ENV_MESSAGES):
        return RunStatus.ENV_ERROR
    return RunStatus.PROGRAM_ERROR
