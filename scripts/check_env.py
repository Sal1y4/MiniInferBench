"""Print the software and GPU environment as JSON and run a tiny GPU sanity check.

Usage: python scripts/check_env.py
"""

from __future__ import annotations

import json
import platform
import sys
from typing import Any


def collect() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None

    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    info |= {
        "gpu_name": props.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_memory_gib": round(props.total_memory / 2**30, 2),
        "arch_list": torch.cuda.get_arch_list(),
        "arch_supported_by_wheel": arch in torch.cuda.get_arch_list(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }

    # A real kernel launch catches "no kernel image is available" errors
    # that is_available() alone does not.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    result = (a @ b).float().sum().item()
    torch.cuda.synchronize()
    info["bf16_matmul_ok"] = bool(result == result)  # False only for NaN
    return info


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
