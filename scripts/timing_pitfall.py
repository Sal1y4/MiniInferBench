"""Show why GPU timing needs synchronization.

Times the same batch of matmuls three ways:
  1. perf_counter() with no synchronize   -> measures only kernel *launch* time
  2. synchronize() + perf_counter()       -> wall-clock time the caller observes
  3. CUDA events                          -> GPU-side time between two stream points

Usage: python scripts/timing_pitfall.py [--size 4096] [--iters 20]
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this demo needs a CUDA device")

    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)

    def work() -> None:
        for _ in range(args.iters):
            a @ b

    work()  # warmup: cuBLAS heuristics, clocks ramping up
    torch.cuda.synchronize()

    start = time.perf_counter()
    work()
    no_sync_ms = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()  # drain the queue before the next measurement

    torch.cuda.synchronize()
    start = time.perf_counter()
    work()
    torch.cuda.synchronize()
    sync_ms = (time.perf_counter() - start) * 1000

    begin = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    begin.record()
    work()
    end.record()
    end.synchronize()
    event_ms = begin.elapsed_time(end)

    flops = 2 * args.size**3 * args.iters
    print(f"{args.iters} x bf16 matmul {args.size}x{args.size} on {torch.cuda.get_device_name(0)}")
    print(f"  no synchronize : {no_sync_ms:9.3f} ms   <- launch time only, wrong")
    print(f"  synchronize    : {sync_ms:9.3f} ms   ({flops / sync_ms / 1e9:.1f} TFLOP/s)")
    print(f"  CUDA events    : {event_ms:9.3f} ms   ({flops / event_ms / 1e9:.1f} TFLOP/s)")


if __name__ == "__main__":
    main()
