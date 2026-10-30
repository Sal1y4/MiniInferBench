"""Timers for GPU work that runs asynchronously from Python.

CUDA kernels are queued by the CPU and executed later by the GPU, so reading
`time.perf_counter()` right after a model call measures only how long it took
to *launch* the work. Two correct options exist, and we record both:

- `EventTimeline`: CUDA events recorded on the stream. Their timestamps mark
  when the GPU reached that point in the stream, so the gap between two events
  is the GPU-side time between them (kernel time *plus* any idle gaps where the
  GPU waited for the CPU). Recording an event does not block the CPU.
- `WallClock`: `torch.cuda.synchronize()` followed by `perf_counter()`. This is
  the application-level latency a caller would observe, including Python and
  launch overhead.

Pure kernel execution time is neither of these; it needs a profiler trace.
On CPU-only machines both timers fall back to `perf_counter()` so the same
code paths can be unit-tested.
"""

from __future__ import annotations

import time

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class EventTimeline:
    """Marks points on the device timeline; read the gaps after `finish()`."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._events: list[torch.cuda.Event] = []
        self._cpu_marks: list[float] = []
        self._finished = False

    def mark(self) -> None:
        if self._finished:
            raise RuntimeError("cannot mark a finished timeline")
        if self._device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            event.record()
            self._events.append(event)
        else:
            self._cpu_marks.append(time.perf_counter())

    def finish(self) -> list[float]:
        """Wait for the device and return the gaps between marks in milliseconds."""
        self._finished = True
        if self._device.type == "cuda":
            if len(self._events) < 2:
                return []
            self._events[-1].synchronize()
            return [a.elapsed_time(b) for a, b in zip(self._events, self._events[1:], strict=False)]
        return [
            (b - a) * 1000.0 for a, b in zip(self._cpu_marks, self._cpu_marks[1:], strict=False)
        ]


class WallClock:
    """Synchronized wall-clock interval: `start()`, run work, `stop()` -> ms."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._start: float | None = None

    def start(self) -> None:
        synchronize(self._device)
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("stop() called before start()")
        synchronize(self._device)
        return (time.perf_counter() - self._start) * 1000.0
