"""Decision rules: given two recorded runs, did a metric change?

`mib compare` needs a rule that turns two sets of noisy timings into a
verdict. Several rules are in common use (a fixed percentage, a band derived
from the runs' own spread, confidence intervals, a rank test). They are
collected here behind one signature so that `noise.py` can measure, on real
repeated runs, how often each one raises a false alarm and how small a
slowdown it can detect.

Every rule returns a `Verdict`: +1 (candidate slower), -1 (candidate faster)
or 0 (no change claimed). The unit of evidence is one value per measured run
(for inter-token latency: that run's median or P99). Individual token gaps
inside a run are not independent samples, so they are not pooled.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, Literal

from miniinferbench import metrics

Record = dict[str, Any]
Verdict = Literal[-1, 0, 1]
Rule = Callable[[Record, Record, str], Verdict]

# Per-run metrics a rule can judge, and the summary field `compare_records`
# uses for the same quantity.
SUMMARY_FIELD = {
    "ttft_ms": "ttft_ms_p50",
    "itl_ms_p50": "itl_ms_p50",
    "itl_ms_p99": "itl_ms_p99",
    "e2e_wall_ms": "e2e_wall_ms_mean",
}
METRICS = tuple(SUMMARY_FIELD)


def run_samples(record: Record, metric: str) -> list[float]:
    """One value per measured run for `metric`, from the record's raw samples."""
    raw = record["raw"]
    if metric in ("ttft_ms", "e2e_wall_ms"):
        return [float(v) for v in raw[metric]]
    if metric in ("itl_ms_p50", "itl_ms_p99"):
        q = 50.0 if metric == "itl_ms_p50" else 99.0
        return [metrics.percentile(run, q) for run in raw["itl_ms"]]
    raise ValueError(f"unknown metric {metric!r}")


def _sign(value: float) -> Verdict:
    return 1 if value > 0 else -1 if value < 0 else 0


def _relative_change(a: Sequence[float], b: Sequence[float]) -> float:
    return metrics.mean(b) / metrics.mean(a) - 1.0


def fixed_threshold(threshold: float) -> Rule:
    """Changed if the means differ by more than a fixed fraction (e.g. 0.05)."""

    def rule(baseline: Record, candidate: Record, metric: str) -> Verdict:
        change = _relative_change(run_samples(baseline, metric), run_samples(candidate, metric))
        return _sign(change) if abs(change) > threshold else 0

    return rule


def cv_band(baseline: Record, candidate: Record, metric: str) -> Verdict:
    """The rule `mib compare` uses: max(2%, 3 x the larger end-to-end CV)."""
    from miniinferbench.harness import compare_records

    field = SUMMARY_FIELD[metric]
    for diff in compare_records(baseline, candidate):
        if diff.name == field:
            return _sign(diff.relative_change) if diff.significant else 0
    raise ValueError(f"metric {metric!r} missing from one of the records")


# Two-sided 95% critical values of Student's t for 1..30 degrees of freedom.
_T_975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    """95% t-interval for the mean. Needs at least two values."""
    n = len(values)
    if n < 2:
        raise ValueError("a confidence interval needs at least two values")
    m = metrics.mean(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    t = _T_975[n - 2] if n - 1 <= len(_T_975) else 1.96
    half = t * sd / math.sqrt(n)
    return m - half, m + half


def ci_overlap(baseline: Record, candidate: Record, metric: str) -> Verdict:
    """Changed if the two 95% confidence intervals of the mean do not overlap."""
    a_low, a_high = confidence_interval(run_samples(baseline, metric))
    b_low, b_high = confidence_interval(run_samples(candidate, metric))
    if b_low > a_high:
        return 1
    if b_high < a_low:
        return -1
    return 0


def bootstrap(resamples: int = 500, seed: int = 0) -> Rule:
    """Changed if the 95% percentile-bootstrap interval of the relative change
    of the means excludes zero. Seeded, so a verdict is reproducible."""

    def rule(baseline: Record, candidate: Record, metric: str) -> Verdict:
        a, b = run_samples(baseline, metric), run_samples(candidate, metric)
        rng = random.Random(seed)
        changes = sorted(
            _relative_change(rng.choices(a, k=len(a)), rng.choices(b, k=len(b)))
            for _ in range(resamples)
        )
        low = metrics.percentile(changes, 2.5)
        high = metrics.percentile(changes, 97.5)
        if low > 0:
            return 1
        if high < 0:
            return -1
        return 0

    return rule


@cache
def _arrangements(n_a: int, n_b: int, u: int) -> int:
    """Number of orderings of n_a + n_b distinct values whose U statistic is `u`."""
    if u < 0 or u > n_a * n_b:
        return 0
    if n_a == 0 or n_b == 0:
        return 1 if u == 0 else 0
    return _arrangements(n_a - 1, n_b, u - n_b) + _arrangements(n_a, n_b - 1, u)


def mann_whitney_p(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided exact p-value of the Mann-Whitney U test, and U for `b`.

    U counts the (a, b) pairs where b is larger (ties count one half). The
    null distribution assumes no ties, which holds for timing samples.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        raise ValueError("both samples must be non-empty")
    u = sum(1.0 if y > x else 0.5 if y == x else 0.0 for x in a for y in b)
    total = math.comb(n_a + n_b, n_a)
    extreme = min(u, n_a * n_b - u)
    tail = sum(_arrangements(n_a, n_b, k) for k in range(math.floor(extreme) + 1))
    return min(1.0, 2.0 * tail / total), u


def rank_test(alpha: float = 0.05) -> Rule:
    """Changed if the exact Mann-Whitney U test rejects at level `alpha`."""

    def rule(baseline: Record, candidate: Record, metric: str) -> Verdict:
        a, b = run_samples(baseline, metric), run_samples(candidate, metric)
        p, u = mann_whitney_p(a, b)
        if p >= alpha:
            return 0
        return 1 if u > len(a) * len(b) / 2 else -1

    return rule


RULES: dict[str, Rule] = {
    "fixed_5pct": fixed_threshold(0.05),
    "fixed_7pct": fixed_threshold(0.07),
    "cv_band": cv_band,
    "ci_overlap": ci_overlap,
    "bootstrap": bootstrap(),
    "rank_test": rank_test(),
}
