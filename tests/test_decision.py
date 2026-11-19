import math
from typing import Any

import pytest

from miniinferbench import metrics
from miniinferbench.decision import (
    RULES,
    ci_overlap,
    confidence_interval,
    cv_band,
    fixed_threshold,
    mann_whitney_p,
    rank_test,
    run_samples,
)


def make_record(e2e: list[float], itl_scale: float = 1.0) -> dict[str, Any]:
    """A minimal ok record: per-run e2e times and a fixed ITL pattern per run."""
    itl = [[10.0 * itl_scale, 11.0 * itl_scale, 30.0 * itl_scale] for _ in e2e]
    pooled = [gap for run in itl for gap in run]
    return {
        "run_id": f"r{sum(e2e)}",
        "status": "ok",
        "input_fingerprint": "abc",
        "raw": {"ttft_ms": [5.0] * len(e2e), "itl_ms": itl, "e2e_wall_ms": e2e},
        "summary": {
            "ttft_ms_p50": 5.0,
            "itl_ms_p50": metrics.percentile(pooled, 50),
            "itl_ms_p99": metrics.percentile(pooled, 99),
            "e2e_wall_ms_mean": metrics.mean(e2e),
            "e2e_wall_ms_cv": metrics.coefficient_of_variation(e2e),
        },
    }


BASE = make_record([100.0, 101.0, 99.0, 100.5, 99.5])
SAME = make_record([100.2, 100.8, 99.1, 100.4, 99.6])
SLOWER = make_record([110.0, 111.0, 109.0, 110.5, 109.5])


def test_run_samples_gives_one_value_per_run() -> None:
    assert run_samples(BASE, "e2e_wall_ms") == [100.0, 101.0, 99.0, 100.5, 99.5]
    assert run_samples(BASE, "itl_ms_p50") == [11.0] * 5
    assert run_samples(BASE, "itl_ms_p99") == pytest.approx([29.62] * 5)
    with pytest.raises(ValueError, match="unknown metric"):
        run_samples(BASE, "nope")


@pytest.mark.parametrize("name", list(RULES))
def test_every_rule_sees_a_clear_slowdown_and_its_direction(name: str) -> None:
    rule = RULES[name]
    assert rule(BASE, SLOWER, "e2e_wall_ms") == 1
    assert rule(SLOWER, BASE, "e2e_wall_ms") == -1
    assert rule(BASE, SAME, "e2e_wall_ms") == 0


def test_fixed_threshold_ignores_the_spread() -> None:
    # +3% with almost no noise: below a 5% threshold, yet clearly real.
    tight = make_record([103.0, 103.01, 102.99, 103.0, 103.0])
    base = make_record([100.0, 100.01, 99.99, 100.0, 100.0])
    assert fixed_threshold(0.05)(base, tight, "e2e_wall_ms") == 0
    assert ci_overlap(base, tight, "e2e_wall_ms") == 1
    assert rank_test()(base, tight, "e2e_wall_ms") == 1


def test_cv_band_matches_mib_compare() -> None:
    # Band is max(2%, 3 x CV). CV here is ~0.8%, so the band is ~2.4%.
    assert cv_band(BASE, make_record([102.0, 103.0, 101.0, 102.5, 101.5]), "e2e_wall_ms") == 0
    assert cv_band(BASE, make_record([103.0, 104.0, 102.0, 103.5, 102.5]), "e2e_wall_ms") == 1
    # ITL is judged against the same end-to-end band.
    assert cv_band(BASE, make_record(BASE["raw"]["e2e_wall_ms"], 1.1), "itl_ms_p50") == 1


def test_confidence_interval_uses_student_t() -> None:
    low, high = confidence_interval([1.0, 2.0, 3.0])
    half = 4.303 * 1.0 / math.sqrt(3)
    assert (low, high) == pytest.approx((2.0 - half, 2.0 + half))
    with pytest.raises(ValueError):
        confidence_interval([1.0])


def test_mann_whitney_exact_p_values() -> None:
    # Fully separated samples of 5 and 5: 2 of C(10, 5) = 252 orderings.
    p, u = mann_whitney_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert p == pytest.approx(2 / 252) and u == 25
    # One inversion: U = 24, tail holds the orderings with U <= 1.
    p, u = mann_whitney_p([1, 2, 3, 4, 6], [5, 7, 8, 9, 10])
    assert p == pytest.approx(4 / 252) and u == 24
    # Interleaved samples are not significant.
    assert mann_whitney_p([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])[0] > 0.5


def test_rank_test_cannot_reject_with_too_few_runs() -> None:
    # With 3 runs per side the smallest possible p is 2/20 = 0.1.
    a, b = make_record([1.0, 2.0, 3.0]), make_record([10.0, 11.0, 12.0])
    assert rank_test()(a, b, "e2e_wall_ms") == 0
