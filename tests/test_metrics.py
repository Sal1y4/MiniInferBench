import pytest

from miniinferbench import metrics


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert metrics.percentile(values, 0) == 1.0
    assert metrics.percentile(values, 50) == 2.5
    assert metrics.percentile(values, 100) == 4.0
    assert metrics.percentile(values, 25) == pytest.approx(1.75)


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        metrics.percentile([], 50)
    with pytest.raises(ValueError):
        metrics.percentile([1.0], 101)


def test_tpot_excludes_first_token() -> None:
    # 10 ms to first token, then 4 more tokens at 5 ms each -> E2E 30 ms.
    assert metrics.tpot_ms(e2e_ms=30.0, ttft_ms=10.0, output_len=5) == 5.0


def test_steps_per_s_is_not_tokens_per_s() -> None:
    itl = [10.0] * 10  # 10 steps of 10 ms -> 100 steps/s
    assert metrics.decode_steps_per_s(itl) == pytest.approx(100.0)
    assert metrics.decode_tokens_per_s(itl, batch_size=8) == pytest.approx(800.0)


def test_throughputs() -> None:
    assert metrics.prefill_tokens_per_s(2, 512, ttft_ms=100.0) == pytest.approx(10240.0)
    assert metrics.output_tokens_per_s(4, 32, e2e_ms=1000.0) == pytest.approx(128.0)


def test_coefficient_of_variation() -> None:
    assert metrics.coefficient_of_variation([5.0]) == 0.0
    assert metrics.coefficient_of_variation([10.0, 10.0, 10.0]) == 0.0
    assert metrics.coefficient_of_variation([9.0, 11.0]) == pytest.approx(2**0.5 / 10)
