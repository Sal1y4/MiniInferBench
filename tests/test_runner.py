import pytest
import torch
from transformers import LlamaForCausalLM

from miniinferbench.runner import check_correctness, greedy_decode, measure_run
from miniinferbench.workload import make_prompts

from .tiny_model import make_tiny_model


@pytest.fixture(scope="module")
def model() -> LlamaForCausalLM:
    return make_tiny_model()


def test_greedy_decode_shape(model: LlamaForCausalLM) -> None:
    input_ids = make_prompts(3, 8, vocab_size=128, seed=0)
    out = greedy_decode(model, input_ids, output_len=5)
    assert out.shape == (3, 5)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_matches_hf_generate(model: LlamaForCausalLM, batch_size: int) -> None:
    input_ids = make_prompts(batch_size, 12, vocab_size=128, seed=1)
    report = check_correctness(model, input_ids, output_len=10)
    assert report.matches, f"first mismatch at {report.first_mismatch}"


def test_detects_a_wrong_loop(model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fault injection: a "decoder" that always returns token 0 must be caught.
    import miniinferbench.runner as runner

    monkeypatch.setattr(
        runner,
        "greedy_decode",
        lambda m, ids, n, t=None: torch.zeros((ids.shape[0], n), dtype=torch.int64),
    )
    report = runner.check_correctness(
        model, make_prompts(1, 8, vocab_size=128, seed=2), output_len=6
    )
    assert not report.matches
    assert report.first_mismatch is not None


def test_measure_run_sample_counts(model: LlamaForCausalLM) -> None:
    samples = measure_run(model, make_prompts(2, 8, vocab_size=128, seed=0), output_len=6)
    assert len(samples.itl_ms) == 5  # one gap per decode step after the first token
    assert samples.ttft_ms > 0
    assert all(gap > 0 for gap in samples.itl_ms)
    assert samples.peak_memory_bytes is None  # CPU


def test_rejects_zero_output_len(model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError):
        greedy_decode(model, make_prompts(1, 4, vocab_size=128, seed=0), output_len=0)
