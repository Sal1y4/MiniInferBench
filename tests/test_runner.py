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


def test_reference_does_not_suppress_eos(model: LlamaForCausalLM) -> None:
    # Regression test: make a token our loop emits mid-sequence the model's
    # EOS. The reference must still produce it (we ignore EOS on both sides),
    # not mask it and fall back to the runner-up.
    input_ids = make_prompts(2, 8, vocab_size=128, seed=3)
    ours = greedy_decode(model, input_ids, output_len=6)
    saved = model.generation_config.eos_token_id
    model.generation_config.eos_token_id = int(ours[0, 3])
    try:
        report = check_correctness(model, input_ids, output_len=6)
    finally:
        model.generation_config.eos_token_id = saved
    assert report.kind == "exact", report.divergences


def test_detects_a_wrong_loop(model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fault injection: a "decoder" that always returns token 0 must be caught.
    import miniinferbench.runner as runner

    monkeypatch.setattr(
        runner,
        "greedy_decode",
        lambda m, ids, n, t=None, **kw: torch.zeros(
            (ids.shape[0], n), dtype=torch.int64, device=ids.device
        ),
    )
    report = runner.check_correctness(
        model, make_prompts(1, 8, vocab_size=128, seed=2), output_len=6
    )
    assert not report.matches
    assert report.kind == "mismatch"
    assert report.first_mismatch is not None


def _reference_with_swap(
    model: LlamaForCausalLM, input_ids: torch.Tensor, pos: int, rank: int
) -> tuple[torch.Tensor, float]:
    """Our own output, with the token at `pos` of sequence 0 replaced by the
    model's `rank`-th choice (rank 1 = runner-up); also returns how far that
    token's logit is below the best one."""
    ours = greedy_decode(model, input_ids, output_len=6)
    prefix = torch.cat([input_ids[0], ours[0, :pos]]).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_ids=prefix).logits[0, -1].float()
    top = logits.topk(rank + 1)
    ref = ours.clone()
    ref[0, pos] = top.indices[rank]
    return ref, float(top.values[0] - top.values[rank])


@pytest.mark.parametrize(
    ("rank", "within_tolerance", "kind"),
    [(1, True, "near_tie"), (1, False, "mismatch"), (5, False, "mismatch")],
)
def test_classifies_divergence_by_gap_to_best_logit(
    model: LlamaForCausalLM,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    within_tolerance: bool,
    kind: str,
) -> None:
    # A flip to a token whose logit is within tolerance of the best is a near
    # tie; a flip to a token further below the best is a mismatch.
    import miniinferbench.runner as runner

    input_ids = make_prompts(2, 8, vocab_size=128, seed=3)
    ref, gap = _reference_with_swap(model, input_ids, pos=3, rank=rank)
    _, runner_up_gap = _reference_with_swap(model, input_ids, pos=3, rank=1)
    tolerance = gap * 1.01 if within_tolerance else runner_up_gap * 0.5
    monkeypatch.setattr(runner, "reference_generate", lambda m, ids, n: ref)
    report = runner.check_correctness(model, input_ids, 6, tie_tolerance=tolerance)
    assert report.kind == kind
    assert report.first_mismatch == (0, 3)
    (d,) = report.divergences
    assert d.ours_gap == pytest.approx(0.0, abs=1e-5)  # our token is the best here
    assert d.reference_gap == pytest.approx(gap, rel=1e-4)


def test_measure_run_sample_counts(model: LlamaForCausalLM) -> None:
    samples = measure_run(model, make_prompts(2, 8, vocab_size=128, seed=0), output_len=6)
    assert len(samples.itl_ms) == 5  # one gap per decode step after the first token
    assert samples.ttft_ms > 0
    assert all(gap > 0 for gap in samples.itl_ms)
    assert samples.peak_memory_bytes is None  # CPU


def test_rejects_zero_output_len(model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError):
        greedy_decode(model, make_prompts(1, 4, vocab_size=128, seed=0), output_len=0)


@pytest.mark.parametrize("regression", ["host_sync_each_step", "no_kv_cache"])
def test_injected_regressions_keep_output_identical(
    model: LlamaForCausalLM, regression: str
) -> None:
    input_ids = make_prompts(2, 10, vocab_size=128, seed=5)
    report = check_correctness(model, input_ids, output_len=8, regression=regression)  # type: ignore[arg-type]
    assert report.matches


@pytest.mark.parametrize("mode", ["static", "compiled", "cuda_graph"])
def test_static_cache_modes_match_generate(model: LlamaForCausalLM, mode: str) -> None:
    # On CPU these run the static-cache path without compilation; the second
    # call reuses (and resets) the same cache, as graph replay requires.
    input_ids = make_prompts(2, 8, vocab_size=128, seed=4)
    for _ in range(2):
        report = check_correctness(model, input_ids, output_len=6, decode_mode=mode)  # type: ignore[arg-type]
        assert report.kind == "exact", report.divergences


def test_cuda_graph_mode_rejects_no_kv_cache(model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError):
        greedy_decode(
            model,
            make_prompts(1, 4, vocab_size=128, seed=0),
            output_len=3,
            regression="no_kv_cache",
            decode_mode="cuda_graph",
        )
