import torch

from miniinferbench.workload import fingerprint, make_prompts


def test_same_seed_same_tokens() -> None:
    a = make_prompts(2, 16, vocab_size=100, seed=3)
    b = make_prompts(2, 16, vocab_size=100, seed=3)
    assert torch.equal(a, b)
    assert fingerprint(a) == fingerprint(b)


def test_different_seed_different_tokens() -> None:
    a = make_prompts(2, 16, vocab_size=100, seed=3)
    b = make_prompts(2, 16, vocab_size=100, seed=4)
    assert not torch.equal(a, b)
    assert fingerprint(a) != fingerprint(b)


def test_shape_dtype_and_range() -> None:
    prompts = make_prompts(4, 32, vocab_size=50, seed=0)
    assert prompts.shape == (4, 32)
    assert prompts.dtype == torch.int64
    assert int(prompts.min()) >= 0 and int(prompts.max()) < 50


def test_special_tokens_are_excluded() -> None:
    excluded = {0, 1, 2}
    prompts = make_prompts(8, 64, vocab_size=10, seed=0, excluded_ids=excluded)
    assert not any(int(t) in excluded for t in prompts.flatten())


def test_fingerprint_depends_on_shape() -> None:
    flat = make_prompts(1, 32, vocab_size=100, seed=0)
    assert fingerprint(flat) != fingerprint(flat.view(2, 16))
