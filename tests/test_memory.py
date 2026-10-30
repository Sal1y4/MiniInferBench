from miniinferbench.memory import ModelShape, estimate


def _shape(**overrides: int) -> ModelShape:
    base = {
        "num_layers": 32,
        "num_kv_heads": 5,
        "head_dim": 64,
        "hidden_size": 960,
        "intermediate_size": 2560,
        "vocab_size": 49152,
        "weight_bytes": 724_000_000,
        "dtype_bytes": 2,
    }
    return ModelShape(**(base | overrides))


def test_kv_bytes_per_token_formula() -> None:
    # 2 (K,V) x 32 layers x 5 heads x 64 dims x 2 bytes = 40960 bytes = 40 KiB
    assert _shape().kv_bytes_per_token == 40_960


def test_gqa_vs_mha_kv_cache() -> None:
    gqa = _shape(num_kv_heads=5)
    mha = _shape(num_kv_heads=15)
    assert mha.kv_bytes_per_token == 3 * gqa.kv_bytes_per_token


def test_estimate_grows_with_batch_and_length() -> None:
    small = estimate(_shape(), batch_size=1, input_len=128, output_len=32)
    big = estimate(_shape(), batch_size=8, input_len=1024, output_len=256)
    assert big.total > small.total
    assert big.weights == small.weights
    assert big.kv_cache == 8 * (1024 + 256) * 40_960
