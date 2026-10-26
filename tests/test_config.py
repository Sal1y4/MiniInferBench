from pathlib import Path

import pytest

from miniinferbench.config import BenchmarkConfig, ConfigError


def test_defaults_are_valid() -> None:
    config = BenchmarkConfig()
    assert config.batch_size == 1
    assert config.dtype == "bfloat16"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config = BenchmarkConfig(batch_size=4, input_len=512, output_len=128, seed=7)
    path = tmp_path / "config.json"
    config.save(path)
    assert BenchmarkConfig.load(path) == config


@pytest.mark.parametrize("field", ["batch_size", "input_len", "output_len", "measure_runs"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_non_positive_integers(field: str, value: object) -> None:
    with pytest.raises(ConfigError, match=field):
        BenchmarkConfig(**{field: value})  # type: ignore[arg-type]


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigError, match="unknown config fields: typo_field"):
        BenchmarkConfig.from_dict({"typo_field": 1})


def test_rejects_unsupported_dtype() -> None:
    with pytest.raises(ConfigError, match="dtype"):
        BenchmarkConfig(dtype="int8")  # type: ignore[arg-type]


def test_load_reports_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read config"):
        BenchmarkConfig.load(path)


def test_config_is_immutable() -> None:
    config = BenchmarkConfig()
    with pytest.raises(AttributeError):
        config.batch_size = 2  # type: ignore[misc]
