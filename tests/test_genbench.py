import json
from pathlib import Path
from typing import Any

import pytest
import torch

import miniinferbench.genbench as genbench
import miniinferbench.trace as trace
from miniinferbench.cli import main
from miniinferbench.config import BenchmarkConfig
from miniinferbench.genbench import (
    compare,
    format_comparison,
    pooled_evidence,
    run_generate,
    run_generation_task,
)
from miniinferbench.results import append_record
from miniinferbench.runner import greedy_decode, reference_generate
from miniinferbench.timing import EventTimeline
from miniinferbench.workload import make_prompts

from .test_diagnosis import BASE, STEPS, evidence
from .tiny_model import make_tiny_model

CONFIG = BenchmarkConfig(
    model_name="tiny", batch_size=2, input_len=8, output_len=6, warmup_runs=0, measure_runs=2
)


def test_marked_generate_matches_the_plain_reference_and_the_loop() -> None:
    model = make_tiny_model()
    input_ids = make_prompts(2, 8, vocab_size=128, seed=3)
    timeline = EventTimeline(input_ids.device)
    tokens = run_generate(model, input_ids, 6, timeline)
    assert torch.equal(tokens, reference_generate(model, input_ids, 6))
    assert torch.equal(tokens, greedy_decode(model, input_ids, 6))
    # start, one mark per generated token, end -> 7 gaps for 6 tokens.
    assert len(timeline.finish()) == 7


def test_generate_restores_the_models_generation_config() -> None:
    model = make_tiny_model()
    before = model.generation_config
    run_generate(model, make_prompts(1, 4, vocab_size=128, seed=0), 3)
    assert model.generation_config is before


def stub_analyze(path: Path) -> dict[str, Any]:
    names = [e.get("name") for e in json.loads(path.read_text())["traceEvents"]]
    report: dict[str, Any] = BASE["trace"] | {"steps": [], "seen_steps": names.count("decode_step")}
    return report


@pytest.mark.parametrize("path", ["generate", "loop"])
def test_task_times_then_traces_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    # CPU traces have no GPU events, so the analysis is stubbed; it still
    # counts the step ranges the real trace contains.
    monkeypatch.setattr(trace, "analyze", stub_analyze)
    record = run_generation_task(
        CONFIG,
        path,  # type: ignore[arg-type]
        tmp_path / "t.trace.json",
        require_cuda=False,
        model_and_ids=(make_tiny_model(), set()),
    )
    assert record["status"] == "ok", record.get("traceback")
    timing = record["evidence"]["timing"]
    assert len(timing["e2e_wall_ms"]) == 2
    assert [len(gaps) for gaps in timing["itl_ms"]] == [5, 5]  # 6 tokens -> 5 gaps
    assert "steps" not in record["evidence"]["trace"]
    assert record["evidence"]["trace"]["seen_steps"] >= 5
    assert not (tmp_path / "t.trace.json").exists()


def test_both_paths_produce_the_same_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "analyze", stub_analyze)
    model = (make_tiny_model(), set[int]())
    records = [
        run_generation_task(CONFIG, p, tmp_path / "t.json", require_cuda=False, model_and_ids=model)
        for p in genbench.PATHS
    ]
    assert records[0]["output_fingerprint"] == records[1]["output_fingerprint"]
    assert records[0]["input_fingerprint"] == records[1]["input_fingerprint"]


def test_failure_becomes_a_classified_record(tmp_path: Path) -> None:
    record = run_generation_task(CONFIG, "generate", tmp_path / "t.json", require_cuda=True)
    if not torch.cuda.is_available():
        assert record["status"] == "env_error"
        assert "evidence" not in record


def gen_record(label: str, ev: dict[str, Any], output: str = "same") -> dict[str, Any]:
    return {
        "task": genbench.TASK,
        "label": label,
        "path": "generate",
        "status": "ok",
        "input_fingerprint": "in",
        "output_fingerprint": output,
        "evidence": ev,
        "manifest": {"env": {"transformers": "1.0"}},
    }


def test_pooling_adds_runs_and_keeps_one_trace() -> None:
    pooled = pooled_evidence([gen_record("a", BASE), gen_record("a", evidence(10.1))])
    assert len(pooled["timing"]["e2e_wall_ms"]) == 10
    assert pooled["trace"] == BASE["trace"]


def test_compare_reports_slower_with_a_cause() -> None:
    slow = evidence(12.0, host_waits=3 * STEPS)
    result = compare([gen_record("old", BASE)], [gen_record("new", slow)])
    assert result["verdict"] == "slower"
    assert result["cause"] == "host_sync"
    assert result["host_waits_per_step"] == [0.0, 3.0]
    assert result["change_per_token"] == pytest.approx(0.2, abs=0.03)
    text = format_comparison(result, "old", "new")
    assert "**new** vs **old**: slower, most likely cause `host_sync`" in text
    assert "Output tokens identical: yes" in text


def test_compare_reports_faster_and_no_change() -> None:
    assert compare([gen_record("a", BASE)], [gen_record("b", evidence(8.0))])["verdict"] == "faster"
    same = compare([gen_record("a", BASE)], [gen_record("b", evidence(10.02))])
    assert same["verdict"] == "no change" and same["cause"] is None


def test_compare_flags_different_output_and_refuses_different_input() -> None:
    other = gen_record("b", evidence(10.02), output="different")
    assert compare([gen_record("a", BASE)], [other])["same_output"] is False
    with pytest.raises(ValueError, match="same input"):
        compare([gen_record("a", BASE)], [gen_record("b", BASE) | {"input_fingerprint": "x"}])


def test_check_command_compares_neighbours_and_ends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "gen.jsonl"
    for label, ev in (("v1", BASE), ("v2", evidence(10.02)), ("v3", evidence(12.0, cpu=11000.0))):
        append_record(results, gen_record(label, ev))
    out = tmp_path / "check.json"
    assert main(["check", "v1", "v2", "v3", "--results", str(results), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "**v2** vs **v1**: no change" in printed
    assert "**v3** vs **v2**: slower, most likely cause `cpu_stall`" in printed
    assert "**v3** vs **v1**: slower" in printed
    assert [r["candidate"] for r in json.loads(out.read_text())] == ["v2", "v3", "v3"]
    assert main(["check", "v1", "nope", "--results", str(results)]) == 1


def test_gen_command_appends_a_labelled_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(trace, "analyze", stub_analyze)
    monkeypatch.setattr(genbench, "load_model", lambda config, device: (make_tiny_model(), set()))
    config_path = tmp_path / "c.json"
    CONFIG.save(config_path)
    results = tmp_path / "gen.jsonl"
    args = ["gen", "--config", str(config_path), "--results", str(results), "--allow-cpu"]
    args += ["--traces", str(tmp_path / "traces")]
    assert main([*args, "--label", "mine"]) == 0
    assert main([*args, "--path", "loop"]) == 0
    first, second = (json.loads(line) for line in results.read_text().splitlines())
    assert first["label"] == "mine" and first["path"] == "generate"
    assert second["label"].startswith("loop@")
    assert "[ok] mine" in capsys.readouterr().out
