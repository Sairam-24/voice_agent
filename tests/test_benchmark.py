import asyncio
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark  # noqa: E402
from voice_client import TurnTrace  # noqa: E402

from app.startup import Services  # noqa: E402
from tests.conftest import sine_wav  # noqa: E402
from tests.fakes import FakeLLM, FakeSTT, FakeTTS  # noqa: E402


def ok_trace(first_audio, done, stt_ms=100.0) -> TurnTrace:
    return TurnTrace(
        transport="ws", ok=True, t_first_audio=first_audio, t_done=done,
        server_metrics={"stt_ms": stt_ms, "total_ms": done * 1000},
    )


def failed_trace(code="server_busy") -> TurnTrace:
    return TurnTrace(transport="ws", ok=False, error_code=code, t_done=0.01)


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        ("What is the capital of France?", "what is the capital of france", 0.0),  # case/punctuation ignored
        ("one two three four", "one two three", 0.25),  # one deletion
        ("one two three four", "one 2 three four", 0.25),  # one substitution
        ("one two", "one two three four", 1.0),  # two insertions / two reference words
        ("a three-day trip", "a three day trip", 0.0),  # hyphen == space
        ("hello", "", 1.0),
    ],
)
def test_word_error_rate(reference, hypothesis, expected):
    assert benchmark.word_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_word_error_rate_rejects_an_empty_reference():
    with pytest.raises(ValueError):
        benchmark.word_error_rate("...", "anything")


def test_describe_uses_linear_interpolated_percentiles():
    stats = benchmark.describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (stats["n"], stats["median"], stats["min"], stats["max"], stats["mean"]) == (5, 3.0, 1.0, 5.0, 3.0)
    assert stats["p95"] == pytest.approx(4.8)
    assert benchmark.describe([]) is None


def test_summarize_run_counts_failures_and_excludes_them_from_latency():
    traces = [ok_trace(1.0, 2.0), ok_trace(3.0, 4.0), failed_trace(), failed_trace("llm_timeout")]
    run = benchmark.summarize_run(traces, wall_seconds=8.0)
    assert (run["requests"], run["ok"], run["failed"]) == (4, 2, 2)
    assert run["error_rate"] == 0.5
    assert run["errors"] == {"server_busy": 1, "llm_timeout": 1}
    assert run["throughput_rps"] == 0.25  # 2 successful requests in 8 s; failures do not count as throughput
    assert run["first_audio_s"]["median"] == 2.0 and run["turn_done_s"]["median"] == 3.0
    assert run["stages_ms"]["stt_ms"]["n"] == 2


def test_summarize_run_with_only_failures_does_not_crash():
    run = benchmark.summarize_run([failed_trace()], wall_seconds=1.0)
    assert run["ok"] == 0 and run["first_audio_s"] is None and run["stages_ms"] == {}


def test_aggregate_runs_reports_spread_across_runs():
    runs = [
        benchmark.summarize_run([ok_trace(1.0, 2.0)] * 3, 3.0),
        benchmark.summarize_run([ok_trace(2.0, 3.0)] * 3, 3.0),
        benchmark.summarize_run([ok_trace(3.0, 4.0), failed_trace()], 2.0),
    ]
    agg = benchmark.aggregate_runs(runs)
    assert (agg["runs"], agg["requests"], agg["ok"], agg["failed"]) == (3, 8, 7, 1)
    assert agg["error_rate"] == pytest.approx(1 / 8)
    span = agg["first_audio_s_median_of_runs"]
    assert (span["min"], span["median"], span["max"]) == (1.0, 2.0, 3.0)
    assert agg["errors"] == {"server_busy": 1}


def test_csv_rows_contain_every_request_and_every_stage_column():
    run = {"transport": "ws", "concurrency": 2, "run": 1,
           "traces": [benchmark.trace_to_dict(ok_trace(1.0, 2.0), "short.wav"),
                      benchmark.trace_to_dict(failed_trace(), "long.wav")]}
    rows = benchmark.csv_rows("lbl", [run])
    assert len(rows) == 2
    assert set(benchmark.STAGE_KEYS) <= set(rows[0])
    assert rows[1]["error_code"] == "server_busy" and rows[1]["input"] == "long.wav"


def test_arguments_cold_implies_spawning_and_levels_are_parsed():
    args = benchmark.parse_args(["--cold", "--concurrency", "1,2,4", "--env", "STT_MODEL=tiny.en"])
    assert args.spawn_server and args.cold
    assert args.concurrency == [1, 2, 4]
    assert args.env_overrides == {"STT_MODEL": "tiny.en"}


@pytest.mark.parametrize("bad", [["--requests", "0"], ["--runs", "0"], ["--concurrency", "0"]])
def test_invalid_arguments_are_rejected(bad):
    with pytest.raises(SystemExit):
        benchmark.parse_args(bad)


def test_machine_info_never_raises_and_has_the_documented_fields():
    info = benchmark.machine_info()
    assert {"os", "python", "cpu", "logical_cpus", "ram_gb", "gpu"} <= set(info)


def test_full_benchmark_run_against_a_server_writes_json_and_csv(live_server, tmp_path):
    """Exercises the real script end to end. The server uses FAKE models, and the saved
    result says so in its `server` section, so it can never pass for model performance."""
    base = live_server(Services(FakeSTT(delay=0.05), FakeLLM(), FakeTTS(delay=0.01)), max_inflight_turns=2)
    inputs = []
    for name in ("short", "long"):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(sine_wav(0.5 if name == "short" else 1.5))
        inputs.append(path)

    args = benchmark.parse_args([
        "--url", base, "--transport", "both", "--concurrency", "1,4", "--requests", "4", "--runs", "3",
        "--warmup", "1", "--label", "unit", "--output-dir", str(tmp_path / "out"), "--inputs", *map(str, inputs),
    ])
    assert asyncio.run(benchmark.main(args)) == 0

    (json_path,) = (tmp_path / "out").glob("*-unit.json")
    result = json.loads(json_path.read_text())
    assert result["server"]["components"]["stt"] == {"engine": "fake-stt"}  # visibly not a real model
    assert set(result["summary"]) == {"ws-c1", "ws-c4", "http-c1", "http-c4"}
    assert len(result["runs"]) == 3 * 4
    assert set(result["inputs"]) == {"short.wav", "long.wav"} and all(len(v["sha256"]) == 64 for v in result["inputs"].values())
    assert result["machine"]["python"]

    for key in ("ws-c1", "http-c1"):
        assert result["summary"][key]["failed"] == 0
        assert result["summary"][key]["requests"] == 12
    # 4 concurrent requests against max_inflight_turns=2: the surplus is rejected, and counted, not hidden
    for key in ("ws-c4", "http-c4"):
        assert result["summary"][key]["errors"].get("server_busy", 0) > 0
        assert result["summary"][key]["error_rate"] > 0

    with json_path.with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 4 * 4  # runs x configs x requests
    assert {"transport", "concurrency", "t_first_audio", "stt_ms"} <= set(rows[0])
