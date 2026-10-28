import time

import pytest

from app.services.metrics import MetricsRegistry, TurnMetrics, percentile


def test_percentile_matches_known_values():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 50
    assert percentile(values, 95) == pytest.approx(48.0)
    assert percentile([7], 95) == 7


def test_percentile_rejects_empty_input():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_add_ms_accumulates_from_none():
    m = TurnMetrics(request_id="r1", transport="http")
    m.add_ms("tts_ms", 10.0)
    m.add_ms("tts_ms", 5.5)
    assert m.tts_ms == 15.5


def test_finish_records_total_and_keeps_first_outcome():
    m = TurnMetrics(request_id="r1", transport="http")
    time.sleep(0.01)
    m.finish("timeout", "stt_timeout")
    first_total = m.total_ms
    m.finish("ok")  # a later finish must not overwrite the real outcome
    assert (m.status, m.error_code, m.total_ms) == ("timeout", "stt_timeout", first_total)
    assert first_total >= 9  # ms; slept 10ms


def test_first_audio_is_only_marked_once():
    m = TurnMetrics(request_id="r1", transport="ws")
    m.mark_first_audio()
    first = m.first_audio_ms
    time.sleep(0.005)
    m.mark_first_audio()
    assert m.first_audio_ms == first


def test_to_dict_hides_private_fields_and_rounds():
    m = TurnMetrics(request_id="r1", transport="http", stt_ms=12.34567)
    data = m.to_dict()
    assert "_started" not in data
    assert data["stt_ms"] == 12.35


def test_registry_is_bounded_and_summarises_only_ok_turns():
    registry = MetricsRegistry(keep=3)
    for i, status in enumerate(["ok", "ok", "error", "ok", "ok"]):
        m = TurnMetrics(request_id=f"r{i}", transport="http", stt_ms=float(100 * (i + 1)))
        m.finish(status)
        registry.record(m)

    summary = registry.summary()
    assert summary["window_size"] == 3  # r2, r3, r4 remain
    assert summary["turns_since_start"] == {"ok": 4, "error": 1}  # counts cover everything seen
    # Window holds r2 (error, excluded), r3=400, r4=500 -> only ok turns summarised
    assert summary["stages_ms"]["stt_ms"] == {"n": 2, "median": 450.0, "p95": 495.0}
    assert [t["request_id"] for t in registry.recent()] == ["r2", "r3", "r4"]
