import asyncio

import numpy as np
import pytest

from app.errors import (
    ProviderError,
    ServerBusyError,
    ShuttingDownError,
    StageTimeoutError,
)
from app.services.llm import ChatMessage
from app.services.metrics import MetricsRegistry, TurnMetrics
from app.services.voice_pipeline import VoicePipeline
from tests.conftest import make_settings
from tests.fakes import FakeLLM, FakeSTT, FakeTTS

AUDIO = np.zeros(16000, dtype=np.float32)  # one second; the fakes ignore the content


def build(stt=None, llm=None, tts=None, **overrides):
    stt, llm, tts = stt or FakeSTT(), llm or FakeLLM(), tts or FakeTTS()
    registry = MetricsRegistry()
    pipeline = VoicePipeline(stt, llm, tts, make_settings(**overrides), registry)
    return pipeline, registry, (stt, llm, tts)


async def run_turn(pipeline, history=None, audio=AUDIO):
    metrics = TurnMetrics(request_id="t", transport="http")
    async with pipeline.turn(metrics):
        result = await pipeline.run_batch(audio, history or [], metrics)
    return result, metrics


# ------------------------------------------------------------------------ success path

async def test_successful_turn_returns_text_audio_and_complete_metrics():
    pipeline, registry, _ = build()
    result, m = await run_turn(pipeline)

    assert result.transcript == "what is the capital of france"
    assert result.reply == "The capital of France is Paris. It is a lovely city."
    assert result.sample_rate == 22050
    assert len(result.pcm16) > 0 and len(result.pcm16) % 2 == 0

    assert m.status == "ok" and m.error_code is None
    for name in ("stt_ms", "stt_queue_ms", "llm_queue_ms", "llm_ttft_ms", "llm_ms",
                 "tts_queue_ms", "tts_first_chunk_ms", "tts_ms", "first_audio_ms", "total_ms"):
        assert getattr(m, name) is not None, name
    assert m.llm_ttft_ms <= m.llm_ms
    assert m.tts_first_chunk_ms <= m.tts_ms
    assert m.first_audio_ms <= m.total_ms
    assert m.audio_seconds == pytest.approx(1.0)
    assert m.llm_output_tokens == 11  # FakeLLM reports one token per word of its 11-word reply
    assert m.tts_audio_seconds == pytest.approx(0.4, abs=0.01)  # two fake sentences x 0.2 s
    assert registry.summary()["turns_since_start"] == {"ok": 1}


async def test_llm_receives_system_prompt_history_and_user_text_in_order():
    pipeline, _, (_, llm, _) = build(llm=FakeLLM())
    history = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")]
    await run_turn(pipeline, history)
    assert [(m.role, m.content) for m in llm.last_messages] == [
        ("system", pipeline._s.llm_system_prompt),
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "what is the capital of france"),
    ]


async def test_history_is_trimmed_to_the_configured_number_of_turns():
    pipeline, _, (_, llm, _) = build(max_history_turns=1)
    history = [ChatMessage("user", f"q{i}") if i % 2 == 0 else ChatMessage("assistant", f"a{i}") for i in range(6)]
    await run_turn(pipeline, history)
    contents = [m.content for m in llm.last_messages]
    assert contents[1:-1] == ["q4", "a5"]  # only the most recent exchange survives


async def test_markdown_in_the_reply_is_cleaned_before_it_is_spoken():
    pipeline, _, (_, _, tts) = build(llm=FakeLLM(reply="**Paris** is the capital."))
    result, _ = await run_turn(pipeline)
    assert result.reply == "Paris is the capital."
    assert tts.spoken == ["Paris is the capital."]


async def test_silence_returns_no_speech_and_never_calls_llm_or_tts():
    pipeline, registry, (_, llm, tts) = build(stt=FakeSTT(text=""))
    result, m = await run_turn(pipeline)
    assert (result.transcript, result.reply, result.pcm16) == ("", "", b"")
    assert m.status == "no_speech"
    assert llm.probe.total == 0 and tts.probe.total == 0
    assert registry.summary()["turns_since_start"] == {"no_speech": 1}


# ------------------------------------------------------------------------ provider errors

@pytest.mark.parametrize(
    "stage,kwargs",
    [
        ("stt", {"stt": FakeSTT(error=ProviderError("stt", "model crashed"))}),
        ("llm", {"llm": FakeLLM(error=ProviderError("llm", "ollama down"))}),
    ],
)
async def test_provider_failure_surfaces_with_its_stage_code_and_is_recorded(stage, kwargs):
    pipeline, registry, _ = build(**kwargs)
    metrics = TurnMetrics(request_id="t", transport="http")
    with pytest.raises(ProviderError) as info:
        async with pipeline.turn(metrics):
            await pipeline.run_batch(AUDIO, [], metrics)
    assert info.value.code == f"{stage}_error"
    assert (metrics.status, metrics.error_code) == ("error", f"{stage}_error")
    assert registry.summary()["turns_since_start"] == {"error": 1}
    assert pipeline.stats()["inflight_turns"] == 0  # the slot is returned even on failure


async def test_without_a_voice_the_reply_is_returned_as_text_only():
    pipeline, registry, _ = build(tts=FakeTTS(error=ProviderError("tts", "voice broken")))
    metrics = TurnMetrics(request_id="t", transport="http")
    async with pipeline.turn(metrics):
        result = await pipeline.run_batch(AUDIO, [], metrics)
    assert result.reply and result.pcm16 == b""
    assert metrics.tts_fallback == "Text only: voice broken"
    assert registry.summary()["turns_since_start"] == {"ok": 1}


async def test_empty_llm_reply_is_an_error_not_silence():
    pipeline, _, _ = build(llm=FakeLLM(reply="  "))
    with pytest.raises(ProviderError, match="empty reply"):
        await run_turn(pipeline)


async def test_unexpected_exception_is_recorded_as_internal_error_and_reraised():
    pipeline, registry, _ = build(stt=FakeSTT(error=RuntimeError("bug")))
    metrics = TurnMetrics(request_id="t", transport="http")
    with pytest.raises(RuntimeError):
        async with pipeline.turn(metrics):
            await pipeline.run_batch(AUDIO, [], metrics)
    assert (metrics.status, metrics.error_code) == ("error", "internal_error")


# ------------------------------------------------------------------------ timeouts

@pytest.mark.parametrize(
    "kwargs,overrides,code",
    [
        ({"stt": FakeSTT(delay=1)}, {"stt_timeout_s": 0.05}, "stt_timeout"),
        ({"llm": FakeLLM(first_token_delay=1)}, {"llm_first_token_timeout_s": 0.05}, "llm_timeout"),
        ({"llm": FakeLLM(token_delay=0.05)}, {"llm_timeout_s": 0.12}, "llm_timeout"),
        ({"tts": FakeTTS(delay=1)}, {"tts_timeout_s": 0.05}, "tts_timeout"),
        ({"stt": FakeSTT(delay=1)}, {"turn_timeout_s": 0.05}, "turn_timeout"),
    ],
    ids=["stt", "llm-first-token", "llm-total", "tts", "whole-turn"],
)
async def test_every_stage_and_the_whole_turn_have_deadlines(kwargs, overrides, code):
    pipeline, registry, _ = build(**kwargs, **overrides)
    metrics = TurnMetrics(request_id="t", transport="http")
    with pytest.raises(StageTimeoutError) as info:
        async with pipeline.turn(metrics):
            await pipeline.run_batch(AUDIO, [], metrics)
    assert info.value.code == code
    assert (metrics.status, metrics.error_code) == ("timeout", code)
    assert metrics.total_ms < 600, "a timed-out turn must stop promptly, not run to completion"
    assert pipeline.stats()["inflight_turns"] == 0


async def test_llm_timeout_distinguishes_no_first_token_from_slow_generation():
    pipeline, _, _ = build(llm=FakeLLM(first_token_delay=1), llm_first_token_timeout_s=0.05)
    with pytest.raises(StageTimeoutError, match="no output within"):
        await run_turn(pipeline)


# ------------------------------------------------------------------------ cancellation

async def test_cancelling_a_turn_stops_the_work_and_records_cancelled():
    stt = FakeSTT(delay=5)
    pipeline, registry, _ = build(stt=stt)
    task = asyncio.create_task(run_turn(pipeline))
    await asyncio.sleep(0.05)
    assert pipeline.stats()["inflight_turns"] == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stt.cancel_seen
    assert pipeline.stats()["inflight_turns"] == 0
    assert registry.summary()["turns_since_start"] == {"cancelled": 1}
    assert registry.recent()[0]["error_code"] == "cancelled"


async def test_cancelling_during_llm_generation_closes_the_llm_stream():
    llm = FakeLLM(token_delay=0.05, reply=" ".join(["word"] * 100))
    pipeline, _, _ = build(llm=llm)
    task = asyncio.create_task(run_turn(pipeline))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.cancelled
    assert llm.probe.current == 0


# ------------------------------------------------------------------------ admission and concurrency

async def test_turns_beyond_the_inflight_limit_are_rejected_immediately():
    pipeline, registry, _ = build(stt=FakeSTT(delay=0.3), max_inflight_turns=2)
    running = [asyncio.create_task(run_turn(pipeline)) for _ in range(2)]
    await asyncio.sleep(0.05)

    metrics = TurnMetrics(request_id="late", transport="http")
    with pytest.raises(ServerBusyError):
        async with pipeline.turn(metrics):
            pytest.fail("a rejected turn must not run")
    assert metrics.status == "rejected" and metrics.total_ms < 50

    await asyncio.gather(*running)  # the admitted turns are unaffected
    assert registry.summary()["turns_since_start"] == {"ok": 2, "rejected": 1}
    assert pipeline.stats()["inflight_turns"] == 0


@pytest.mark.parametrize("limit", [1, 2])
async def test_stage_concurrency_never_exceeds_its_configured_limit(limit):
    stt, tts = FakeSTT(delay=0.05), FakeTTS(delay=0.02)
    pipeline, _, _ = build(stt=stt, tts=tts, stt_concurrency=limit, tts_concurrency=limit, max_inflight_turns=8)
    await asyncio.gather(*(run_turn(pipeline) for _ in range(6)))
    assert stt.probe.maximum == limit
    assert tts.probe.maximum == limit
    assert stt.probe.total == 6


async def test_waiting_for_a_slot_is_measured_separately_from_work():
    pipeline, _, _ = build(stt=FakeSTT(delay=0.1), stt_concurrency=1, max_inflight_turns=4)
    results = await asyncio.gather(*(run_turn(pipeline) for _ in range(3)))
    waits = sorted(m.stt_queue_ms for _, m in results)
    assert waits[0] < 20  # first one runs at once
    assert waits[2] > 150  # last one waited for two full STT runs (~200 ms)
    assert all(m.stt_ms == pytest.approx(100, abs=60) for _, m in results)  # work time is unaffected by queuing


# ------------------------------------------------------------------------ shutdown

async def test_shutdown_rejects_new_turns_and_cancels_stragglers():
    stt = FakeSTT(delay=10)
    pipeline, registry, _ = build(stt=stt)
    straggler = asyncio.create_task(run_turn(pipeline))
    await asyncio.sleep(0.05)

    await pipeline.shutdown(grace_s=0.1)  # grace expires -> the running turn is cancelled
    with pytest.raises(asyncio.CancelledError):
        await straggler
    assert stt.cancel_seen

    metrics = TurnMetrics(request_id="after", transport="http")
    with pytest.raises(ShuttingDownError):
        async with pipeline.turn(metrics):
            pytest.fail("must not run")
    assert metrics.status == "rejected"


async def test_shutdown_waits_for_a_turn_that_finishes_within_the_grace_period():
    pipeline, _, _ = build(stt=FakeSTT(delay=0.1))
    turn = asyncio.create_task(run_turn(pipeline))
    await asyncio.sleep(0.02)
    await pipeline.shutdown(grace_s=2.0)
    result, metrics = await turn
    assert metrics.status == "ok" and result.reply
