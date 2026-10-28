import asyncio

import numpy as np
import pytest

from app.errors import ProviderError, StageTimeoutError, TurnCancelledError
from app.services.metrics import MetricsRegistry, TurnMetrics, now
from app.services.voice_pipeline import SENTENCE_QUEUE_SIZE, VoicePipeline
from tests.conftest import make_settings
from tests.fakes import FakeLLM, FakeSTT, FakeTTS

AUDIO = np.zeros(16000, dtype=np.float32)


class RecordingSink:
    """Collects the pipeline's events with timestamps so ordering and timing can be asserted."""

    def __init__(self, fail_on_audio_after: int | None = None):
        self.events: list[tuple[str, object, float]] = []
        self.fail_on_audio_after = fail_on_audio_after
        self.segments: list[tuple[int, str]] = []
        self._start = now()

    def _add(self, kind, value=None):
        self.events.append((kind, value, now() - self._start))

    async def status(self, state):
        self._add("status", state)

    async def transcript(self, text, final, language=None):
        self._add("transcript", (text, final))

    async def reply_delta(self, text):
        self._add("delta", text)

    async def audio(self, chunk, index, sentence, segment=0):
        if self.fail_on_audio_after is not None and index >= self.fail_on_audio_after:
            raise TurnCancelledError("client gone")
        self.segments.append((segment, sentence))
        self._add("audio", (index, sentence, len(chunk.pcm16)))

    def kinds(self):
        return [kind for kind, _, _ in self.events]

    def first_time(self, kind):
        return next(t for k, _, t in self.events if k == kind)

    def values(self, kind):
        return [v for k, v, _ in self.events if k == kind]


def build(stt=None, llm=None, tts=None, **overrides):
    stt, llm, tts = stt or FakeSTT(), llm or FakeLLM(), tts or FakeTTS()
    pipeline = VoicePipeline(stt, llm, tts, make_settings(**overrides), MetricsRegistry())
    return pipeline, (stt, llm, tts)


async def run_streaming(pipeline, sink, history=None):
    metrics = TurnMetrics(request_id="s", transport="ws")
    async with pipeline.turn(metrics):
        result = await pipeline.run_streaming(AUDIO, history or [], metrics, sink)
    return result, metrics


async def test_events_arrive_in_pipeline_order():
    pipeline, _ = build()
    sink = RecordingSink()
    result, metrics = await run_streaming(pipeline, sink)

    assert sink.values("status") == ["transcribing", "thinking", "speaking"]
    assert sink.values("transcript") == [("what is the capital of france", True)]
    assert "".join(sink.values("delta")) == "The capital of France is Paris. It is a lovely city."
    audio = sink.values("audio")
    assert [a[0] for a in audio] == [0, 1]  # one chunk per sentence, in order
    assert [a[1] for a in audio] == ["The capital of France is Paris.", "It is a lovely city."]
    # transcript before any LLM text; LLM text before any audio
    kinds = sink.kinds()
    assert kinds.index("transcript") < kinds.index("delta") < kinds.index("audio")
    assert result.reply == "The capital of France is Paris. It is a lovely city."
    assert result.pcm16 == b"" and metrics.status == "ok"
    assert metrics.tts_audio_seconds == pytest.approx(0.4, abs=0.01)


async def test_audio_is_produced_while_the_llm_is_still_generating():
    """The point of streaming: first audio must not wait for the end of the LLM reply."""
    reply = "First sentence is here. " + "More words keep coming along. " * 5
    llm = FakeLLM(reply=reply, token_delay=0.04)
    pipeline, _ = build(llm=llm)
    sink = RecordingSink()
    _, metrics = await run_streaming(pipeline, sink)

    last_delta = max(t for k, _, t in sink.events if k == "delta")
    assert sink.first_time("audio") < last_delta - 0.3
    assert metrics.first_audio_ms < metrics.llm_ms - 300
    # whereas the sequential path cannot beat the full LLM time
    batch_metrics = TurnMetrics(request_id="b", transport="http")
    async with pipeline.turn(batch_metrics):
        await pipeline.run_batch(AUDIO, [], batch_metrics)
    assert batch_metrics.first_audio_ms > batch_metrics.llm_ms


async def test_silence_sends_an_empty_final_transcript_and_nothing_else():
    pipeline, (_, llm, tts) = build(stt=FakeSTT(text=""))
    sink = RecordingSink()
    result, metrics = await run_streaming(pipeline, sink)
    assert sink.values("transcript") == [("", True)]
    assert "audio" not in sink.kinds() and "delta" not in sink.kinds()
    assert metrics.status == "no_speech" and llm.probe.total == 0


async def test_llm_error_midway_surfaces_the_original_error_not_a_taskgroup_wrapper():
    pipeline, _ = build(llm=FakeLLM(error=ProviderError("llm", "ollama down")))
    with pytest.raises(ProviderError) as info:
        await run_streaming(pipeline, RecordingSink())
    assert info.value.code == "llm_error"


class BreaksAfterFirstSentence(FakeTTS):
    async def synthesize(self, text, cancel):
        if self.spoken:  # the second sentence
            raise ProviderError("tts", "voice broken")
        async for chunk in super().synthesize(text, cancel):
            yield chunk


async def test_a_voice_that_breaks_mid_reply_is_an_error_and_stops_the_llm():
    llm = FakeLLM(reply=" ".join(["Sentence number one."] * 40), token_delay=0.01)
    pipeline, _ = build(llm=llm, tts=BreaksAfterFirstSentence())
    with pytest.raises(ProviderError) as info:
        await run_streaming(pipeline, RecordingSink())
    assert info.value.code == "tts_error"
    assert llm.cancelled or llm.yielded < 40 * 3, "the LLM kept generating after TTS failed"
    assert llm.probe.current == 0


async def test_a_voice_that_fails_before_speaking_leaves_a_text_only_reply():
    """No audio at all (e.g. the free voice quota is used up): the reply is still worth showing."""
    pipeline, _ = build(tts=FakeTTS(error=ProviderError("tts", "quota used up")))
    sink = RecordingSink()
    result, metrics = await run_streaming(pipeline, sink)
    assert result.reply == "The capital of France is Paris. It is a lovely city."
    assert sink.values("audio") == []
    assert metrics.tts_fallback == "Text only: quota used up"
    assert metrics.status == "ok"


async def test_slow_llm_hits_its_deadline_during_streaming():
    pipeline, _ = build(llm=FakeLLM(token_delay=0.1), llm_timeout_s=0.25)
    with pytest.raises(StageTimeoutError) as info:
        await run_streaming(pipeline, RecordingSink())
    assert info.value.code == "llm_timeout"


async def test_sink_failure_cancels_the_whole_turn():
    """If sending to the client fails (disconnect), STT/LLM/TTS work must stop, not run on."""
    llm = FakeLLM(reply=" ".join(["A sentence that is long enough."] * 20), token_delay=0.01)
    pipeline, _ = build(llm=llm)
    metrics = TurnMetrics(request_id="s", transport="ws")
    with pytest.raises(TurnCancelledError):
        async with pipeline.turn(metrics):
            await pipeline.run_streaming(AUDIO, [], metrics, RecordingSink(fail_on_audio_after=1))
    assert metrics.status == "cancelled"
    assert llm.yielded < 20 * 6, "the LLM kept generating after the client was gone"
    assert pipeline.stats()["inflight_turns"] == 0


async def test_llm_cannot_run_unboundedly_ahead_of_slow_tts():
    """The sentence queue is bounded, so a slow voice applies backpressure to the LLM."""
    sentences = 30
    llm = FakeLLM(reply=" ".join(f"Sentence number {i}." for i in range(sentences)))
    tts = FakeTTS(delay=0.05)
    pipeline, _ = build(llm=llm, tts=tts)
    sink = RecordingSink()

    task = asyncio.create_task(run_streaming(pipeline, sink))
    await asyncio.sleep(0.4)  # TTS has spoken only a handful of sentences by now
    spoken = len(sink.values("audio"))
    words_per_sentence = 3
    sentences_generated = llm.yielded / words_per_sentence
    assert spoken < sentences - 5, "test premise: TTS should still be busy"
    # generated = spoken + queue capacity + one being synthesized + one blocked in put() (+1 slack)
    assert sentences_generated <= spoken + SENTENCE_QUEUE_SIZE + 3
    await task


@pytest.mark.parametrize(
    "first_clause_chars,expected_first_spoken",
    [
        (0, "Well this is a rather long opening clause, and then it keeps going for a while."),
        (20, "Well this is a rather long opening clause,"),
    ],
)
async def test_first_clause_setting_changes_what_the_voice_speaks_first(first_clause_chars, expected_first_spoken):
    reply = "Well this is a rather long opening clause, and then it keeps going for a while. Done."
    tts = FakeTTS()
    pipeline, _ = build(llm=FakeLLM(reply=reply), tts=tts, tts_first_clause_chars=first_clause_chars)
    await run_streaming(pipeline, RecordingSink())
    assert tts.spoken[0] == expected_first_spoken
    assert " ".join(tts.spoken) == reply  # nothing lost or duplicated by the early cut


async def test_partial_transcription_skips_when_stt_is_busy_and_never_queues():
    stt = FakeSTT(delay=0.3)
    pipeline, _ = build(stt=stt)
    import threading

    first = asyncio.create_task(pipeline.transcribe_partial(AUDIO, threading.Event()))
    await asyncio.sleep(0.05)
    assert await pipeline.transcribe_partial(AUDIO, threading.Event()) is None  # slot taken -> skipped at once
    assert await first == "what is the capital of france"
    assert stt.probe.maximum == 1
