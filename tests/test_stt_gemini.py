"""Gemini transcription and the recogniser fallback (no network)."""

import base64
import json
import threading

import httpx
import numpy as np
import pytest

from app.errors import ProviderError, TurnCancelledError
from app.services.metrics import MetricsRegistry, TurnMetrics
from app.services.stt import FallbackSTT, FasterWhisperSTT, GeminiSTT, Transcript, normalize_language
from app.services.voice_pipeline import VoicePipeline
from tests.conftest import make_settings
from tests.fakes import FakeLLM, FakeSTT, FakeTTS
from tests.test_pipeline_streaming import AUDIO, RecordingSink

SECOND = np.zeros(16000, dtype=np.float32)


def reply(language: str, text: str) -> httpx.Response:
    body = json.dumps({"language": language, "text": text})
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": body}]}}]})


def gemini(handler, **overrides) -> GeminiSTT:
    settings = make_settings(stt_provider="gemini", gemini_api_key="test-key-123", **overrides)
    return GeminiSTT(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_sends_the_audio_as_wav_with_an_instruction_and_reads_the_json_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = request.url, request.headers, json.loads(request.content)
        return reply("hi", "  भारत की राजधानी क्या है?  ")

    transcript = await gemini(handler, stt_language="auto").transcribe(SECOND, threading.Event())

    assert transcript == Transcript(text="भारत की राजधानी क्या है?", language="hi", audio_seconds=1.0, engine="gemini gemini-3.5-flash-lite")
    assert seen["url"].path.endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "test-key-123"
    assert "test-key-123" not in str(seen["url"])
    audio_part, text_part = seen["body"]["contents"][0]["parts"]
    wav = base64.b64decode(audio_part["inlineData"]["data"])
    assert audio_part["inlineData"]["mimeType"] == "audio/wav" and wav[:4] == b"RIFF"
    assert int.from_bytes(wav[24:28], "little") == 16000
    assert "Never translate" in text_part["text"]
    assert "language with code" not in text_part["text"], "auto-detect gives no language hint"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


async def test_a_configured_language_is_passed_as_a_hint():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply("", "hola")

    transcript = await gemini(handler, stt_language="es").transcribe(SECOND, threading.Event())
    assert "Spanish (es)" in seen["contents"][0]["parts"][1]["text"]
    assert transcript.language == "es", "falls back to the configured language when the model gives none"


async def test_expected_languages_are_named_in_the_instruction():
    """Measured on 2026-10-27: without this hint a Telugu question came back as a Hindi translation."""
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply("Telugu", "భారతదేశ రాజధాని ఏమిటి?")

    transcript = await gemini(handler, stt_language="auto", stt_languages="te, en").transcribe(SECOND, threading.Event())
    assert "Telugu (te) or English (en)" in seen["contents"][0]["parts"][1]["text"]
    assert transcript.language == "te", "a language name in the reply is turned into its code"


@pytest.mark.parametrize("raw,code", [("te", "te"), ("Telugu", "te"), ("EN-us", "en"), ("en_GB", "en"), (" hi ", "hi")])
def test_language_values_are_normalised_to_codes(raw, code):
    assert normalize_language(raw) == code


class StubWhisper:
    """Stands in for faster-whisper's WhisperModel: Hindi is likelier, but only te/en are allowed."""

    class model:
        is_multilingual = True

    def __init__(self):
        self.used_language = "unset"

    def detect_language(self, audio):
        return "hi", 0.6, [("hi", 0.6), ("te", 0.3), ("en", 0.1)]

    def transcribe(self, audio, language=None, **kwargs):
        self.used_language = language
        info = type("Info", (), {"language": language or "hi"})()
        return iter([type("Segment", (), {"text": " text "})()]), info


async def test_whisper_picks_the_likeliest_of_the_expected_languages():
    stt = FasterWhisperSTT(make_settings(stt_model="base", stt_language="auto", stt_languages="te,en"))
    stt._model = StubWhisper()
    transcript = await stt.transcribe(SECOND, threading.Event())
    assert stt._model.used_language == "te"
    assert transcript.language == "te"
    await stt.aclose()


async def test_rate_limit_is_reported_as_such():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded. Please retry in 9s."}})

    with pytest.raises(ProviderError, match="rate limit"):
        await gemini(handler).transcribe(SECOND, threading.Event())


async def test_an_unexpected_reply_is_a_provider_error_not_a_crash():
    def handler(request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})

    with pytest.raises(ProviderError, match="unexpected reply"):
        await gemini(handler).transcribe(SECOND, threading.Event())


async def test_without_a_key_loading_fails_clearly():
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        await GeminiSTT(make_settings(stt_provider="gemini")).load()


# ----------------------------------------------------------------------------- fallback


async def fallback_with(primary_error: Exception):
    stt = FallbackSTT(FakeSTT(error=primary_error), FakeSTT(text="local transcript"))
    await stt.load()
    return stt


async def test_the_fallback_transcribes_when_the_primary_refuses():
    stt = await fallback_with(ProviderError("stt", "Gemini transcription returned 429 (free-tier rate limit)"))
    transcript = await stt.transcribe(SECOND, threading.Event())
    assert transcript.text == "local transcript"
    assert "rate limit" in transcript.fallback_reason


async def test_cancellation_is_not_turned_into_a_fallback():
    stt = await fallback_with(TurnCancelledError("cancelled"))
    with pytest.raises(TurnCancelledError):
        await stt.transcribe(SECOND, threading.Event())


# ----------------------------------------------------------------------------- pipeline


class NoPartials(FakeSTT):
    supports_partials = False


async def test_a_cloud_recogniser_is_never_asked_for_partial_transcripts():
    stt = NoPartials()
    pipeline = VoicePipeline(stt, FakeLLM(), FakeTTS(), make_settings(), MetricsRegistry())
    assert await pipeline.transcribe_partial(SECOND, threading.Event()) is None
    assert stt.probe.total == 0, "each partial would cost a request against the quota"


async def test_the_turn_records_a_recogniser_fallback():
    stt = await fallback_with(ProviderError("stt", "rate limit"))
    pipeline = VoicePipeline(stt, FakeLLM(), FakeTTS(), make_settings(), MetricsRegistry())
    metrics = TurnMetrics(request_id="t", transport="ws")
    async with pipeline.turn(metrics):
        await pipeline.run_streaming(AUDIO, [], metrics, RecordingSink())
    assert metrics.stt_fallback == "rate limit"
