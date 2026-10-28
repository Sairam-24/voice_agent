"""Gemini voice, the fallback wrapper, and how the pipeline uses a whole-reply voice (no network)."""

import base64
import json
import struct
import threading

import httpx
import pytest

from app.errors import ProviderError
from app.services.metrics import MetricsRegistry, TurnMetrics
from app.services.tts import AudioChunk, FallbackTTS, GeminiTTS, LocalVoices, dominant_script, pcm_from_inline_audio
from app.services.voice_pipeline import VoicePipeline
from tests.conftest import make_settings
from tests.fakes import FakeLLM, FakeSTT, FakeTTS
from tests.test_pipeline_streaming import AUDIO, RecordingSink

L16 = "audio/L16;codec=pcm;rate=24000"


def sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in events)


def audio_event(raw: bytes, mime: str = L16) -> dict:
    data = base64.b64encode(raw).decode()
    return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": mime, "data": data}}]}}]}


def wav(pcm: bytes, rate: int) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm


def gemini(handler, **overrides) -> GeminiTTS:
    settings = make_settings(tts_provider="gemini", gemini_api_key="test-key-123", **overrides)
    return GeminiTTS(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def collect(tts, text: str = "Hola, ¿qué tal?") -> list[AudioChunk]:
    return [chunk async for chunk in tts.synthesize(text, threading.Event())]


# ----------------------------------------------------------------------------- GeminiTTS


async def test_streams_audio_chunks_and_builds_the_expected_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = request.url, request.headers, json.loads(request.content)
        return httpx.Response(200, content=sse(audio_event(b"\x01\x00\x02\x00"), audio_event(b"\x03\x00")))

    chunks = await collect(gemini(handler, gemini_tts_voice="Puck"))

    assert b"".join(c.pcm16 for c in chunks) == b"\x01\x00\x02\x00\x03\x00"
    assert len(chunks) == 2, "each streamed piece is passed on as soon as it arrives"
    assert {(c.sample_rate, c.engine) for c in chunks} == {(24000, "gemini gemini-3.8-flash-lite-tts (Puck)")}
    assert seen["url"].path.endswith("/models/gemini-3.8-flash-lite-tts:streamGenerateContent")
    assert seen["url"].params["alt"] == "sse"
    assert seen["headers"]["x-goog-api-key"] == "test-key-123"
    assert "test-key-123" not in str(seen["url"])
    config = seen["body"]["generationConfig"]
    assert config["responseModalities"] == ["AUDIO"]
    assert config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "Hola, ¿qué tal?"


async def test_a_wav_header_is_removed_and_its_sample_rate_used():
    pcm = b"\x10\x00\x20\x00"
    handler = lambda request: httpx.Response(200, content=sse(audio_event(wav(pcm, 22050), "audio/wav")))  # noqa: E731
    [chunk] = await collect(gemini(handler))
    assert (chunk.pcm16, chunk.sample_rate) == (pcm, 22050)


async def test_a_sample_split_across_two_pieces_is_rejoined():
    handler = lambda request: httpx.Response(200, content=sse(audio_event(b"\x01"), audio_event(b"\x00\x02\x00")))  # noqa: E731
    chunks = await collect(gemini(handler))
    assert b"".join(c.pcm16 for c in chunks) == b"\x01\x00\x02\x00"
    assert all(len(c.pcm16) % 2 == 0 for c in chunks)


def test_sample_rate_is_read_from_the_mime_type():
    assert pcm_from_inline_audio("audio/L16;codec=pcm;rate=16000", b"\x00\x00") == (b"\x00\x00", 16000)
    assert pcm_from_inline_audio("audio/L16", b"\x00\x00")[1] == 24000


async def test_after_a_rate_limit_it_refuses_locally_until_the_retry_time():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        message = "Rate limit exceeded for model gemini-3.8-flash-lite-tts (limit: 3 requests per minute on Free Tier). Please retry in 30s."
        return httpx.Response(429, json={"error": {"code": 429, "message": message}})

    tts = gemini(handler)
    with pytest.raises(ProviderError, match="rate-limited"):
        await collect(tts)
    with pytest.raises(ProviderError, match="Try again in"):
        await collect(tts)
    assert calls == 1, "the second request must not be sent while we know it would be refused"


async def test_other_api_errors_are_reported_with_the_servers_message():
    handler = lambda request: httpx.Response(400, json={"error": {"message": "Voice 'Nope' not found."}})  # noqa: E731
    with pytest.raises(ProviderError, match="Voice 'Nope' not found"):
        await collect(gemini(handler))


async def test_without_a_key_loading_fails_clearly():
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        await GeminiTTS(make_settings(tts_provider="gemini")).load()


# ----------------------------------------------------------------------------- FallbackTTS


class Refusing:
    whole_reply = True
    description = "refusing voice"
    info = {"engine": "refusing"}
    sample_rate = 24000

    def __init__(self, after_chunks: int = 0):
        self.after_chunks = after_chunks

    async def load(self):
        pass

    async def synthesize(self, text, cancel):
        for _ in range(self.after_chunks):
            yield AudioChunk(b"\x00\x00", 24000, engine="refusing")
        raise ProviderError("tts", "The Gemini voice is rate-limited (free tier).")

    async def aclose(self):
        pass


class Unloadable(FakeTTS):
    async def load(self):
        raise RuntimeError("voice file missing")


async def loaded(primary, fallback) -> FallbackTTS:
    tts = FallbackTTS(primary, fallback)
    await tts.load()
    return tts


async def test_the_fallback_voice_speaks_when_the_primary_refuses():
    fallback = FakeTTS()
    chunks = await collect(await loaded(Refusing(), fallback), "Paris is the capital of France.")
    assert fallback.spoken == ["Paris is the capital of France."]
    assert chunks and all("rate-limited" in (c.fallback_reason or "") for c in chunks)


async def test_no_fallback_once_the_primary_has_produced_audio():
    fallback = FakeTTS()
    with pytest.raises(ProviderError):
        await collect(await loaded(Refusing(after_chunks=1), fallback), "Hello there.")
    assert fallback.spoken == [], "switching voices mid-reply would repeat the start of it"


async def test_text_the_fallback_cannot_speak_keeps_the_original_error():
    english = FakeTTS()
    with pytest.raises(ProviderError, match="rate-limited"):
        await collect(await loaded(Refusing(), LocalVoices({"latin": english})), "नमस्ते, आप कैसे हैं?")
    assert english.spoken == []


async def test_a_fallback_that_failed_to_load_is_not_used():
    tts = await loaded(Refusing(), Unloadable())
    with pytest.raises(ProviderError, match="rate-limited"):
        await collect(tts, "Hello there.")


def test_script_detection():
    assert dominant_script("Hello, how are you?") == "latin"
    assert dominant_script("¿Dónde está la estación? Ça va très bien.") == "latin"
    assert dominant_script("123 !?") == "latin"
    assert dominant_script("నాకు ఒక చిన్న కథ చెప్పు.") == "telugu"
    assert dominant_script("నాకు Python అంటే ఇష్టం.") == "telugu", "English words inside a Telugu sentence"
    assert dominant_script("नमस्ते, आप कैसे हैं?") == "devanagari"
    assert dominant_script("こんにちは") == "other"


# ----------------------------------------------------------------------------- LocalVoices


class Unnamed(FakeTTS):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.description = name


async def test_each_sentence_goes_to_the_voice_for_its_script():
    english, telugu = Unnamed("english"), Unnamed("telugu")
    voices = LocalVoices({"latin": english, "telugu": telugu})
    await voices.load()
    await collect(voices, "Hello there.")
    await collect(voices, "నమస్కారం, ఎలా ఉన్నారు?")
    assert english.spoken == ["Hello there."]
    assert telugu.spoken == ["నమస్కారం, ఎలా ఉన్నారు?"]
    assert voices.can_speak("Hello") and voices.can_speak("నమస్కారం") and not voices.can_speak("नमस्ते")


async def test_a_script_without_a_voice_is_refused_with_a_hint():
    voices = LocalVoices({"latin": FakeTTS()})
    await voices.load()
    with pytest.raises(ProviderError, match="TTS_TELUGU_VOICE"):
        await collect(voices, "నమస్కారం")


async def test_one_voice_failing_to_load_does_not_silence_the_others():
    english = FakeTTS()
    voices = LocalVoices({"latin": english, "telugu": Unloadable()})
    await voices.load()
    await collect(voices, "Still works.")
    assert english.spoken == ["Still works."]
    assert not voices.can_speak("నమస్కారం")


async def test_no_voice_at_all_is_a_load_error():
    with pytest.raises(ProviderError, match="No local voice"):
        await LocalVoices({"latin": Unloadable()}).load()


async def test_with_the_gemini_voice_refusing_telugu_goes_to_the_local_telugu_voice():
    telugu = Unnamed("telugu")
    tts = await loaded(Refusing(), LocalVoices({"latin": FakeTTS(), "telugu": telugu}))
    chunks = await collect(tts, "భారతదేశ రాజధాని న్యూఢిల్లీ.")
    assert telugu.spoken == ["భారతదేశ రాజధాని న్యూఢిల్లీ."]
    assert chunks and chunks[0].fallback_reason


# ----------------------------------------------------------------------------- the pipeline with such voices


class WholeReplyTTS(FakeTTS):
    whole_reply = True


async def run(tts, stt=None, **overrides):
    llm = FakeLLM()
    pipeline = VoicePipeline(stt or FakeSTT(), llm, tts, make_settings(**overrides), MetricsRegistry())
    sink = RecordingSink()
    metrics = TurnMetrics(request_id="t", transport="ws")
    async with pipeline.turn(metrics):
        await pipeline.run_streaming(AUDIO, [], metrics, sink)
    return sink, metrics, llm


async def test_a_whole_reply_voice_gets_the_reply_in_one_request():
    tts = WholeReplyTTS()
    sink, _, _ = await run(tts)
    assert tts.spoken == ["The capital of France is Paris. It is a lovely city."]
    assert [index for index, _, _ in sink.values("audio")] == [0, 1]


async def test_audio_chunks_are_sent_while_the_voice_is_still_working():
    sink, _, _ = await run(WholeReplyTTS(delay=0.25))
    times = [t for kind, _, t in sink.events if kind == "audio"]
    # Collecting all chunks before sending (the old behaviour) would send both at the same moment.
    assert times[1] - times[0] >= 0.2


async def test_the_turn_records_which_voice_spoke_and_why():
    _, metrics, _ = await run(await loaded(Refusing(), FakeTTS()))
    assert "rate-limited" in metrics.tts_fallback
    assert metrics.language == "en"


async def test_with_language_detection_the_model_is_told_the_language():
    stt = FakeSTT(text="hola, que tal")
    _, _, llm = await run(FakeTTS(), stt=stt, stt_language="auto")
    assert "'en'" in llm.last_messages[0].content  # FakeSTT always reports "en"


async def test_with_a_fixed_language_the_prompt_is_unchanged():
    _, _, llm = await run(FakeTTS())
    assert "speaking the language" not in llm.last_messages[0].content


async def test_a_used_up_daily_quota_is_explained_and_not_retried_until_it_resets():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        message = (
            "You exceeded your current quota. Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 10, "
            "description: Generate content requests per day limit exceeded on Free Tier. Please retry in 1h30m36.8s."
        )
        return httpx.Response(429, json={"error": {"code": 429, "message": message}})

    tts = gemini(handler)
    with pytest.raises(ProviderError, match="used today.s free-tier quota") as first:
        await collect(tts)
    assert "midnight Pacific" in str(first.value) and "1.5 h" in str(first.value)
    with pytest.raises(ProviderError, match="free-tier quota"):
        await collect(tts)
    assert calls == 1


@pytest.mark.parametrize(
    "detail,seconds",
    [("Please retry in 8.5s.", 8.5), ("Please retry in 1h30m36.8s.", 5436.8), ("retry in 45m", 2700.0), ("no hint", None)],
)
def test_retry_times_are_read_from_the_error(detail, seconds):
    from app.services.tts import retry_after_seconds

    assert retry_after_seconds(detail) == (pytest.approx(seconds) if seconds else None)


async def test_a_whole_mixed_reply_is_split_and_each_sentence_gets_its_voice():
    english, telugu = Unnamed("english"), Unnamed("telugu")
    voices = LocalVoices({"latin": english, "telugu": telugu})
    await voices.load()
    await collect(voices, "భారతదేశ రాజధాని న్యూఢిల్లీ. It is a big city. ఇది చాలా అందమైనది.")
    assert telugu.spoken == ["భారతదేశ రాజధాని న్యూఢిల్లీ.", "ఇది చాలా అందమైనది."]
    assert english.spoken == ["It is a big city."]


async def test_after_speaking_starts_a_sentence_without_a_voice_is_skipped_not_fatal():
    english = FakeTTS()
    voices = LocalVoices({"latin": english})
    await voices.load()
    chunks = await collect(voices, "Hello there. నమస్కారం. Goodbye now.")
    assert english.spoken == ["Hello there.", "Goodbye now."]
    assert chunks


# ----------------------------------------------------------------------------- several Gemini voice models


def quota_refusal(model: str) -> httpx.Response:
    message = (  # the wording the API used on 2026-10-27
        "You exceeded your current quota. Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 10, "
        f"model: {model}, description: Generate content requests per day limit exceeded on Free Tier. "
        "Please retry in 2h0m0s."
    )
    return httpx.Response(429, json={"error": {"code": 429, "message": message}})


async def test_when_one_voice_model_is_used_up_the_next_one_speaks():
    asked = []

    def handler(request):
        model = request.url.path.rsplit("/", 1)[-1].split(":")[0]
        asked.append(model)
        return quota_refusal(model) if model == "model-a" else httpx.Response(200, content=sse(audio_event(b"\x01\x00")))

    tts = gemini(handler, gemini_tts_model="model-a, model-b")
    [chunk] = await collect(tts)
    assert chunk.engine == "gemini model-b (Kore)"
    await collect(tts)
    assert asked == ["model-a", "model-b", "model-b"], "a used-up model is not asked again until it resets"


async def test_when_every_voice_model_is_used_up_the_message_says_so_without_naming_them():
    tts = gemini(lambda request: quota_refusal("x"), gemini_tts_model="model-a,model-b")
    with pytest.raises(ProviderError, match="All 2 natural voices have used today's free-tier quota") as refused:
        await collect(tts)
    assert "model-a" not in str(refused.value) and "2.0 h" in str(refused.value)


async def test_a_voice_model_the_key_cannot_use_is_skipped():
    def handler(request):
        if "model-a" in request.url.path:
            return httpx.Response(404, json={"error": {"message": "models/model-a is not found"}})
        return httpx.Response(200, content=sse(audio_event(b"\x01\x00")))

    [chunk] = await collect(gemini(handler, gemini_tts_model="model-a,model-b"))
    assert chunk.engine == "gemini model-b (Kore)"


async def test_local_voices_label_each_chunk_with_its_sentence():
    voices = LocalVoices({"latin": FakeTTS()})
    await voices.load()
    chunks = await collect(voices, "First one. Second one.")
    assert [c.text for c in chunks] == ["First one.", "Second one."]
