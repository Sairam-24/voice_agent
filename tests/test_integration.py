"""Tests that use real models or real external services.

Run only these:   pytest -m integration -v
Skip them:        pytest -m "not integration"

Each test skips itself, with a reason shown by `-ra`, when its prerequisite is missing:
  * Whisper / Piper tests need the files from `python scripts/download_models.py`.
  * Ollama tests need a running Ollama with the configured model pulled.
  * Gemini tests need GEMINI_API_KEY in the environment (they make one tiny real request).
"""

import asyncio
import base64
import io
import os
import threading
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import ProviderError, TurnCancelledError
from app.main import create_app
from app.services.audio import decode_audio_capped, pcm16_to_wav
from app.services.llm import ChatMessage, GeminiLLM, LLMUsage, OllamaLLM
from app.services.stt import FasterWhisperSTT, GeminiSTT
from app.services.tts import GeminiTTS, KokoroTTS, LocalVoices, PiperTTS
from app.startup import Services
from tests.conftest import make_settings, wait_until_ready
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration

DEFAULTS = Settings(_env_file=None)
HAVE_WHISPER = (DEFAULTS.stt_model_dir / DEFAULTS.stt_model / "model.bin").exists()
HAVE_PIPER = (DEFAULTS.tts_model_dir / f"{DEFAULTS.tts_voice}.onnx").exists()
needs_speech_models = pytest.mark.skipif(
    not (HAVE_WHISPER and HAVE_PIPER), reason="run `python scripts/download_models.py` first"
)

PHRASE = "Hello, what is the capital of France?"


def real_settings(**overrides) -> Settings:
    # Same model choices as the defaults, but quiet logs and no warm-up to keep the run short.
    return make_settings(stt_model=DEFAULTS.stt_model, tts_voice=DEFAULTS.tts_voice, **overrides)


@pytest.fixture(scope="module")
def stt():
    adapter = FasterWhisperSTT(real_settings())
    asyncio.run(adapter.load())
    return adapter


@pytest.fixture(scope="module")
def tts():
    adapter = PiperTTS(real_settings())
    asyncio.run(adapter.load())
    return adapter


@pytest.fixture(scope="module")
def spoken_phrase(tts) -> bytes:
    """A WAV of real synthesized speech, used as repeatable test input for the STT."""

    async def synth() -> bytes:
        pcm = b"".join([c.pcm16 async for c in tts.synthesize(PHRASE, threading.Event())])
        return pcm16_to_wav(pcm, tts.sample_rate)

    return asyncio.run(synth())


# ------------------------------------------------------------------------ TTS

@needs_speech_models
async def test_piper_yields_valid_pcm_chunks(tts):
    chunks = [c async for c in tts.synthesize("Hello there. This is a second sentence.", threading.Event())]
    assert len(chunks) == 2  # Piper yields one chunk per sentence
    for chunk in chunks:
        assert chunk.sample_rate == tts.sample_rate == 22050
        assert len(chunk.pcm16) % 2 == 0 and len(chunk.pcm16) > 2 * 22050 * 0.3
    samples = np.frombuffer(b"".join(c.pcm16 for c in chunks), dtype="<i2")
    assert np.abs(samples).max() > 1000, "output is silent"


@needs_speech_models
async def test_piper_stops_when_cancelled(tts):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TurnCancelledError):
        async for _ in tts.synthesize("One. Two. Three.", cancel):
            pytest.fail("a cancelled synthesis must not produce audio")


# ------------------------------------------------------------------------ STT

@needs_speech_models
async def test_whisper_transcribes_synthesized_speech(stt, spoken_phrase):
    audio = decode_audio_capped(spoken_phrase, max_seconds=30)
    transcript = await stt.transcribe(audio, threading.Event())
    text = transcript.text.lower()
    assert "capital" in text and "france" in text, text
    assert transcript.language == "en"
    assert transcript.audio_seconds == pytest.approx(len(audio) / 16000)


@needs_speech_models
async def test_whisper_returns_empty_text_for_silence(stt):
    transcript = await stt.transcribe(np.zeros(16000 * 2, dtype=np.float32), threading.Event())
    assert transcript.text == ""


@needs_speech_models
async def test_whisper_stops_when_cancelled(stt, spoken_phrase):
    cancel = threading.Event()
    cancel.set()
    audio = decode_audio_capped(spoken_phrase, max_seconds=30)
    with pytest.raises(TurnCancelledError):
        await stt.transcribe(audio, cancel)


# ------------------------------------------------------------------------ whole app, real STT + TTS

@needs_speech_models
def test_full_http_turn_with_real_speech_models_and_a_fake_llm(spoken_phrase):
    """Real Whisper and Piper through the real HTTP stack; only the LLM is replaced."""
    llm = FakeLLM(reply="The capital of France is Paris.")
    settings = real_settings()
    app = create_app(settings, Services(FasterWhisperSTT(settings), llm, PiperTTS(settings)))
    with TestClient(app) as client:
        wait_until_ready(client, timeout=60)
        response = client.post("/api/turn", files={"audio": ("speech.wav", spoken_phrase, "audio/wav")})

    assert response.status_code == 200, response.text
    body = response.json()
    assert "france" in body["transcript"].lower()
    assert llm.last_messages[-1].content == body["transcript"]  # the LLM got what Whisper heard
    wav = base64.b64decode(body["audio"]["data_base64"])
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 22050 and w.getnframes() / 22050 > 1.0
    assert body["metrics"]["stt_ms"] > 0 and body["metrics"]["tts_ms"] > 0


# ------------------------------------------------------------------------ real LLM providers

def _ollama_ready() -> tuple[bool, str]:
    settings = Settings(_env_file=None)
    try:
        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1.0).json()
    except (httpx.HTTPError, ValueError):
        return False, f"Ollama is not running at {settings.ollama_base_url}"
    wanted = settings.ollama_model if ":" in settings.ollama_model else settings.ollama_model + ":latest"
    if wanted not in {m["name"] for m in tags.get("models", [])}:
        return False, f"model not installed: ollama pull {settings.ollama_model}"
    return True, ""


_OLLAMA_OK, _OLLAMA_WHY = _ollama_ready()


@pytest.mark.skipif(not _OLLAMA_OK, reason=_OLLAMA_WHY)
async def test_real_ollama_streams_a_reply_and_reports_usage():
    llm = OllamaLLM(make_settings(llm_max_output_tokens=30))
    usage = LLMUsage()
    messages = [ChatMessage("system", "Answer in one short sentence."), ChatMessage("user", "What is 2 plus 2?")]
    try:
        deltas = [d async for d in llm.stream(messages, usage)]
        ok, detail = await llm.check()
    finally:
        await llm.aclose()
    assert "".join(deltas).strip()
    assert len(deltas) > 1, "expected incremental deltas, got one blob"
    assert usage.output_tokens and usage.output_tokens > 0
    assert ok, detail


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
async def test_real_gemini_streams_a_reply():
    llm = GeminiLLM(make_settings(llm_provider="gemini", gemini_api_key=os.environ["GEMINI_API_KEY"], llm_max_output_tokens=200))
    usage = LLMUsage()
    messages = [ChatMessage("system", "Answer in one short sentence."), ChatMessage("user", "What is 2 plus 2?")]
    try:
        text = "".join([d async for d in llm.stream(messages, usage)])
    finally:
        await llm.aclose()
    assert "4" in text or "four" in text.lower()
    assert usage.output_tokens


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
async def test_real_gemini_transcribes_speech():
    if not HAVE_PIPER:
        pytest.skip("needs the Piper voice to make test speech")
    tts = PiperTTS(make_settings())
    await tts.load()
    pcm = b"".join([c.pcm16 async for c in tts.synthesize("What is the capital of France?", threading.Event())])
    audio = decode_audio_capped(pcm16_to_wav(pcm, tts.sample_rate), 30)
    stt = GeminiSTT(make_settings(stt_provider="gemini", stt_language="auto", gemini_api_key=os.environ["GEMINI_API_KEY"]))
    try:
        transcript = await stt.transcribe(audio, threading.Event())
    finally:
        await stt.aclose()
        await tts.aclose()
    assert "capital" in transcript.text.lower() and "france" in transcript.text.lower()
    assert transcript.language == "en"


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
async def test_real_gemini_voice_streams_audio():
    """Skips (says why) when the free tier's voice quota is used up: 10 requests a day when written."""
    tts = GeminiTTS(make_settings(tts_provider="gemini", gemini_api_key=os.environ["GEMINI_API_KEY"]))
    try:
        chunks = [c async for c in tts.synthesize("Hello there.", threading.Event())]
    except ProviderError as exc:
        if "quota" in str(exc) or "rate-limited" in str(exc):
            pytest.skip(f"Gemini voice quota: {exc}")
        raise
    finally:
        await tts.aclose()
    assert chunks and all(c.sample_rate == 24000 and len(c.pcm16) % 2 == 0 for c in chunks)
    assert sum(len(c.pcm16) for c in chunks) / 2 / 24000 > 0.3, "expected at least a short phrase of audio"


HAVE_KOKORO = all((DEFAULTS.kokoro_model_dir / name).exists() for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"))
TELUGU_VOICE = "te_IN-padmavathi-medium"
HAVE_TELUGU = (DEFAULTS.tts_model_dir / f"{TELUGU_VOICE}.onnx").exists()


@pytest.mark.skipif(not HAVE_KOKORO, reason="Kokoro files missing: set TTS_LOCAL_ENGLISH=kokoro and run download_models.py")
async def test_real_kokoro_speaks_english():
    tts = KokoroTTS(make_settings())
    await tts.load()
    try:
        chunks = [c async for c in tts.synthesize("Hello there, how are you?", threading.Event())]
    finally:
        await tts.aclose()
    seconds = sum(len(c.pcm16) for c in chunks) / 2 / chunks[0].sample_rate
    assert chunks[0].sample_rate == 24000 and 0.8 < seconds < 5
    assert np.abs(np.frombuffer(b"".join(c.pcm16 for c in chunks), dtype="<i2")).max() > 1000, "not silence"


@pytest.mark.skipif(not HAVE_TELUGU, reason=f"{TELUGU_VOICE} missing: set TTS_TELUGU_VOICE and run download_models.py")
async def test_real_local_voices_speak_telugu_and_english_in_one_reply():
    voices = LocalVoices({"latin": PiperTTS(make_settings()), "telugu": PiperTTS(make_settings(), voice=TELUGU_VOICE)})
    await voices.load()
    try:
        chunks = [c async for c in voices.synthesize("నమస్కారం, మీరు ఎలా ఉన్నారు? I am fine.", threading.Event())]
    finally:
        await voices.aclose()
    assert [c.engine for c in chunks] == [f"piper {TELUGU_VOICE}", f"piper {DEFAULTS.tts_voice}"]
    assert all(len(c.pcm16) > 22050 for c in chunks), "each sentence should be at least half a second"
