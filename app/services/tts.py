"""Text-to-speech adapters.

* PiperTTS     local, offline (ONNX). Fast on a CPU. Voices exist for many languages, Telugu included.
* KokoroTTS    local Kokoro-82M: a natural-sounding English voice, about real time on a 2-core CPU.
* LocalVoices  the local voices, one per script; each sentence goes to the voice for its script.
* GeminiTTS    Gemini's TTS models over the REST API. Natural voices, and the language is detected
               from the text, so it speaks whatever language the reply is in.
* FallbackTTS  speaks with a primary voice and switches to a second one when the first refuses
               before producing any audio (for example the Gemini free tier's quota).
"""

import asyncio
import base64
import functools
import json
import logging
import re
import struct
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings
from app.errors import ProviderError, TurnCancelledError
from app.services.llm import provider_error_message
from app.services.speech_text import SentenceChunker

log = logging.getLogger("voice_agent.tts")


@dataclass
class AudioChunk:
    """Mono, 16-bit little-endian PCM."""

    pcm16: bytes
    sample_rate: int
    engine: str = ""  # which voice produced it, e.g. "piper te_IN-padmavathi-medium"; shown to the user
    fallback_reason: str | None = None  # set when a fallback voice spoke instead of the configured one
    # The words this chunk speaks, when the voice knows them more precisely than its caller (local
    # voices split a whole reply into sentences). The browser uses it to show text as it is heard.
    text: str = ""


class TextToSpeech(Protocol):
    description: str
    info: dict
    sample_rate: int  # valid after load()

    async def load(self) -> None: ...

    def synthesize(self, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        """Yield audio as it is produced. Piper yields one chunk per sentence."""
        ...

    async def aclose(self) -> None: ...


def ensure_piper_voice(voice: str, root: Path) -> Path:
    """Return the path of the voice's .onnx file, downloading it on first use."""
    model_path = root / f"{voice}.onnx"
    if not (model_path.exists() and (root / f"{voice}.onnx.json").exists()):
        from piper.download_voices import download_voice

        log.info("downloading Piper voice %s to %s", voice, root)
        root.mkdir(parents=True, exist_ok=True)
        download_voice(voice, root)
    return model_path


class PiperTTS:
    whole_reply = False  # cheap per call, so the pipeline speaks sentence by sentence

    def __init__(self, settings: Settings, voice: str | None = None):
        self._settings = settings
        self._voice_name = voice or settings.tts_voice
        self._voice = None
        self.sample_rate = 22050  # replaced by the voice's real rate in load()
        self._executor = ThreadPoolExecutor(max_workers=settings.tts_concurrency, thread_name_prefix="tts")
        self.description = f"piper {self._voice_name}"
        self.info = {"engine": "piper", "voice": self._voice_name}

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_blocking)

    def _load_blocking(self) -> None:
        from piper import PiperVoice

        model_path = ensure_piper_voice(self._voice_name, self._settings.tts_model_dir)
        self._voice = PiperVoice.load(model_path)
        self.sample_rate = self._voice.config.sample_rate
        self.info["sample_rate"] = self.sample_rate
        if self._settings.warmup_on_start:
            for _ in self._voice.synthesize("Ready."):
                pass

    async def synthesize(self, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        if self._voice is None:
            raise ProviderError("tts", "Voice model is not loaded yet.")
        loop = asyncio.get_running_loop()
        chunks = iter(self._voice.synthesize(text))  # lazy: synthesis happens inside next()
        done = object()
        while True:
            if cancel.is_set():
                raise TurnCancelledError("Speech synthesis cancelled.")
            try:
                chunk = await loop.run_in_executor(self._executor, next, chunks, done)
            except Exception as exc:
                raise ProviderError("tts", f"Speech synthesis failed: {exc}") from exc
            if chunk is done:
                return
            yield AudioChunk(pcm16=chunk.audio_int16_bytes, sample_rate=chunk.sample_rate, engine=self.description)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------- Kokoro

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


def ensure_kokoro_files(root: Path) -> tuple[Path, Path]:
    """Return (model, voices), downloading them on first use (about 350 MB)."""
    root.mkdir(parents=True, exist_ok=True)
    for name in KOKORO_FILES:
        target = root / name
        if target.exists():
            continue
        log.info("downloading Kokoro %s to %s", name, root)
        partial = target.with_name(target.name + ".part")  # never leave a half file under the real name
        with httpx.stream("GET", f"{KOKORO_RELEASE}/{name}", follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as response:
            response.raise_for_status()
            with partial.open("wb") as out:
                for block in response.iter_bytes(1 << 20):
                    out.write(block)
        partial.replace(target)
    return root / KOKORO_FILES[0], root / KOKORO_FILES[1]


class KokoroTTS:
    """Kokoro-82M (Apache-2.0) through ONNX Runtime: a natural-sounding English voice. It is slower than
    Piper (about real time on a 2-core laptop CPU), so a sentence can end before the next is ready."""

    whole_reply = False
    sample_rate = 24000

    def __init__(self, settings: Settings):
        self._settings = settings
        self._voice = settings.kokoro_voice
        self._lang = "en-gb" if self._voice.startswith("b") else "en-us"  # Kokoro voice names: a* American, b* British
        self._kokoro = None
        self._executor = ThreadPoolExecutor(max_workers=settings.tts_concurrency, thread_name_prefix="tts-kokoro")
        self.description = f"kokoro {self._voice}"
        self.info = {"engine": "kokoro", "voice": self._voice, "sample_rate": self.sample_rate}

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_blocking)

    def _load_blocking(self) -> None:
        from kokoro_onnx import Kokoro

        model, voices = ensure_kokoro_files(self._settings.kokoro_model_dir)
        kokoro = Kokoro(str(model), str(voices))
        if self._voice not in kokoro.get_voices():
            raise ValueError(f"Kokoro has no voice '{self._voice}'. Try af_heart, af_bella, am_michael or bf_emma.")
        if self._settings.warmup_on_start:
            kokoro.create("Ready.", voice=self._voice, lang=self._lang)
        self._kokoro = kokoro

    async def synthesize(self, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        if self._kokoro is None:
            raise ProviderError("tts", "The Kokoro voice is not loaded yet.")
        if cancel.is_set():
            raise TurnCancelledError("Speech synthesis cancelled.")
        work = functools.partial(self._kokoro.create, text, voice=self._voice, lang=self._lang)
        try:
            samples, rate = await asyncio.get_running_loop().run_in_executor(self._executor, work)
        except Exception as exc:
            raise ProviderError("tts", f"Kokoro synthesis failed: {exc}") from exc
        if cancel.is_set():
            raise TurnCancelledError("Speech synthesis cancelled.")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        yield AudioChunk(pcm16=pcm, sample_rate=rate, engine=self.description)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------- one local voice per script

_SCRIPT_RANGES = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
)


def dominant_script(text: str) -> str:
    """The script most of the letters are written in: "latin", "telugu", "devanagari", ... or "other"."""
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if code < 0x0250:
            counts["latin"] += 1
            continue
        counts[next((name for name, low, high in _SCRIPT_RANGES if low <= code <= high), "other")] += 1
    return counts.most_common(1)[0][0] if counts else "latin"


class LocalVoices:
    """The voices on this machine, one per script (e.g. Kokoro for English, Piper for Telugu).

    Each sentence goes to the voice for its script, so a reply that mixes Telugu and English
    sentences is spoken by both. Text in a script with no voice is refused with an explanation;
    the pipeline then shows the reply as text only.
    """

    whole_reply = False

    def __init__(self, voices: dict[str, TextToSpeech]):
        self._voices = voices
        self._ready: dict[str, TextToSpeech] = {}
        self.description = "local " + ", ".join(f"{v.description} ({script})" for script, v in voices.items())
        self.info = {"engine": "local", "voices": {script: v.info for script, v in voices.items()}}

    @property
    def sample_rate(self) -> int:
        return next(iter(self._ready.values()), next(iter(self._voices.values()))).sample_rate

    async def load(self) -> None:
        failures = []
        for script, voice in self._voices.items():
            try:
                await voice.load()
                self._ready[script] = voice
            except Exception as exc:  # one broken voice should not silence the others
                log.warning("could not load the %s voice %s: %s", script, voice.description, exc)
                failures.append(f"{voice.description}: {exc}")
        if not self._ready:
            raise ProviderError("tts", "No local voice could be loaded. " + "; ".join(failures))

    def can_speak(self, text: str) -> bool:
        return dominant_script(text) in self._ready

    async def synthesize(self, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        # Sentence by sentence even when given a whole reply (as the fallback of a whole-reply voice
        # is): the first sentence plays sooner, and a mixed Telugu/English reply gets both voices.
        chunker = SentenceChunker()
        produced = False
        for sentence in filter(None, [*chunker.feed(text), chunker.flush()]):
            script = dominant_script(sentence)
            voice = self._ready.get(script)
            if voice is None:
                if produced:  # the reply is already being spoken: skip what cannot be, rather than fail
                    log.warning("no local voice for a %s sentence; skipped", script)
                    continue
                hint = " Set TTS_TELUGU_VOICE to add one (see README)." if script == "telugu" else ""
                raise ProviderError("tts", f"There is no local voice for {script} text.{hint}")
            async with aclosing(voice.synthesize(sentence, cancel)) as stream:
                async for chunk in stream:
                    produced = True
                    yield replace(chunk, text=sentence)

    async def aclose(self) -> None:
        for voice in self._voices.values():
            await voice.aclose()


# ---------------------------------------------------------------------------- Gemini

GEMINI_TTS_SAMPLE_RATE = 24000  # Gemini TTS returns 24 kHz, 16-bit, mono PCM
_RETRY_IN = re.compile(r"retry in\s+((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)", re.IGNORECASE)
_RATE_IN_MIME = re.compile(r"rate=(\d+)")
DEFAULT_RATE_LIMIT_BACKOFF_S = 20.0


def retry_after_seconds(detail: str) -> float | None:
    """'Please retry in 8.5s' -> 8.5; 'retry in 1h30m36.8s' -> 5436.8; nothing found -> None."""
    match = _RETRY_IN.search(detail)
    if not match or not match.group(1):
        return None
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", match.group(1)):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def describe_wait(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 90:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def pcm_from_inline_audio(mime_type: str, data: bytes) -> tuple[bytes, int]:
    """Return (16-bit little-endian PCM, sample rate) from one inline audio part.

    Gemini sends either a WAV file (RIFF header) or headerless PCM ("audio/L16;...;rate=24000").
    The header is parsed by hand because a streamed WAV header may carry placeholder sizes.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        rate = struct.unpack_from("<I", data, 24)[0]
        position = 12
        while position + 8 <= len(data):
            chunk_id, size = data[position : position + 4], struct.unpack_from("<I", data, position + 4)[0]
            if chunk_id == b"data":
                return data[position + 8 :], rate
            position += 8 + size
        raise ProviderError("tts", "Gemini returned a WAV file without audio data.")
    match = _RATE_IN_MIME.search(mime_type)
    return data, int(match.group(1)) if match else GEMINI_TTS_SAMPLE_RATE


class _Refused(Exception):
    """A Gemini voice model refused before producing audio, so the next model may be tried."""

    def __init__(self, error: ProviderError | None):
        super().__init__(str(error))
        self.error = error  # None = a rate limit or quota, explained by GeminiTTS._wait_message()


class GeminiTTS:
    """Gemini TTS over REST, streamed as server-sent events (`:streamGenerateContent?alt=sse`).

    One request speaks the whole reply (`whole_reply = True`): the free tier allows only a few TTS
    requests per minute (3 for gemini-3.8-flash-lite-tts on 2026-10-27), so a request per sentence
    would run out after one turn. The audio still streams back in chunks, so playback starts before
    synthesis has finished. After a 429 the adapter refuses locally until the server's "retry in"
    time has passed, so a fallback voice can answer at once instead of after another refusal.

    Several models can be configured (GEMINI_TTS_MODEL=a,b,c). Each has its own free quota, so when
    one refuses the next is tried, and the local fallback speaks only once all of them have refused.
    """

    whole_reply = True

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        key = settings.gemini_api_key
        self._key = key.get_secret_value() if key else ""
        self._models = settings.gemini_tts_models or ["gemini-3.8-flash-lite-tts"]
        self._voice = settings.gemini_tts_voice
        self._base_url = settings.gemini_base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0, read=settings.tts_timeout_s)
        )
        self._blocked_until: dict[str, float] = {}  # model -> time.monotonic() before which it would refuse
        self._daily: dict[str, bool] = {}  # model -> whether that refusal was the daily quota
        self.sample_rate = GEMINI_TTS_SAMPLE_RATE
        self.description = f"gemini {', '.join(self._models)} ({self._voice})"
        self.info = {
            "engine": "gemini",
            "model": self._models[0],
            "models": self._models,
            "voice": self._voice,
            "sample_rate": self.sample_rate,
        }

    async def load(self) -> None:
        # Nothing to download. No test request either: on the free tier it would use up quota.
        if not self._key:
            raise ProviderError("tts", "GEMINI_API_KEY is not set, so the Gemini voice cannot be used.")

    def _refuse(self, model: str, detail: str, seconds: float | None = None) -> None:
        """Remember when this model will accept requests again."""
        delay = seconds or retry_after_seconds(detail) or DEFAULT_RATE_LIMIT_BACKOFF_S
        self._blocked_until[model] = time.monotonic() + delay
        self._daily[model] = "per day" in detail.lower()
        log.warning("gemini voice refused (%s): %s", model, detail)

    def _wait_message(self) -> str:
        """Why no model can speak now, and for how long, without naming the models (the UI shows this)."""
        now = time.monotonic()
        soonest = max(1.0, min(self._blocked_until.get(m, now) - now for m in self._models))
        if all(self._daily.get(m) for m in self._models):
            who = f"All {len(self._models)} natural voices have" if len(self._models) > 1 else "The natural voice has"
            return (
                f"{who} used today's free-tier quota. "
                f"It resets at midnight Pacific time (in about {describe_wait(soonest)})."
            )
        return f"The natural voice is rate-limited on the free tier. Try again in {describe_wait(soonest)}."

    async def synthesize(self, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        if not self._key:
            raise ProviderError("tts", "GEMINI_API_KEY is not set.")
        last: ProviderError | None = None
        for model in self._models:
            if self._blocked_until.get(model, 0.0) > time.monotonic():
                continue
            try:
                async with aclosing(self._stream(model, text, cancel)) as chunks:
                    async for chunk in chunks:
                        yield chunk
                return
            except _Refused as refused:
                last = refused.error
        raise last or ProviderError("tts", self._wait_message())

    async def _stream(self, model: str, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        """One model's audio for `text`. Raises _Refused, before any audio, if this model cannot speak now."""
        body = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}},
            },
        }
        url = f"{self._base_url}/models/{model}:streamGenerateContent"
        headers = {"x-goog-api-key": self._key}  # a header, never the URL, so it stays out of logs
        engine = f"gemini {model} ({self._voice})"
        carry = b""  # a chunk can end in the middle of a 2-byte sample
        try:
            async with self._client.stream("POST", url, params={"alt": "sse"}, headers=headers, json=body) as response:
                if response.status_code != 200:
                    detail = provider_error_message(await response.aread())
                    if response.status_code == 429:
                        self._refuse(model, detail)
                        raise _Refused(None)
                    if response.status_code == 404:  # a model this key cannot use: skip it for a while
                        self._refuse(model, detail, seconds=3600)
                        raise _Refused(ProviderError("tts", f"The voice model {model} is not available: {detail}"))
                    raise ProviderError("tts", f"Gemini voice returned {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if cancel.is_set():
                        raise TurnCancelledError("Speech synthesis cancelled.")
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    for candidate in event.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            inline = part.get("inlineData")
                            if not inline:
                                continue
                            pcm, rate = pcm_from_inline_audio(inline.get("mimeType", ""), base64.b64decode(inline["data"]))
                            pcm = carry + pcm
                            whole = len(pcm) - len(pcm) % 2
                            carry = pcm[whole:]
                            if whole:
                                yield AudioChunk(pcm16=pcm[:whole], sample_rate=rate, engine=engine)
        except httpx.TimeoutException as exc:
            raise ProviderError("tts", "The natural voice stopped responding.") from exc
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderError("tts", f"Gemini voice stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------- fallback


class FallbackTTS:
    """Speak with `primary`; if it fails before any audio was produced, speak with `fallback`.

    Every chunk from the fallback carries `fallback_reason`, so the user is told why the voice
    changed. Text the fallback cannot speak (it has a `can_speak()` that says no, e.g. Telugu with
    no local Telugu voice) keeps the original error; the pipeline then shows the reply as text.
    """

    def __init__(self, primary: TextToSpeech, fallback: TextToSpeech, shared_fallback: bool = False):
        # shared_fallback: the fallback is loaded and closed by its owner (the local voices that
        # every Gemini key shares), not by this wrapper.
        self._primary = primary
        self._fallback = fallback
        self._shared = shared_fallback
        self._fallback_ready = shared_fallback
        self.whole_reply = getattr(primary, "whole_reply", False)
        self.description = f"{primary.description}; fallback {fallback.description}"
        self.info = {**primary.info, "fallback": fallback.info}

    @property
    def sample_rate(self) -> int:
        return self._primary.sample_rate

    async def load(self) -> None:
        await self._primary.load()
        if self._shared:
            return
        try:
            await self._fallback.load()
            self._fallback_ready = True
        except Exception as exc:  # the primary voice still works without it
            log.warning("fallback voice could not be loaded: %s", exc)

    async def synthesize(self, text: str, cancel: threading.Event) -> AsyncIterator[AudioChunk]:
        produced = False
        try:
            async with aclosing(self._primary.synthesize(text, cancel)) as stream:
                async for chunk in stream:
                    produced = True
                    yield chunk
            return
        except ProviderError as exc:
            can_speak = getattr(self._fallback, "can_speak", None)
            if produced or not self._fallback_ready or (can_speak and not can_speak(text)):
                raise
            reason = str(exc)
        log.warning("primary voice failed, using the fallback voice: %s", reason)
        async with aclosing(self._fallback.synthesize(text, cancel)) as stream:
            async for chunk in stream:
                yield replace(chunk, fallback_reason=reason)

    async def aclose(self) -> None:
        await self._primary.aclose()
        if not self._shared:
            await self._fallback.aclose()
