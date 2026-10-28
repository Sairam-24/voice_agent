"""Speech-to-text adapters.

* FasterWhisperSTT  local Whisper (CTranslate2). Private and offline; small models are weak on
                    some languages (Hindi comes out in Urdu script with Whisper base, for example).
* GeminiSTT         transcription by a Gemini model over the REST API. Multilingual and accurate,
                    uses no local CPU, but the audio is sent to Google and counts against the quota.
* FallbackSTT       uses the first and switches to the second when the first refuses.
"""

import asyncio
import base64
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings
from app.errors import InvalidRequestError, ProviderError, TurnCancelledError
from app.services.audio import SAMPLE_RATE, pcm16_to_wav
from app.services.llm import provider_error_message

log = logging.getLogger("voice_agent.stt")


@dataclass
class Transcript:
    text: str
    language: str
    audio_seconds: float
    fallback_reason: str | None = None  # set when a fallback recogniser was used
    engine: str = ""  # which recogniser produced it, shown to the user


LANGUAGE_NAMES = {
    "en": "English", "te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "ar": "Arabic", "ru": "Russian",
}
_CODES_BY_NAME = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def normalize_language(value: str) -> str:
    """'te' -> 'te', 'Telugu' -> 'te', 'en-US' -> 'en'. Models do not always return a bare code."""
    value = value.strip().lower()
    if value in _CODES_BY_NAME:
        return _CODES_BY_NAME[value]
    return value.split("-")[0].split("_")[0]


def language_choices(settings: Settings) -> list[str]:
    """The languages a user may pick for a turn: the fixed one, or those listed in STT_LANGUAGES."""
    return [settings.stt_language] if settings.stt_language else settings.expected_languages


def chosen_language(value, settings: Settings) -> str | None:
    """Validate the language a client picked for a turn. None (or "auto") = detect it from the audio."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"}):
        return None
    code = value.strip().lower() if isinstance(value, str) else ""
    if not re.fullmatch(r"[a-z]{2,3}", code):
        raise InvalidRequestError("'language' must be a language code such as 'en', or 'auto'.")
    choices = language_choices(settings)
    if choices and code not in choices:
        raise InvalidRequestError(f"This server is set up for {', '.join(choices)}, not '{code}'.")
    return code


class SpeechToText(Protocol):
    description: str
    info: dict  # non-secret facts about the loaded model, shown by /api/info and in benchmark results

    async def load(self) -> None: ...

    async def transcribe(self, audio: np.ndarray, cancel: threading.Event, language: str | None = None) -> Transcript:
        """`audio` is float32 mono at 16 kHz. Set `cancel` to stop early. `language`: the one the user
        picked for this turn; None = the configured behaviour (a fixed language, or detection)."""
        ...

    async def aclose(self) -> None: ...


def ensure_whisper_model(name: str, root: Path) -> Path:
    """Return a local model directory, downloading it on first use.

    `name` may also be a path to an existing CTranslate2 model directory.
    """
    if Path(name).is_dir():
        return Path(name)
    target = root / name
    if not (target / "model.bin").exists():
        from faster_whisper.utils import download_model

        log.info("downloading Whisper model %s to %s", name, target)
        target.mkdir(parents=True, exist_ok=True)
        # output_dir writes plain files (no HF cache symlinks, which are awkward on Windows).
        download_model(name, output_dir=str(target))
    return target


class FasterWhisperSTT:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        # CTranslate2 releases the GIL, but inference is still blocking, so it runs in
        # worker threads. The pool size equals the concurrency limit: even if a cancelled
        # coroutine releases its slot early, no more than N inferences can ever run at once.
        self._executor = ThreadPoolExecutor(max_workers=settings.stt_concurrency, thread_name_prefix="stt")
        self.description = (
            f"faster-whisper {settings.stt_model} ({settings.stt_device}, {settings.stt_compute_type})"
        )
        self._allowed = settings.expected_languages
        self.info = {
            "engine": "faster-whisper",
            "model": settings.stt_model,
            "device": settings.stt_device,
            "compute_type": settings.stt_compute_type,
            "cpu_threads": settings.stt_cpu_threads,
            "beam_size": settings.stt_beam_size,
            "vad_filter": settings.stt_vad_filter,
            "language": settings.stt_language or "auto",
        }

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_blocking)

    def _load_blocking(self) -> None:
        from faster_whisper import WhisperModel

        s = self._settings
        path = ensure_whisper_model(s.stt_model, s.stt_model_dir)
        self._model = WhisperModel(
            str(path),
            device=s.stt_device,
            compute_type=s.stt_compute_type,
            cpu_threads=s.stt_cpu_threads,
            num_workers=s.stt_concurrency,
        )
        if s.warmup_on_start:
            # One second of silence with VAD off, so the encoder really runs once.
            self._transcribe_blocking(np.zeros(SAMPLE_RATE, dtype=np.float32), threading.Event(), vad_filter=False)

    async def transcribe(self, audio: np.ndarray, cancel: threading.Event, language: str | None = None) -> Transcript:
        if self._model is None:
            raise ProviderError("stt", "Speech model is not loaded yet.")
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self._transcribe_blocking, audio, cancel, self._settings.stt_vad_filter, language
            )
        except (TurnCancelledError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise ProviderError("stt", f"Transcription failed: {exc}") from exc

    def _transcribe_blocking(
        self, audio: np.ndarray, cancel: threading.Event, vad_filter: bool, chosen: str | None = None
    ) -> Transcript:
        s = self._settings
        multilingual = self._model.model.is_multilingual
        language = chosen if chosen and multilingual else s.stt_language  # an English-only model can only do English
        if language is None and self._allowed and multilingual:
            # Detect, but only among the languages the user speaks.
            _, _, probabilities = self._model.detect_language(audio)
            allowed = [(code, p) for code, p in probabilities if code in self._allowed]
            language = max(allowed, key=lambda item: item[1])[0] if allowed else None
        # transcribe() returns a lazy generator: the work happens while we iterate, which
        # is what lets us check `cancel` between segments. language=None = detect it.
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=s.stt_beam_size,
            vad_filter=vad_filter,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        texts = []
        for segment in segments:
            if cancel.is_set():
                raise TurnCancelledError("Transcription cancelled.")
            texts.append(segment.text.strip())
        return Transcript(
            text=" ".join(t for t in texts if t),
            language=info.language,
            audio_seconds=len(audio) / SAMPLE_RATE,
            engine=f"whisper {s.stt_model}",
        )

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------- Gemini

GEMINI_STT_PROMPT = (
    "Transcribe the speech in this audio word for word, in the language that is actually spoken, written in "
    "that language's own script. Never translate into another language and never answer the speech. "
    "If the speaker mixes in English words, keep them as spoken. If there is no speech, return an empty text."
)
_TRANSCRIPT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"language": {"type": "STRING", "description": "ISO 639-1 code"}, "text": {"type": "STRING"}},
    "required": ["language", "text"],
}


class GeminiSTT:
    """One `generateContent` request per utterance: the audio as WAV plus an instruction, with a JSON
    reply of {language, text}. It is not a streaming recogniser, so it never serves partial
    transcripts (each one would cost a request)."""

    supports_partials = False

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        key = settings.gemini_api_key
        self._key = key.get_secret_value() if key else ""
        self._model = settings.gemini_stt_model
        self._language = settings.stt_language
        self._expected = settings.expected_languages
        self._thinking_level = settings.gemini_thinking_level
        self._url = f"{settings.gemini_base_url.rstrip('/')}/models/{self._model}:generateContent"
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0, read=settings.stt_timeout_s)
        )
        self.description = f"gemini {self._model} (transcription)"
        self.info = {"engine": "gemini", "model": self._model, "language": self._language or "auto"}

    async def load(self) -> None:
        if not self._key:
            raise ProviderError("stt", "GEMINI_API_KEY is not set, so Gemini transcription cannot be used.")

    async def transcribe(self, audio: np.ndarray, cancel: threading.Event, language: str | None = None) -> Transcript:
        if cancel.is_set():
            raise TurnCancelledError("Transcription cancelled.")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        prompt = GEMINI_STT_PROMPT
        fixed = language or self._language
        if fixed:
            prompt += f" The speech is in {language_name(fixed)} ({fixed})."
        elif self._expected:
            # Measured: without this, a Telugu question came back as a Hindi translation every time.
            choices = " or ".join(f"{language_name(code)} ({code})" for code in self._expected)
            prompt += f" The speaker speaks {choices}; the speech is in one of these."
        config: dict = {"responseMimeType": "application/json", "responseSchema": _TRANSCRIPT_SCHEMA, "temperature": 0}
        if self._thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(pcm16_to_wav(pcm, SAMPLE_RATE)).decode()}},
                    {"text": prompt},
                ],
            }],
            "generationConfig": config,
        }
        try:
            response = await self._client.post(self._url, headers={"x-goog-api-key": self._key}, json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError("stt", "Gemini transcription stopped responding.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("stt", f"Gemini transcription failed: {exc}") from exc
        if response.status_code != 200:
            detail = provider_error_message(response.content)
            limited = " (free-tier rate limit)" if response.status_code == 429 else ""
            raise ProviderError("stt", f"Gemini transcription returned {response.status_code}{limited}: {detail}")
        try:
            parts = response.json()["candidates"][0]["content"].get("parts", [])
            result = json.loads("".join(part.get("text", "") for part in parts) or "{}")
        except (ValueError, KeyError, IndexError) as exc:
            raise ProviderError("stt", f"Gemini transcription sent an unexpected reply: {exc}") from exc
        language = normalize_language(str(result.get("language") or fixed or ""))
        return Transcript(
            text=str(result.get("text", "")).strip(),
            language=language,
            audio_seconds=len(audio) / SAMPLE_RATE,
            engine=f"gemini {self._model}",
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class FallbackSTT:
    """Transcribe with `primary`; if it fails (not if it is cancelled), use `fallback` and say why."""

    def __init__(self, primary: SpeechToText, fallback: SpeechToText, shared_fallback: bool = False):
        # shared_fallback: the fallback is loaded and closed by its owner (the local recogniser that
        # every Gemini key shares), not by this wrapper.
        self._primary = primary
        self._fallback = fallback
        self._shared = shared_fallback
        self._fallback_ready = shared_fallback
        self.supports_partials = getattr(primary, "supports_partials", True)
        self.description = f"{primary.description}; fallback {fallback.description}"
        self.info = {**primary.info, "fallback": fallback.info}

    async def load(self) -> None:
        await self._primary.load()
        if self._shared:
            return
        try:
            await self._fallback.load()
            self._fallback_ready = True
        except Exception as exc:  # the primary still works without it
            log.warning("fallback recogniser could not be loaded: %s", exc)

    async def transcribe(self, audio: np.ndarray, cancel: threading.Event, language: str | None = None) -> Transcript:
        try:
            return await self._primary.transcribe(audio, cancel, language)
        except ProviderError as exc:
            if not self._fallback_ready:
                raise
            log.warning("primary recogniser failed, using the fallback: %s", exc)
            return replace(await self._fallback.transcribe(audio, cancel, language), fallback_reason=str(exc))

    async def aclose(self) -> None:
        await self._primary.aclose()
        if not self._shared:
            await self._fallback.aclose()
