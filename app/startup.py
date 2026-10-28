"""Builds the STT/LLM/TTS adapters and loads their models in the background.

There are two kinds of component:

* local ones (Whisper, Ollama, the local voices), built and loaded once;
* Gemini ones, which need an API key. They are built per key on top of the local ones, which stay
  their fallbacks. The server's own key (GEMINI_API_KEY) gives the default set; a key a user enters
  in the browser gives a set used only for that user's questions.

With no key anywhere, everything runs locally.
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass

from pydantic import SecretStr

from app.config import Settings
from app.readiness import Readiness
from app.services.llm import FallbackLLM, GeminiLLM, LanguageModel, OllamaLLM
from app.services.metrics import elapsed_ms, now
from app.services.stt import FallbackSTT, FasterWhisperSTT, GeminiSTT, SpeechToText
from app.services.tts import FallbackTTS, GeminiTTS, KokoroTTS, LocalVoices, PiperTTS, TextToSpeech

log = logging.getLogger("voice_agent.startup")

KEY_CACHE_SIZE = 32  # users' own keys with a component set kept ready


@dataclass
class Services:
    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech
    cloud: frozenset[str] = frozenset()  # the stages that use Gemini in this set


def build_local_voices(settings: Settings) -> LocalVoices:
    english: TextToSpeech = KokoroTTS(settings) if settings.tts_local_english == "kokoro" else PiperTTS(settings)
    voices = {"latin": english}
    if settings.tts_telugu_voice:
        voices["telugu"] = PiperTTS(settings, voice=settings.tts_telugu_voice)
    return LocalVoices(voices)


def build_local(settings: Settings) -> Services:
    return Services(stt=FasterWhisperSTT(settings), llm=OllamaLLM(settings), tts=build_local_voices(settings))


def build_cloud(settings: Settings, local: Services) -> Services:
    """Gemini for each stage that may use it (`settings` carries the key), on top of the local components,
    which stay the fallbacks. Those are shared and already loaded: this set never loads or closes them."""
    stages = settings.cloud_stages
    stt, llm, tts = local.stt, local.llm, local.tts
    if "stt" in stages:
        stt = GeminiSTT(settings)
        if settings.stt_fallback_to_whisper:
            stt = FallbackSTT(stt, local.stt, shared_fallback=True)
    if "llm" in stages:
        llm = GeminiLLM(settings)
        if settings.llm_fallback_to_ollama:
            llm = FallbackLLM(llm, local.llm, shared_fallback=True)
    if "tts" in stages:
        tts = GeminiTTS(settings)
        if settings.tts_fallback_to_local:
            tts = FallbackTTS(tts, local.tts, shared_fallback=True)
    return Services(stt=stt, llm=llm, tts=tts, cloud=stages)


class Providers:
    """The components a question runs on: the default set, or the set for a user's own Gemini key."""

    def __init__(self, settings: Settings, local: Services):
        self._settings = settings
        self.local = local
        use_server_key = settings.gemini_api_key is not None and settings.cloud_stages
        self.default = build_cloud(settings, local) if use_server_key else local
        self._by_key: OrderedDict[str, Services] = OrderedDict()

    @property
    def server_key(self) -> bool:
        """Whether questions without a user's own key run on Gemini with the server's key."""
        return self.default is not self.local

    def for_key(self, key: str | None) -> Services:
        """The components for a user's own Gemini key, or the default set without one."""
        if not key or not self._settings.cloud_stages:
            return self.default
        digest = hashlib.sha256(key.encode()).hexdigest()  # the cache is not keyed by the secret itself
        services = self._by_key.get(digest)
        if services is None:
            settings = self._settings.model_copy(update={"gemini_api_key": SecretStr(key)})
            services = self._by_key[digest] = build_cloud(settings, self.local)
            while len(self._by_key) > KEY_CACHE_SIZE:
                # Not closed: a connection may still be using it. Its HTTP clients go with it.
                self._by_key.popitem(last=False)
        self._by_key.move_to_end(digest)
        return services

    async def aclose(self) -> None:
        local = {id(self.local.stt), id(self.local.llm), id(self.local.tts)}
        cloud_sets = [*self._by_key.values(), *([self.default] if self.server_key else [])]
        own = [c for s in cloud_sets for c in (s.stt, s.llm, s.tts) if id(c) not in local]
        shared = [self.local.stt, self.local.llm, self.local.tts]
        await asyncio.gather(*(c.aclose() for c in [*own, *shared]), return_exceptions=True)


async def load_components(providers: Providers, settings: Settings, readiness: Readiness) -> None:
    """Load the local models one after another. Loading in parallel would only make them fight for the same cores.

    The server is already accepting connections while this runs; /health reports
    "starting" until it finishes, so the UI can show progress instead of a blank page.
    """
    local = providers.local
    for name, adapter in (("stt", local.stt), ("tts", local.tts)):
        readiness.set(name, "loading", adapter.description)
    for name, adapter in (("stt", local.stt), ("tts", local.tts)):
        started = now()
        try:
            await adapter.load()
        except Exception as exc:  # report any load failure through /health instead of crashing startup
            log.exception("failed to load %s", name)
            readiness.set(name, "error", f"Could not load {adapter.description}: {exc}")
            continue
        readiness.set(name, "ready", adapter.description)
        log.info("loaded %s", name, extra={"fields": {"component": name, "load_ms": round(elapsed_ms(started))}})

    llm = providers.default.llm
    ok, detail = await llm.check()
    if not ok:
        log.warning("language model not ready: %s", detail)
    elif settings.warmup_on_start:
        started = now()
        try:
            await llm.warm_up()
            log.info("warmed up llm", extra={"fields": {"warmup_ms": round(elapsed_ms(started))}})
        except Exception as exc:
            log.warning("llm warm-up failed: %s", exc)
