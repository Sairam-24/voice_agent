"""Deterministic stand-ins for the STT, LLM and TTS adapters.

They exist for fast, repeatable unit tests only. They are never selectable in the real
app, and nothing measured with them may be reported as model performance.
"""

import asyncio
import threading

import numpy as np

from app.errors import ProviderError, TurnCancelledError
from app.services.llm import ChatMessage, LLMUsage
from app.services.stt import Transcript
from app.services.tts import AudioChunk


class ConcurrencyProbe:
    """Records how many calls overlapped, to check that configured limits are respected."""

    def __init__(self) -> None:
        self.current = 0
        self.maximum = 0
        self.total = 0

    def enter(self) -> None:
        self.current += 1
        self.total += 1
        self.maximum = max(self.maximum, self.current)

    def exit(self) -> None:
        self.current -= 1


class FakeSTT:
    description = "fake stt"
    info = {"engine": "fake-stt"}

    def __init__(
        self,
        text: str = "what is the capital of france",
        delay: float = 0.0,
        error: Exception | None = None,
        load_delay: float = 0.0,
        load_error: Exception | None = None,
    ):
        self.text, self.delay, self.error = text, delay, error
        self.load_delay, self.load_error = load_delay, load_error
        self.probe = ConcurrencyProbe()
        self.loaded = False
        self.closed = False
        self.cancel_seen = False
        self.last_audio_seconds: float | None = None

    async def load(self) -> None:
        await asyncio.sleep(self.load_delay)
        if self.load_error:
            raise self.load_error
        self.loaded = True

    async def transcribe(self, audio: np.ndarray, cancel: threading.Event, language: str | None = None) -> Transcript:
        self.last_language = language
        self.probe.enter()
        self.last_audio_seconds = len(audio) / 16000
        try:
            await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            return Transcript(text=self.text, language="en", audio_seconds=self.last_audio_seconds)
        except asyncio.CancelledError:
            self.cancel_seen = True
            raise
        finally:
            self.probe.exit()

    async def aclose(self) -> None:
        self.closed = True


class FakeLLM:
    description = "fake llm"
    info = {"provider": "fake-llm"}

    def __init__(
        self,
        reply: str = "The capital of France is Paris. It is a lovely city.",
        first_token_delay: float = 0.0,
        token_delay: float = 0.0,
        error: Exception | None = None,
        healthy: bool = True,
    ):
        self.reply = reply
        self.first_token_delay, self.token_delay = first_token_delay, token_delay
        self.error = error
        self.healthy = healthy
        self.probe = ConcurrencyProbe()
        self.last_messages: list[ChatMessage] = []
        self.yielded = 0  # words handed to the pipeline so far
        self.cancelled = False
        self.closed = False
        self.warmed_up = False

    async def check(self) -> tuple[bool, str]:
        return (True, self.description) if self.healthy else (False, "fake llm is down")

    async def warm_up(self) -> None:
        self.warmed_up = True

    async def stream(self, messages: list[ChatMessage], usage: LLMUsage):
        self.probe.enter()
        self.last_messages = list(messages)
        self.yielded = 0
        try:
            if self.error:
                raise self.error
            words = self.reply.split(" ")
            for index, word in enumerate(words):
                await asyncio.sleep(self.first_token_delay if index == 0 else self.token_delay)
                self.yielded += 1
                yield word if index == len(words) - 1 else word + " "
            usage.prompt_tokens = 10
            usage.output_tokens = len(words)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.probe.exit()

    async def aclose(self) -> None:
        self.closed = True


class FakeTTS:
    description = "fake tts"
    info = {"engine": "fake-tts"}
    sample_rate = 22050

    def __init__(self, delay: float = 0.0, error: Exception | None = None, seconds_per_sentence: float = 0.2):
        self.delay, self.error = delay, error
        self.seconds_per_sentence = seconds_per_sentence
        self.probe = ConcurrencyProbe()
        self.spoken: list[str] = []
        self.loaded = False
        self.closed = False

    async def load(self) -> None:
        self.loaded = True

    async def synthesize(self, text: str, cancel: threading.Event):
        self.probe.enter()
        self.spoken.append(text)
        try:
            if self.error:
                raise self.error
            # Like Piper: one chunk per sentence.
            for _ in [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]:
                if cancel.is_set():
                    raise TurnCancelledError("cancelled")
                await asyncio.sleep(self.delay)
                samples = int(self.sample_rate * self.seconds_per_sentence)
                yield AudioChunk(pcm16=b"\x01\x00" * samples, sample_rate=self.sample_rate)
        finally:
            self.probe.exit()

    async def aclose(self) -> None:
        self.closed = True


def provider_error(stage: str, message: str = "boom") -> ProviderError:
    return ProviderError(stage, message)
