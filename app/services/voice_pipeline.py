"""Orchestrates one voice turn: audio -> transcript -> LLM reply -> speech.

Concurrency model
-----------------
* `max_inflight_turns` caps how many turns are admitted at once. Beyond that a new turn
  is rejected immediately ("server busy") instead of joining an unbounded queue.
* Each stage has its own semaphore, so at most N inferences of each kind run at once.
  Time spent waiting for a slot is recorded separately from time spent working, which
  is how you can tell "the model is slow" from "the box is overloaded".
* STT and TTS run in worker threads (see their adapters); the LLM is an async HTTP stream.

Timeouts
--------
Every `await` on a stage has a deadline, and the whole turn has one more. Deadlines are
applied per await rather than around a `yield`, because a timeout scope that stays open
across a yield in an async generator can cancel the wrong task.
"""

import asyncio
import logging
import threading
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from app.config import Settings
from app.errors import (
    ProviderError,
    ServerBusyError,
    ShuttingDownError,
    StageTimeoutError,
    TurnCancelledError,
    VoiceAgentError,
)
from app.services.audio import SAMPLE_RATE, pcm16_seconds
from app.services.documents import DOCUMENT_INSTRUCTIONS, Document, build_context
from app.services.llm import ChatMessage, LanguageModel, LLMUsage
from app.services.metrics import MetricsRegistry, TurnMetrics, elapsed_ms, now
from app.services.speech_text import SentenceChunker, clean_for_speech
from app.services.stt import SpeechToText, Transcript, language_name
from app.services.tts import AudioChunk, TextToSpeech

if TYPE_CHECKING:
    from app.startup import Services

log = logging.getLogger("voice_agent.pipeline")

_DONE = object()
SENTENCE_QUEUE_SIZE = 4


class TurnSink(Protocol):
    """Receives a streaming turn's results as they are produced (the WebSocket session implements this)."""

    async def status(self, state: str) -> None: ...

    async def transcript(self, text: str, final: bool, language: str | None = None) -> None: ...

    async def reply_delta(self, text: str) -> None: ...

    async def audio(self, chunk: AudioChunk, index: int, sentence: str, segment: int) -> None:
        """`sentence`: the words this chunk speaks. `segment` counts those pieces of text; the
        chunks of one piece share it, so the browser can show each piece as it is heard."""
        ...


async def _next_before(iterator, seconds: float, stage: str, message: str):
    """Await the next item of an async iterator, or raise StageTimeoutError."""
    try:
        async with asyncio.timeout(seconds):
            return await anext(iterator, _DONE)
    except TimeoutError:
        raise StageTimeoutError(stage, message) from None


@dataclass(frozen=True)
class TurnOptions:
    """What the user chose for one turn."""

    language: str | None = None  # the language picked in the UI; None = detect it from the audio
    documents: list[Document] = field(default_factory=list)  # attached files to answer questions about
    # The models for this turn, e.g. Gemini with the user's own key. None = the pipeline's default set.
    services: "Services | None" = None


@dataclass
class TurnResult:
    transcript: str
    reply: str
    pcm16: bytes
    sample_rate: int
    language: str | None = None


class VoicePipeline:
    def __init__(
        self,
        stt: SpeechToText,
        llm: LanguageModel,
        tts: TextToSpeech,
        settings: Settings,
        registry: MetricsRegistry,
    ):
        self._stt, self._llm, self._tts = stt, llm, tts
        self._s = settings
        self._registry = registry
        self._stt_slots = asyncio.Semaphore(settings.stt_concurrency)
        self._llm_slots = asyncio.Semaphore(settings.llm_concurrency)
        self._tts_slots = asyncio.Semaphore(settings.tts_concurrency)
        self._inflight = 0
        self._active: set[asyncio.Task] = set()
        self._closing = False

    # ------------------------------------------------------------------ lifecycle

    def stats(self) -> dict:
        return {"inflight_turns": self._inflight, "max_inflight_turns": self._s.max_inflight_turns}

    async def shutdown(self, grace_s: float = 5.0) -> None:
        """Refuse new turns, give running ones a moment to finish, then cancel the rest."""
        self._closing = True
        pending = [task for task in self._active if not task.done()]
        if not pending:
            return
        _, still_running = await asyncio.wait(pending, timeout=grace_s)
        for task in still_running:
            task.cancel()
        await asyncio.gather(*still_running, return_exceptions=True)

    @asynccontextmanager
    async def turn(self, metrics: TurnMetrics):
        """Wrap everything that belongs to one turn.

        Admits or rejects it, applies the overall deadline, tracks it for shutdown, and
        records the outcome (ok / error / timeout / cancelled / rejected) exactly once.
        """
        task = asyncio.current_task()
        admitted = False
        try:
            if self._closing:
                raise ShuttingDownError("The server is shutting down.")
            if self._inflight >= self._s.max_inflight_turns:
                raise ServerBusyError(
                    f"Too many requests in progress ({self._s.max_inflight_turns}). Try again in a moment."
                )
            self._inflight += 1
            admitted = True
            self._active.add(task)
            try:
                async with asyncio.timeout(self._s.turn_timeout_s):
                    yield
            except TimeoutError:
                # Stage-level timeouts are already StageTimeoutError; this is the overall deadline.
                raise StageTimeoutError("turn", f"The turn took longer than {self._s.turn_timeout_s:g}s.") from None
            metrics.finish()
        except asyncio.CancelledError:
            metrics.finish("cancelled", "cancelled")
            raise
        except VoiceAgentError as exc:
            metrics.finish(self._status_for(exc, admitted), exc.code)
            raise
        except Exception:
            metrics.finish("error", "internal_error")
            raise
        finally:
            if admitted:
                self._inflight -= 1
                self._active.discard(task)
            self._registry.record(metrics)
            log.info("turn finished", extra={"fields": metrics.to_dict()})

    @staticmethod
    def _status_for(exc: VoiceAgentError, admitted: bool) -> str:
        if not admitted and isinstance(exc, (ServerBusyError, ShuttingDownError)):
            return "rejected"
        if isinstance(exc, StageTimeoutError):
            return "timeout"
        if isinstance(exc, TurnCancelledError):
            return "cancelled"
        return "error"

    # ------------------------------------------------------------------ the simple, sequential turn

    async def run_batch(
        self, audio: np.ndarray, history: list[ChatMessage], metrics: TurnMetrics, options: TurnOptions | None = None
    ) -> TurnResult:
        """STT, then the whole LLM reply, then TTS of the whole reply. Nothing is sent early."""
        options = options or TurnOptions()
        stt, llm, tts = self._components(options)
        cancel = threading.Event()
        try:
            metrics.audio_seconds = len(audio) / SAMPLE_RATE
            transcript = await self._transcribe(audio, metrics, cancel, options.language, stt)
            if not transcript.text:
                metrics.status = "no_speech"
                return TurnResult("", "", b"", tts.sample_rate)

            reply = await self._reply(transcript, history, metrics, options, llm)
            try:
                chunks = await self._synthesize(reply, metrics, cancel, tts)
            except ProviderError as exc:  # no voice: the reply is still useful as text
                metrics.tts_fallback = f"Text only: {exc}"
                return TurnResult(transcript.text, reply, b"", tts.sample_rate, transcript.language)
            if not chunks:
                raise ProviderError("tts", "The voice produced no audio for this reply.")
            pcm = b"".join(chunk.pcm16 for chunk in chunks)
            metrics.tts_audio_seconds = pcm16_seconds(pcm, chunks[0].sample_rate)
            return TurnResult(transcript.text, reply, pcm, chunks[0].sample_rate, transcript.language)
        finally:
            # A worker thread cannot be interrupted, only asked to stop at its next checkpoint.
            cancel.set()

    # ------------------------------------------------------------------ the streaming turn

    async def run_streaming(
        self,
        audio: np.ndarray,
        history: list[ChatMessage],
        metrics: TurnMetrics,
        sink: "TurnSink",
        options: TurnOptions | None = None,
    ) -> TurnResult:
        """Same stages as run_batch, but results go to `sink` as soon as they exist.

        LLM text is cut into sentences; each sentence is synthesized and sent while the LLM
        keeps generating, so the first audio is ready after the first sentence rather than
        after the whole reply. A voice that should get the whole reply in one request (a
        rate-limited cloud voice) gets it once the LLM has finished; its audio still streams.
        The returned TurnResult carries no audio (it was already sent).
        """
        options = options or TurnOptions()
        stt, llm, tts = self._components(options)
        cancel = threading.Event()
        try:
            metrics.audio_seconds = len(audio) / SAMPLE_RATE
            await sink.status("transcribing")
            transcript = await self._transcribe(audio, metrics, cancel, options.language, stt)
            await sink.transcript(transcript.text, final=True, language=transcript.language)
            if not transcript.text:
                metrics.status = "no_speech"
                return TurnResult("", "", b"", tts.sample_rate)

            await sink.status("thinking")
            # Bounded on purpose: if TTS falls behind, the LLM loop blocks on put() instead of
            # buffering an unlimited amount of text.
            sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=SENTENCE_QUEUE_SIZE)
            reply_parts: list[str] = []
            usage = LLMUsage()
            spoken_seconds = 0.0
            voice_error: str | None = None
            whole_reply = getattr(tts, "whole_reply", False)

            async def generate() -> None:
                chunker = SentenceChunker(first_clause_chars=self._s.tts_first_clause_chars)
                messages = self._messages(transcript.text, history, transcript.language, options)
                async with aclosing(self._stream_llm(messages, metrics, usage, llm)) as deltas:
                    async for delta in deltas:
                        reply_parts.append(delta)
                        await sink.reply_delta(delta)
                        if not whole_reply:
                            for sentence in chunker.feed(delta):
                                await sentences.put(sentence)
                tail = "".join(reply_parts) if whole_reply else chunker.flush()
                if tail:
                    await sentences.put(tail)
                await sentences.put(None)  # end of reply; only reached on success

            async def speak() -> None:
                nonlocal spoken_seconds, voice_error
                index = 0
                segment = -1
                while (sentence := await sentences.get()) is not None:
                    spoken = clean_for_speech(sentence)
                    if not spoken or voice_error:  # keep draining the queue so the LLM is never blocked
                        continue
                    if index == 0:
                        await sink.status("speaking")
                    words = None
                    try:
                        # Each chunk goes out as soon as it exists, so a streaming voice starts playing early.
                        async with aclosing(self._synthesize_stream(spoken, metrics, cancel, tts)) as chunks:
                            async for chunk in chunks:
                                spoken_seconds += len(chunk.pcm16) / 2 / chunk.sample_rate
                                if words is None or (chunk.text or spoken) != words:
                                    words = chunk.text or spoken
                                    segment += 1
                                await sink.audio(chunk, index, words, segment)
                                index += 1
                    except ProviderError as exc:
                        if index:  # part of the reply was already spoken: a half-spoken reply is an error
                            raise
                        voice_error = str(exc)  # no voice at all: the reply is still useful as text

            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(generate())
                    group.create_task(speak())
            except BaseExceptionGroup as failures:
                # Surface the original error (with its error code), not the TaskGroup wrapper.
                errors = list(failures.exceptions)
                raise next((e for e in errors if isinstance(e, VoiceAgentError)), errors[0]) from None

            reply = clean_for_speech("".join(reply_parts))
            metrics.llm_output_chars = len(reply)
            metrics.llm_output_tokens = usage.output_tokens
            metrics.llm_fallback = usage.fallback_reason
            if not reply:
                raise ProviderError("llm", "The language model returned an empty reply.")
            if voice_error:
                metrics.tts_fallback = metrics.tts_fallback or f"Text only: {voice_error}"
            elif spoken_seconds == 0:
                raise ProviderError("tts", "The voice produced no audio for this reply.")
            metrics.tts_audio_seconds = spoken_seconds
            return TurnResult(transcript.text, reply, b"", tts.sample_rate, transcript.language)
        finally:
            cancel.set()

    async def transcribe_partial(self, audio: np.ndarray, cancel: threading.Event) -> str | None:
        """Best-effort interim transcript of audio that is still being recorded.

        Whisper is not a streaming recognizer: this re-decodes everything recorded so far, which
        costs a full inference each time. It never waits for or competes with a real turn:
        if no STT slot is free right now it just skips this round.
        """
        if self._closing or self._stt_slots.locked() or not getattr(self._stt, "supports_partials", True):
            return None
        async with self._stt_slots:
            try:
                async with asyncio.timeout(self._s.stt_timeout_s):
                    return (await self._stt.transcribe(audio, cancel)).text
            except (TimeoutError, TurnCancelledError):
                return None

    # ------------------------------------------------------------------ stages

    def _components(self, options: TurnOptions) -> tuple[SpeechToText, LanguageModel, TextToSpeech]:
        chosen = options.services
        return (chosen.stt, chosen.llm, chosen.tts) if chosen else (self._stt, self._llm, self._tts)

    async def _transcribe(
        self,
        audio: np.ndarray,
        metrics: TurnMetrics,
        cancel: threading.Event,
        language: str | None = None,
        stt: SpeechToText | None = None,
    ) -> Transcript:
        stt = stt or self._stt
        queued = now()
        async with self._stt_slots:
            metrics.stt_queue_ms = elapsed_ms(queued)
            started = now()
            try:
                async with asyncio.timeout(self._s.stt_timeout_s):
                    transcript = await stt.transcribe(audio, cancel, language)
                    metrics.language = transcript.language or None
                    metrics.stt_fallback = transcript.fallback_reason
                    metrics.stt_engine = transcript.engine or None
                    return transcript
            except TimeoutError:
                cancel.set()
                raise StageTimeoutError("stt", f"Transcription took longer than {self._s.stt_timeout_s:g}s.") from None
            finally:
                metrics.stt_ms = elapsed_ms(started)

    def _messages(
        self, user_text: str, history: list[ChatMessage], language: str | None = None, options: TurnOptions | None = None
    ) -> list[ChatMessage]:
        options = options or TurnOptions()
        keep = self._s.max_history_turns * 2
        recent = history[-keep:] if keep else []
        system = self._s.llm_system_prompt
        if options.language:
            name = language_name(options.language)
            system += f" The user chose {name}: always reply in {name}."
        elif self._s.stt_language is None and language:
            # Detected from the audio. A short transcript in a less common language is easy for the
            # model to misread, so say which language it is.
            system += f" The user is speaking the language with code '{language}'."
        if options.documents:
            context = build_context(options.documents, user_text, self._s.document_budget)
            system += f"\n\n{DOCUMENT_INSTRUCTIONS}\n\n{context}"
        return [ChatMessage("system", system), *recent, ChatMessage("user", user_text)]

    async def _stream_llm(
        self, messages: list[ChatMessage], metrics: TurnMetrics, usage: LLMUsage, llm: LanguageModel | None = None
    ):
        """Yield LLM text deltas while holding an LLM slot. Consume it with `aclosing()`."""
        llm = llm or self._llm
        queued = now()
        async with self._llm_slots:
            metrics.llm_queue_ms = elapsed_ms(queued)
            started = now()
            deadline = started + self._s.llm_timeout_s
            first = True
            async with aclosing(llm.stream(messages, usage)) as stream:
                try:
                    while True:
                        remaining = deadline - now()
                        if first:
                            wait = min(remaining, self._s.llm_first_token_timeout_s)
                            message = f"The language model sent no output within {wait:g}s."
                        else:
                            wait = remaining
                            message = f"The language model took longer than {self._s.llm_timeout_s:g}s."
                        delta = await _next_before(stream, wait, "llm", message)
                        if delta is _DONE:
                            return
                        if first:
                            metrics.llm_ttft_ms = elapsed_ms(started)
                            first = False
                        yield delta
                finally:
                    metrics.llm_ms = elapsed_ms(started)

    async def _reply(
        self,
        transcript: Transcript,
        history: list[ChatMessage],
        metrics: TurnMetrics,
        options: TurnOptions,
        llm: LanguageModel | None = None,
    ) -> str:
        usage = LLMUsage()
        parts: list[str] = []
        messages = self._messages(transcript.text, history, transcript.language, options)
        async with aclosing(self._stream_llm(messages, metrics, usage, llm)) as deltas:
            async for delta in deltas:
                parts.append(delta)
        reply = clean_for_speech("".join(parts))
        metrics.llm_output_chars = len(reply)
        metrics.llm_output_tokens = usage.output_tokens
        metrics.llm_fallback = usage.fallback_reason
        if not reply:
            raise ProviderError("llm", "The language model returned an empty reply.")
        return reply

    async def _synthesize(
        self, text: str, metrics: TurnMetrics, cancel: threading.Event, tts: TextToSpeech | None = None
    ) -> list[AudioChunk]:
        """Synthesize `text` completely and return all of its audio."""
        async with aclosing(self._synthesize_stream(text, metrics, cancel, tts)) as stream:
            return [chunk async for chunk in stream]

    async def _synthesize_stream(
        self, text: str, metrics: TurnMetrics, cancel: threading.Event, tts: TextToSpeech | None = None
    ):
        """Yield the audio of `text` chunk by chunk while holding a TTS slot. Consume it with `aclosing()`.

        The deadline is applied to each wait for the voice, never across a `yield`, so a slow
        consumer cannot trigger the TTS timeout (see `_next_before`).
        """
        tts = tts or self._tts
        queued = now()
        async with self._tts_slots:
            metrics.add_ms("tts_queue_ms", elapsed_ms(queued))
            started = now()
            deadline = started + self._s.tts_timeout_s
            try:
                async with aclosing(tts.synthesize(text, cancel)) as stream:
                    while True:
                        message = f"Speech synthesis took longer than {self._s.tts_timeout_s:g}s."
                        chunk = await _next_before(stream, deadline - now(), "tts", message)
                        if chunk is _DONE:
                            break
                        if metrics.tts_first_chunk_ms is None:
                            metrics.tts_first_chunk_ms = elapsed_ms(started)
                            metrics.mark_first_audio()
                        metrics.tts_engine = metrics.tts_engine or chunk.engine or None
                        metrics.tts_fallback = metrics.tts_fallback or chunk.fallback_reason
                        yield chunk
            except StageTimeoutError:
                cancel.set()
                raise
            finally:
                metrics.add_ms("tts_ms", elapsed_ms(started))
