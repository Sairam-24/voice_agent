"""WebSocket endpoint: browser microphone in, incremental transcript/text/audio out.

Protocol (JSON text frames unless noted)
----------------------------------------
client -> server
  {"type": "start", "language": "te", "documents": ["id", ...]}
                             begin an utterance. Both fields are optional: language "auto" or absent =
                             detect it; documents = ids from POST /api/documents to answer questions about
  <binary frame>             PCM: signed 16-bit little-endian, mono, 16 kHz (any frame size)
  {"type": "stop"}           end of utterance: run the turn
  {"type": "cancel"}         discard the recording, or abort the running turn
  {"type": "reset"}          forget the conversation history
  {"type": "configure", "gemini_key": "..." | null}
                             use this Gemini API key for this connection's questions (null = the
                             server's default: its own key, or the local models). Never logged.

server -> client
  {"type": "ready", "input": {...}}                         sent once after connecting
  {"type": "configured", "cloud": ["llm", "stt", "tts"]}    after "configure": the stages on Gemini
  {"type": "status", "state": "idle|listening|transcribing|thinking|speaking"}
  {"type": "transcript", "text": "...", "final": false|true, "language": "en"}   language: final only
  {"type": "reply_delta", "text": "..."}                    LLM text as it is generated
  {"type": "audio", "index": n, "sample_rate": 22050, "sentence": "...", "segment": k, "engine": "piper"}
        immediately followed by ONE binary frame of 16-bit mono PCM (a sentence, or part of one).
        "sentence" is the text the chunk speaks; consecutive chunks of one text share "segment".
  {"type": "done", "status": "ok|no_speech", "metrics": {...}}
  {"type": "cancelled"}
  {"type": "error", "code": "...", "message": "..."}       the connection stays open; some codes add
                                                            fields, e.g. document_missing has "ids"
"""

import asyncio
import json
import logging
import threading
from urllib.parse import urlparse

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.errors import AudioTooLongError, InvalidRequestError, NotReadyError, TurnCancelledError, VoiceAgentError
from app.logging_setup import new_request_id, request_id_var
from app.readiness import Readiness, speech_models_ready
from app.services.audio import SAMPLE_RATE, pcm16_to_float32
from app.services.documents import DocumentStore, parse_document_ids
from app.services.llm import ChatMessage
from app.services.metrics import TurnMetrics
from app.services.stt import chosen_language
from app.services.tts import AudioChunk
from app.services.voice_pipeline import TurnOptions, VoicePipeline
from app.startup import Providers

log = logging.getLogger("voice_agent.ws")

MAX_FRAME_BYTES = 256 * 1024  # a 100 ms frame is 3.2 KB; anything this large is a client bug
MIN_PARTIAL_SECONDS = 1.0


def origin_allowed(websocket: WebSocket, extra_origins: str) -> bool:
    """Block cross-site WebSocket hijacking: another website's script must not be able to
    open a socket to a voice agent running on the user's machine."""
    origin = websocket.headers.get("origin")
    if origin is None:  # not a browser (scripts, benchmarks)
        return True
    allowed = {o.strip() for o in extra_origins.split(",") if o.strip()}
    return urlparse(origin).netloc == websocket.headers.get("host") or origin in allowed


class VoiceSession:
    """One browser connection. Handles one utterance/turn at a time and keeps the chat history."""

    def __init__(
        self,
        websocket: WebSocket,
        pipeline: VoicePipeline,
        settings: Settings,
        readiness: Readiness,
        documents: DocumentStore | None = None,
        providers: Providers | None = None,
    ):
        self._ws = websocket
        self._pipeline = pipeline
        self._settings = settings
        self._readiness = readiness
        self._documents = documents or DocumentStore()
        self._options = TurnOptions()  # chosen at "start", used by the turn that "stop" runs
        self._providers = providers
        self._services = providers.default if providers else None  # changed by "configure"
        self._send_lock = asyncio.Lock()  # a JSON header and its audio frame must stay adjacent
        self._connection_id = new_request_id()
        self._turn_count = 0
        self._history: list[ChatMessage] = []
        self._recording = False
        self._buffer = bytearray()
        self._max_buffer_bytes = int(settings.max_audio_seconds * SAMPLE_RATE * 2)
        self._turn_task: asyncio.Task | None = None
        self._partial_task: asyncio.Task | None = None
        self._partial_cancel = threading.Event()

    # ------------------------------------------------------------------ connection loop

    async def run(self) -> None:
        request_id_var.set(self._connection_id)
        if not origin_allowed(self._ws, self._settings.allowed_origins):
            log.warning("rejected websocket from origin %s", self._ws.headers.get("origin"))
            await self._ws.close(code=1008)  # policy violation
            return
        await self._ws.accept()
        log.info("websocket connected")
        try:
            await self._send_json(
                {
                    "type": "ready",
                    "input": {"encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE, "channels": 1},
                    "max_audio_seconds": self._settings.max_audio_seconds,
                    "partial_transcripts": self._settings.partial_transcripts,
                    "models_ready": speech_models_ready(self._readiness.snapshot()),
                }
            )
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self._on_audio(message["bytes"])
                elif message.get("text") is not None:
                    await self._on_control(message["text"])
        except (WebSocketDisconnect, TurnCancelledError):
            pass
        finally:
            log.info("websocket closed")
            await self._cancel_work()

    async def _on_control(self, text: str) -> None:
        try:
            message = json.loads(text)
            kind = message.get("type")
        except (ValueError, AttributeError):
            await self._send_error(InvalidRequestError("Control messages must be JSON objects with a 'type'."))
            return
        try:
            if kind == "start":
                await self._start_recording(message)
            elif kind == "stop":
                await self._stop_recording()
            elif kind == "cancel":
                await self._cancel_current()
            elif kind == "reset":
                self._history.clear()
            elif kind == "configure":
                await self._configure(message.get("gemini_key"))
            else:
                raise InvalidRequestError(f"Unknown message type: {kind!r}")
        except VoiceAgentError as exc:
            await self._send_error(exc)

    async def _configure(self, key) -> None:
        if key is not None and (not isinstance(key, str) or not 0 < len(key.strip()) <= 200):
            raise InvalidRequestError("'gemini_key' must be a Gemini API key, or null.")
        if self._providers is None:
            return
        self._services = self._providers.for_key(key.strip() if key else None)
        await self._send_json({"type": "configured", "cloud": sorted(self._services.cloud)})

    # ------------------------------------------------------------------ recording

    async def _start_recording(self, message: dict) -> None:
        if self._turn_task and not self._turn_task.done():
            raise InvalidRequestError("A turn is still in progress. Cancel it or wait for it to finish.")
        if not speech_models_ready(self._readiness.snapshot()):
            raise NotReadyError("The speech models are still loading or failed to load. Check /health.")
        # Checked now rather than at "stop", so a missing file is reported before the user speaks.
        self._options = TurnOptions(
            language=chosen_language(message.get("language"), self._settings),
            documents=self._documents.resolve(parse_document_ids(message.get("documents"))),
            services=self._services,
        )
        self._recording = True
        self._buffer.clear()
        await self._send_json({"type": "status", "state": "listening"})
        if self._settings.partial_transcripts:
            self._partial_cancel = threading.Event()
            self._partial_task = asyncio.create_task(self._partial_loop(self._partial_cancel))

    async def _on_audio(self, frame: bytes) -> None:
        if not self._recording:
            return  # late frames after stop/cancel are expected; ignore them
        if len(frame) > MAX_FRAME_BYTES:
            self._discard_recording()
            await self._send_error(InvalidRequestError("Audio frame too large."))
            return
        if len(self._buffer) + len(frame) > self._max_buffer_bytes:
            self._discard_recording()
            await self._send_error(
                AudioTooLongError(f"Recording is longer than the {self._settings.max_audio_seconds:g}s limit.")
            )
            await self._send_json({"type": "status", "state": "idle"})
            return
        self._buffer.extend(frame)

    def _discard_recording(self) -> None:
        self._recording = False
        self._buffer.clear()
        self._stop_partials()

    def _stop_partials(self) -> None:
        self._partial_cancel.set()
        if self._partial_task:
            self._partial_task.cancel()
            self._partial_task = None

    async def _stop_recording(self) -> None:
        if not self._recording:
            raise InvalidRequestError("Not recording. Send 'start' first.")
        samples = pcm16_to_float32(bytes(self._buffer))
        self._discard_recording()
        self._turn_count += 1
        self._turn_task = asyncio.create_task(self._run_turn(samples, f"{self._connection_id}-{self._turn_count}"))

    async def _partial_loop(self, cancel: threading.Event) -> None:
        """Periodically transcribe what has been recorded so far (opt-in; see Settings)."""
        last_length = 0
        try:
            while True:
                await asyncio.sleep(self._settings.partial_interval_s)
                if len(self._buffer) <= last_length or len(self._buffer) < MIN_PARTIAL_SECONDS * SAMPLE_RATE * 2:
                    continue
                last_length = len(self._buffer)
                text = await self._pipeline.transcribe_partial(pcm16_to_float32(bytes(self._buffer)), cancel)
                if text and self._recording:
                    await self._send_json({"type": "transcript", "text": text, "final": False})
        except (VoiceAgentError, WebSocketDisconnect):
            return  # partials are best-effort; the final transcript is what matters
        except Exception:
            log.exception("partial transcription loop failed; continuing without partials")

    # ------------------------------------------------------------------ one turn

    async def _run_turn(self, samples: np.ndarray, turn_id: str) -> None:
        request_id_var.set(turn_id)
        metrics = TurnMetrics(request_id=turn_id, transport="ws")
        try:
            async with self._pipeline.turn(metrics):
                result = await self._pipeline.run_streaming(samples, list(self._history), metrics, self, self._options)
            if result.transcript:
                self._remember(result.transcript, result.reply)
            await self._send_json({"type": "done", "status": metrics.status, "metrics": metrics.to_dict()})
        except asyncio.CancelledError:
            await self._send_quietly({"type": "cancelled"})
        except TurnCancelledError:
            pass  # the client is gone; nobody to tell
        except VoiceAgentError as exc:
            await self._send_quietly(
                {"type": "error", "code": exc.code, "message": exc.message, "request_id": turn_id, **(exc.details or {})}
            )
        except Exception:
            log.exception("turn failed unexpectedly")
            await self._send_quietly(
                {"type": "error", "code": "internal_error", "message": "Internal server error", "request_id": turn_id}
            )
        finally:
            await self._send_quietly({"type": "status", "state": "idle"})

    def _remember(self, user_text: str, reply: str) -> None:
        self._history += [ChatMessage("user", user_text), ChatMessage("assistant", reply)]
        keep = self._settings.max_history_turns * 2
        self._history = self._history[-keep:] if keep else []

    async def _cancel_current(self) -> None:
        if self._recording:
            self._discard_recording()
            await self._send_json({"type": "cancelled"})
            await self._send_json({"type": "status", "state": "idle"})
        elif self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()  # _run_turn reports "cancelled" and returns to idle

    async def _cancel_work(self) -> None:
        self._stop_partials()
        task = self._turn_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # ------------------------------------------------------------------ TurnSink (called by the pipeline)

    async def status(self, state: str) -> None:
        await self._send_json({"type": "status", "state": state})

    async def transcript(self, text: str, final: bool, language: str | None = None) -> None:
        message = {"type": "transcript", "text": text, "final": final}
        if language:
            message["language"] = language
        await self._send_json(message)

    async def reply_delta(self, text: str) -> None:
        await self._send_json({"type": "reply_delta", "text": text})

    async def audio(self, chunk: AudioChunk, index: int, sentence: str, segment: int = 0) -> None:
        header = {
            "type": "audio",
            "index": index,
            "sample_rate": chunk.sample_rate,
            "sentence": sentence,
            "segment": segment,
            "engine": chunk.engine,
        }
        async with self._send_lock:
            await self._guarded(self._ws.send_text(json.dumps(header)))
            await self._guarded(self._ws.send_bytes(chunk.pcm16))

    # ------------------------------------------------------------------ sending

    async def _guarded(self, awaitable) -> None:
        """Turn 'the socket is gone' into TurnCancelledError so the pipeline stops working."""
        try:
            await awaitable
        except (WebSocketDisconnect, RuntimeError, OSError) as exc:
            raise TurnCancelledError("The client disconnected.") from exc

    async def _send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self._guarded(self._ws.send_text(json.dumps(payload)))

    async def _send_quietly(self, payload: dict) -> None:
        try:
            await self._send_json(payload)
        except TurnCancelledError:
            pass

    async def _send_error(self, exc: VoiceAgentError) -> None:
        await self._send_quietly({"type": "error", "code": exc.code, "message": exc.message, **(exc.details or {})})
