"""FastAPI application: routes, middleware and lifecycle.

Run with:  uvicorn app.main:create_app --factory
"""

import asyncio
import base64
import json
import logging
import platform
from contextlib import asynccontextmanager, suppress
from importlib.metadata import version as package_version
from pathlib import PurePath

from fastapi import FastAPI, File, Form, Header, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import Settings, get_settings
from app.errors import (
    InvalidRequestError,
    NotReadyError,
    TurnCancelledError,
    UnsupportedDocumentError,
    UploadTooLargeError,
    VoiceAgentError,
)
from app.logging_setup import accept_request_id, configure_logging, request_id_var
from app.readiness import Readiness, all_ready, overall_status, speech_models_ready
from app.schemas import (
    AudioOut,
    ComponentHealth,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    HistoryMessage,
    TurnResponse,
)
from app.services.audio import decode_audio_capped, pcm16_seconds, pcm16_to_wav
from app.services.documents import MAX_DOCUMENTS_PER_TURN, DocumentStore, parse_document_ids, read_document
from app.services.llm import ChatMessage, check_gemini_key
from app.services.metrics import MetricsRegistry, TurnMetrics, elapsed_ms, now
from app.services.stt import chosen_language, language_choices
from app.services.voice_pipeline import TurnOptions, VoicePipeline
from app.startup import Providers, Services, build_local, load_components
from app.voice_socket import VoiceSession
from app.webui import resolve_web_dirs

log = logging.getLogger("voice_agent.http")

# Polled by the browser every few seconds; logging them at INFO would drown real traffic.
QUIET_PATHS = {"/health", "/ready"}

LLM_CHECK_TTL_S = 5.0
DOCUMENT_READ_TIMEOUT_S = 60.0
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # form boundaries and headers around the audio bytes
INFO_PACKAGES = ("faster-whisper", "ctranslate2", "piper-tts", "onnxruntime", "av", "numpy", "fastapi", "uvicorn", "httpx")
_history_adapter = TypeAdapter(list[HistoryMessage])
MAX_KEY_CHARS = 200


class KeyCheck(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_CHARS)


class RequestIdMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware, which breaks streaming responses).

    Assigns a request id, exposes it as the X-Request-ID response header, stores it
    in a ContextVar for logging, and writes one access-log line per request.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        request_id = accept_request_id(headers.get(b"x-request-id", b"").decode("latin-1") or None)
        token = request_id_var.set(request_id)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["started_at"] = now()
        status_code = 500

        async def send_with_request_id(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", request_id.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            level = logging.DEBUG if scope["path"] in QUIET_PATHS else logging.INFO
            log.log(
                level,
                "%s %s -> %s",
                scope["method"],
                scope["path"],
                status_code,
                extra={"fields": {"duration_ms": round(elapsed_ms(state["started_at"]), 1)}},
            )
            request_id_var.reset(token)


def error_response(
    status_code: int, code: str, message: str, request_id: str, details: dict | None = None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id)).model_dump()
    body["error"].update(details or {})  # e.g. the ids of missing documents
    return JSONResponse(status_code=status_code, content=body)


class BodyLimitMiddleware:
    """Refuse an oversized upload from its Content-Length header, before any of it is read.

    Without this, Starlette spools the whole multipart body to memory or a temp file before the
    route can check its size. A request with no Content-Length (chunked encoding) is not caught
    here; a reverse proxy's body-size limit is the real defence for a public deployment.
    """

    def __init__(self, app, max_bytes: int, path_limits: dict[str, int] | None = None):
        self.app = app
        self.max_bytes = max_bytes
        self.path_limits = path_limits or {}  # a different limit for some paths (attached documents)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            limit = self.path_limits.get(scope["path"], self.max_bytes)
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > limit:
                request_id = scope.get("state", {}).get("request_id", "-")
                message = f"Request body is larger than {limit} bytes."
                response = error_response(413, "upload_too_large", message, request_id)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def run_until_disconnect(request: Request, work):
    """Run `work` (a coroutine), cancelling it if the client goes away first.

    Starlette does not cancel a handler when the client disconnects, so without this a
    user who closes the tab would leave an STT/LLM/TTS turn burning CPU for nobody.
    """
    task = asyncio.create_task(work)

    async def watch_for_disconnect():
        # Starlette offers no awaitable "client left" event, only is_disconnected(), so poll it.
        while not await request.is_disconnected():  # noqa: ASYNC110
            await asyncio.sleep(0.25)

    watcher = asyncio.create_task(watch_for_disconnect())
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        raise TurnCancelledError("The client disconnected.")
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def parse_history(raw: str | None, limit: int) -> list[ChatMessage]:
    if not raw:
        return []
    try:
        messages = _history_adapter.validate_json(raw)
    except ValidationError as exc:
        raise InvalidRequestError(
            f"'history' must be a JSON list of {{role, content}} objects: {exc.errors()[0]['msg']}"
        ) from None
    return [ChatMessage(m.role, m.content) for m in messages[-limit:]] if limit else []


def parse_document_field(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return parse_document_ids(json.loads(raw))
    except ValueError:
        raise InvalidRequestError("'documents' must be a JSON list of document ids.") from None


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app. Tests pass fake `services` (used as the local models); production builds the real
    adapters at startup. Gemini components are added on top of them per key (see app/startup.py)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        providers = Providers(settings, services or build_local(settings))
        app.state.providers = providers
        default = providers.default
        app.state.pipeline = VoicePipeline(default.stt, default.llm, default.tts, settings, app.state.metrics)
        loader = asyncio.create_task(load_components(providers, settings, app.state.readiness), name="model-loader")
        try:
            yield
        finally:
            log.info("shutting down")
            loader.cancel()
            with suppress(asyncio.CancelledError):
                await loader
            await app.state.pipeline.shutdown()
            await providers.aclose()

    app = FastAPI(title="Voice Agent", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.started_at = now()
    app.state.readiness = Readiness()
    app.state.metrics = MetricsRegistry()
    app.state.llm_check = (0.0, ComponentHealth(state="loading"))
    app.state.documents = DocumentStore()
    # Added last = outermost, so the request id exists before the size check can reject anything.
    app.add_middleware(
        BodyLimitMiddleware,
        max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
        path_limits={"/api/documents": settings.max_document_bytes + MULTIPART_OVERHEAD_BYTES},
    )
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(VoiceAgentError)
    async def handle_voice_agent_error(request: Request, exc: VoiceAgentError):
        log.warning("request failed: %s", exc.message, extra={"fields": {"error_code": exc.code}})
        return error_response(exc.http_status, exc.code, exc.message, request.state.request_id, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(part) for part in first.get("loc", ()))
        message = f"Invalid request ({where}): {first.get('msg', 'validation failed')}"
        return error_response(422, "invalid_request", message, request.state.request_id)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        # 404/405 etc. from the router: same error shape as everything else.
        return error_response(exc.status_code, f"http_{exc.status_code}", str(exc.detail), request.state.request_id)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        log.exception("unhandled error")
        return error_response(500, "internal_error", "Internal server error", request.state.request_id)

    async def components() -> dict[str, ComponentHealth]:
        """STT/TTS state comes from the loader; the LLM is checked live (cached briefly)."""
        snapshot = app.state.readiness.snapshot()
        llm = getattr(app.state, "providers", None) and app.state.providers.default.llm
        if llm is not None and snapshot:  # only once loading has begun
            checked_at, cached = app.state.llm_check
            if now() - checked_at > LLM_CHECK_TTL_S or cached.state == "loading":
                ok, detail = await llm.check()
                cached = ComponentHealth(state="ready" if ok else "error", detail=detail)
                app.state.llm_check = (now(), cached)
            snapshot["llm"] = cached
        return snapshot

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Liveness plus component state. Always 200 while the process is up."""
        snapshot = await components()
        return HealthResponse(
            status=overall_status(snapshot),
            version=__version__,
            uptime_s=round(elapsed_ms(app.state.started_at) / 1000, 1),
            components=snapshot,
        )

    @app.get("/ready")
    async def ready():
        """Readiness for load balancers: 200 only once every component is usable."""
        is_ready = all_ready(await components())
        return JSONResponse(status_code=200 if is_ready else 503, content={"ready": is_ready})

    @app.get("/api/metrics")
    async def metrics():
        registry: MetricsRegistry = app.state.metrics
        pipeline = getattr(app.state, "pipeline", None)
        return {
            "load": pipeline.stats() if pipeline else None,
            "summary": registry.summary(),
            "recent_turns": registry.recent(20),
        }

    @app.get("/api/info")
    async def info():
        """What this server is running: versions and non-secret settings. Benchmarks save it with their results."""
        providers: Providers = app.state.providers
        active = providers.default
        return {
            "app_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: package_version(name) for name in INFO_PACKAGES},
            "components": {"stt": active.stt.info, "llm": active.llm.info, "tts": active.tts.info},
            # What the UI offers: the languages to pick from ("auto" = detect them), and the file limits.
            "languages": {"choices": language_choices(settings), "auto": settings.stt_language is None},
            # Whether this server has its own Gemini key (never the key itself), and which stages use
            # Gemini when there is one. Users can also bring their own key.
            "cloud": {"server_key": providers.server_key, "stages": sorted(settings.cloud_stages)},
            "documents": {
                "max_bytes": settings.max_document_bytes,
                "max_per_turn": MAX_DOCUMENTS_PER_TURN,
                "context_chars": settings.document_budget,
            },
            "settings": {
                name: getattr(settings, name)
                for name in (
                    "warmup_on_start", "stt_concurrency", "llm_concurrency", "tts_concurrency", "max_inflight_turns",
                    "stt_timeout_s", "llm_first_token_timeout_s", "llm_timeout_s", "tts_timeout_s", "turn_timeout_s",
                    "max_audio_seconds", "max_history_turns", "partial_transcripts", "tts_first_clause_chars",
                    "llm_system_prompt",
                )
            },
        }

    @app.post("/api/gemini-key/check")
    async def check_key(body: KeyCheck):
        """Whether a Gemini API key works, so the UI can say so before the key is used. The key is
        not stored or logged; the browser keeps it and sends it with the user's questions."""
        valid, message = await check_gemini_key(settings, body.key.strip())
        return {"valid": valid, "message": message}

    @app.post("/api/documents")
    async def upload_document(file: UploadFile = File(...)):
        """Attach a file to ask questions about. Its text is kept in memory only; send the returned
        id with each turn (WebSocket "start", or the `documents` field of /api/turn)."""
        data = await file.read(settings.max_document_bytes + 1)
        if len(data) > settings.max_document_bytes:
            raise UploadTooLargeError(f"The file is larger than {settings.max_document_bytes // 1_000_000} MB.")
        if not data:
            raise UnsupportedDocumentError("The file is empty.")
        name = PurePath((file.filename or "document").replace("\\", "/")).name[:120] or "document"
        try:
            async with asyncio.timeout(DOCUMENT_READ_TIMEOUT_S):
                text, pages, truncated = await asyncio.to_thread(read_document, name, data)
        except TimeoutError:
            raise UnsupportedDocumentError("Reading the file took too long.") from None
        document = app.state.documents.add(name, text, pages, truncated)
        log.info("document attached", extra={"fields": {"chars": len(text), "pages": pages, "truncated": truncated}})
        return document.summary()

    @app.delete("/api/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str):
        app.state.documents.remove(document_id)
        return Response(status_code=204)

    @app.post("/api/turn", response_model=TurnResponse)
    async def create_turn(
        request: Request,
        audio: UploadFile = File(...),
        history: str | None = Form(None),
        language: str | None = Form(None),
        documents: str | None = Form(None),
        gemini_key: str | None = Header(None, alias="X-Gemini-Key", max_length=MAX_KEY_CHARS),
    ):
        """One complete voice turn over plain HTTP: upload audio, get transcript, reply and WAV.

        This is NOT streaming: the response is sent after the whole reply has been
        synthesized. Use /ws/voice for incremental results.
        """
        pipeline: VoicePipeline = app.state.pipeline
        if not speech_models_ready(app.state.readiness.snapshot()):
            raise NotReadyError("The speech models are still loading or failed to load. Check /health.")

        metrics = TurnMetrics(request_id=request.state.request_id, transport="http")
        metrics.upload_ms = elapsed_ms(request.state.started_at)

        async with pipeline.turn(metrics):
            data = await audio.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise UploadTooLargeError(f"Upload is larger than {settings.max_upload_bytes} bytes.")
            past_messages = parse_history(history, settings.max_history_turns * 2)
            options = TurnOptions(
                language=chosen_language(language, settings),
                documents=app.state.documents.resolve(parse_document_field(documents)),
                services=app.state.providers.for_key(gemini_key.strip() if gemini_key else None),
            )

            decode_started = now()
            samples = await asyncio.to_thread(decode_audio_capped, data, settings.max_audio_seconds)
            metrics.decode_ms = elapsed_ms(decode_started)

            result = await run_until_disconnect(request, pipeline.run_batch(samples, past_messages, metrics, options))

        audio_out = None
        if result.pcm16:
            wav = pcm16_to_wav(result.pcm16, result.sample_rate)
            audio_out = AudioOut(
                sample_rate=result.sample_rate,
                duration_s=round(pcm16_seconds(result.pcm16, result.sample_rate), 3),
                data_base64=base64.b64encode(wav).decode("ascii"),
            )
        return TurnResponse(
            request_id=metrics.request_id,
            status="ok" if result.transcript else "no_speech",
            transcript=result.transcript,
            language=result.language,
            reply=result.reply,
            audio=audio_out,
            metrics=metrics.to_dict(),
        )

    @app.websocket("/ws/voice")
    async def voice_socket(websocket: WebSocket):
        """Streaming voice turns. The message protocol is documented in app/voice_socket.py."""
        await VoiceSession(
            websocket, app.state.pipeline, settings, app.state.readiness, app.state.documents, app.state.providers
        ).run()

    # Mounted last so they never shadow the API routes above. /classic must come before the
    # catch-all "/" mount.
    main_ui, classic_ui = resolve_web_dirs(settings.web_dir)
    if classic_ui:
        app.mount("/classic", StaticFiles(directory=classic_ui, html=True), name="classic")
    if main_ui:
        app.mount("/", StaticFiles(directory=main_ui, html=True), name="web")

    return app
