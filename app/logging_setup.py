"""Structured logging with a per-request id.

The id lives in a ContextVar, so every log line written while handling a request
carries it without passing it around. asyncio tasks copy the context when they are
created, so tasks spawned for a turn keep the id of the request that started them.
"""

import json
import logging
import re
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def accept_request_id(candidate: str | None) -> str:
    """Reuse a client-supplied id only if it is short and log-safe."""
    if candidate and _SAFE_REQUEST_ID.match(candidate):
        return candidate
    return new_request_id()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": record.request_id,
            "msg": record.getMessage(),
        }
        # Structured fields come from logger.info("...", extra={"fields": {...}}).
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler()
    handler._voice_agent = True  # marks it as ours so a re-configure does not remove other handlers
    handler.addFilter(_RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_voice_agent", False)] + [handler]
    root.setLevel(level.upper())
    # Uvicorn's access log duplicates our own per-request log line.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
