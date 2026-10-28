"""Response schemas for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    state: Literal["loading", "ready", "error"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "degraded"]
    version: str
    uptime_s: float
    components: dict[str, ComponentHealth]


class HistoryMessage(BaseModel):
    """One earlier message in the conversation, sent by the client."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AudioOut(BaseModel):
    format: Literal["wav"] = "wav"
    sample_rate: int
    duration_s: float
    data_base64: str


class TurnResponse(BaseModel):
    request_id: str
    status: Literal["ok", "no_speech"]
    transcript: str
    language: str | None = None  # language of the speech, as detected by (or configured for) STT
    reply: str
    audio: AudioOut | None
    metrics: dict


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
