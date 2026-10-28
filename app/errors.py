"""Errors raised by the pipeline and mapped to HTTP/WebSocket responses.

Each class carries a stable machine-readable `code`. The browser shows a specific
message per code, and tests assert on the code instead of matching message text.
"""


class VoiceAgentError(Exception):
    code = "internal_error"
    http_status = 500
    details: dict | None = None  # extra machine-readable fields for the error body

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedAudioError(VoiceAgentError):
    code = "unsupported_audio"
    http_status = 415


class EmptyAudioError(VoiceAgentError):
    code = "empty_audio"
    http_status = 422


class AudioTooLongError(VoiceAgentError):
    code = "audio_too_long"
    http_status = 413


class UploadTooLargeError(VoiceAgentError):
    code = "upload_too_large"
    http_status = 413


class InvalidRequestError(VoiceAgentError):
    code = "invalid_request"
    http_status = 422


class ServerBusyError(VoiceAgentError):
    code = "server_busy"
    http_status = 429


class ShuttingDownError(VoiceAgentError):
    code = "shutting_down"
    http_status = 503


class NotReadyError(VoiceAgentError):
    code = "models_not_ready"
    http_status = 503


class StageTimeoutError(VoiceAgentError):
    http_status = 504

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.code = f"{stage}_timeout"
        self.stage = stage


class ProviderError(VoiceAgentError):
    """An STT/LLM/TTS backend failed or is unreachable."""

    http_status = 502

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.code = f"{stage}_error"
        self.stage = stage


class TurnCancelledError(VoiceAgentError):
    # 499 is nginx's "client closed request"; there is no standard code for this.
    code = "cancelled"
    http_status = 499


class UnsupportedDocumentError(VoiceAgentError):
    code = "unsupported_document"
    http_status = 415


class DocumentNotFoundError(VoiceAgentError):
    """An attached document is no longer on the server (it restarted, or the document was evicted)."""

    code = "document_missing"
    http_status = 404

    def __init__(self, message: str, ids: list[str]):
        super().__init__(message)
        self.details = {"ids": ids}
