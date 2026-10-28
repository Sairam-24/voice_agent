"""Per-turn timing and a small in-memory summary of recent turns.

All durations come from `time.perf_counter()`, which is monotonic and high
resolution. `time.time()` can jump (NTP, DST), so it is never used for durations.
"""

import time
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field, fields


def now() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return (now() - start) * 1000.0


def percentile(values: Iterable[float], pct: float) -> float:
    """Linear-interpolated percentile (the same definition numpy uses by default)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile() of an empty sequence")
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass
class TurnMetrics:
    """Timings for one voice turn. A `None` field means that stage never ran or is not measurable.

    Definitions (all in milliseconds, measured on the server):
      upload_ms           HTTP only: request start until the audio upload was parsed.
      decode_ms           HTTP only: decoding the upload to 16 kHz PCM.
      stt_queue_ms       waiting for a free STT slot.   stt_ms: transcription itself.
      llm_queue_ms        waiting for a free LLM slot.
      llm_ttft_ms         LLM time to first token (from the moment the LLM request started).
      llm_ms              total LLM generation time.
      tts_queue_ms        waiting for a TTS slot, summed over sentences.
      tts_first_chunk_ms  TTS time to the first audio chunk of the turn (excludes queue wait).
      tts_ms              total TTS synthesis time, summed over sentences.
      first_audio_ms      turn start (input audio complete) until the first audio chunk
                          existed on the server. This is the latency a user perceives
                          with streaming; for the plain HTTP endpoint the client only
                          receives audio at total_ms.
      total_ms            turn start until the turn finished.
    """

    request_id: str
    transport: str  # "http" or "ws"
    status: str = "in_progress"  # ok | no_speech | error | timeout | cancelled | rejected
    error_code: str | None = None
    audio_seconds: float | None = None
    upload_ms: float | None = None
    decode_ms: float | None = None
    stt_queue_ms: float | None = None
    stt_ms: float | None = None
    llm_queue_ms: float | None = None
    llm_ttft_ms: float | None = None
    llm_ms: float | None = None
    tts_queue_ms: float | None = None
    tts_first_chunk_ms: float | None = None
    tts_ms: float | None = None
    first_audio_ms: float | None = None
    total_ms: float | None = None
    llm_output_chars: int = 0
    llm_output_tokens: int | None = None
    tts_audio_seconds: float | None = None
    language: str | None = None  # detected (or configured) language of the user's speech
    stt_fallback: str | None = None  # why a fallback recogniser was used, if it was
    llm_fallback: str | None = None  # why the local model answered instead of the cloud one, if it did
    stt_engine: str | None = None  # the recogniser that transcribed this turn
    tts_engine: str | None = None  # the voice that spoke this reply
    tts_fallback: str | None = None  # why a fallback voice was used, if it was
    _started: float = field(default_factory=now, repr=False, compare=False)

    def add_ms(self, name: str, value: float) -> None:
        """Accumulate into a field; used where a stage runs once per sentence."""
        setattr(self, name, (getattr(self, name) or 0.0) + value)

    def since_start_ms(self) -> float:
        return elapsed_ms(self._started)

    def mark_first_audio(self) -> None:
        if self.first_audio_ms is None:
            self.first_audio_ms = self.since_start_ms()

    def finish(self, status: str | None = None, error_code: str | None = None) -> None:
        """Stop the clock. With no status, keep one set earlier (e.g. "no_speech"), else "ok"."""
        if self.total_ms is not None:  # already finished; keep the first outcome
            return
        self.total_ms = self.since_start_ms()
        self.status = status or (self.status if self.status != "in_progress" else "ok")
        self.error_code = error_code

    def to_dict(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in data.items()}


# Fields summarised by MetricsRegistry (durations only).
TIMING_FIELDS = (
    "upload_ms",
    "decode_ms",
    "stt_queue_ms",
    "stt_ms",
    "llm_queue_ms",
    "llm_ttft_ms",
    "llm_ms",
    "tts_queue_ms",
    "tts_first_chunk_ms",
    "tts_ms",
    "first_audio_ms",
    "total_ms",
)


class MetricsRegistry:
    """Keeps the last `keep` finished turns in memory. Bounded on purpose: no unbounded growth."""

    def __init__(self, keep: int = 200):
        self._recent: deque[dict] = deque(maxlen=keep)
        self._status_counts: Counter[str] = Counter()

    def record(self, metrics: TurnMetrics) -> None:
        self._recent.append(metrics.to_dict())
        self._status_counts[metrics.status] += 1

    def recent(self, limit: int = 20) -> list[dict]:
        return list(self._recent)[-limit:]

    def summary(self) -> dict:
        """Median/p95 per stage over recent successful turns, plus outcome counts since start."""
        ok_turns = [t for t in self._recent if t["status"] == "ok"]
        stages = {}
        for name in TIMING_FIELDS:
            values = [t[name] for t in ok_turns if t.get(name) is not None]
            if values:
                stages[name] = {
                    "n": len(values),
                    "median": round(percentile(values, 50), 1),
                    "p95": round(percentile(values, 95), 1),
                }
        return {
            "turns_since_start": dict(self._status_counts),
            "window_size": len(self._recent),
            "stages_ms": stages,
        }
