"""Measure how much STT, LLM and TTS slow each other down when they share the same CPU.

    python scripts/contention_probe.py            # needs the models and a running Ollama; no server needed

Each operation is timed alone, then again while a background loop keeps one other stage busy:

    TTS  : synthesize a fixed sentence                       (seconds per sentence)
    STT  : transcribe benchmarks/inputs/medium.wav           (seconds per utterance)
    LLM  : generate a reply to a fixed prompt                (output tokens per second)

Run it on an otherwise idle machine and read the slowdown column. A slowdown near 1.0x means the
two stages do not interfere; 2.0x means the operation took twice as long (or produced half the
tokens per second) while the other stage was running. Raw timings are saved to
benchmarks/results/contention-<timestamp>.json.
"""

import asyncio
import json
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import cpu_busy_percent, machine_info  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.audio import decode_audio_capped  # noqa: E402
from app.services.llm import ChatMessage, LLMUsage, OllamaLLM  # noqa: E402
from app.services.stt import FasterWhisperSTT  # noqa: E402
from app.services.tts import PiperTTS  # noqa: E402

SENTENCE = "The capital of France is Paris, and it is known for its museums and its food."
LLM_PROMPT = [
    ChatMessage("system", "You are a helpful assistant."),
    ChatMessage("user", "Tell me about the history of the bicycle in a few paragraphs."),
]
REPS = 5


class Probe:
    def __init__(self) -> None:
        settings = get_settings().model_copy(update={"llm_max_output_tokens": 80})
        self.stt, self.tts, self.llm = FasterWhisperSTT(settings), PiperTTS(settings), OllamaLLM(settings)
        self.audio = decode_audio_capped(Path("benchmarks/inputs/medium.wav").read_bytes(), 60)
        self.cancel = threading.Event()

    async def load(self) -> None:
        await self.stt.load()
        await self.tts.load()
        ok, detail = await self.llm.check()  # also fills llm.info with the model digest
        if not ok:
            raise SystemExit(detail)
        await self.llm.warm_up()

    # One timed unit of work per stage. Each returns its own figure of merit.
    async def tts_once(self) -> float:
        started = time.perf_counter()
        async for _ in self.tts.synthesize(SENTENCE, self.cancel):
            pass
        return time.perf_counter() - started  # seconds, lower is better

    async def stt_once(self) -> float:
        started = time.perf_counter()
        await self.stt.transcribe(self.audio, self.cancel)
        return time.perf_counter() - started

    async def llm_once(self) -> float:
        usage = LLMUsage()
        started = time.perf_counter()
        async for _ in self.llm.stream(LLM_PROMPT, usage):
            pass
        return (usage.output_tokens or 0) / (time.perf_counter() - started)  # tokens/s, higher is better

    async def measure(self, operation, reps: int, background=None) -> list[float]:
        """Time `operation` `reps` times, optionally while `background` is looping in another task."""
        stop = asyncio.Event()

        async def keep_busy() -> None:
            while not stop.is_set():
                await background()

        loader = asyncio.create_task(keep_busy()) if background else None
        try:
            if loader:
                await asyncio.sleep(2)  # let the background stage reach steady state first
            return [await operation() for _ in range(reps)]
        finally:
            stop.set()
            if loader:
                await loader


def row(name: str, unit: str, alone: list[float], busy: list[float], higher_is_better: bool, against: str) -> dict:
    a, b = statistics.median(alone), statistics.median(busy)
    slowdown = (a / b) if higher_is_better else (b / a)
    print(f"{name:<6} while {against:<4} busy: {a:8.2f} -> {b:8.2f} {unit:<8} slowdown {slowdown:4.1f}x")
    return {"operation": name, "against": against, "unit": unit, "alone": alone, "contended": busy,
            "alone_median": a, "contended_median": b, "slowdown": slowdown}


async def main() -> None:
    probe = Probe()
    await probe.load()
    print(f"background CPU before: {await cpu_busy_percent()}%  ({REPS} repetitions per cell, medians shown)\n")
    ops = {"TTS": (probe.tts_once, "s", False), "STT": (probe.stt_once, "s", False), "LLM": (probe.llm_once, "tok/s", True)}
    results = []
    for name, (operation, unit, higher_is_better) in ops.items():
        alone = await probe.measure(operation, REPS)
        for other, (other_operation, _, _) in ops.items():
            if other != name:
                busy = await probe.measure(operation, REPS, background=other_operation)
                results.append(row(name, unit, alone, busy, higher_is_better, other))
    print(f"\nbackground CPU after: {await cpu_busy_percent()}%")

    out = Path("benchmarks/results") / f"contention-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": machine_info(), "reps": REPS, "results": results,
                               "stt": probe.stt.info, "llm": probe.llm.info, "tts": probe.tts.info}, indent=2))
    print(f"saved {out}")
    await asyncio.gather(probe.stt.aclose(), probe.tts.aclose(), probe.llm.aclose())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
