"""Generate the fixed spoken prompts used by the benchmark.

    python scripts/make_test_audio.py

Writes benchmarks/inputs/{short,medium,long}.wav using the local Piper voice. Piper is not
bit-for-bit deterministic, so the files are generated once and then reused: the benchmark
records each file's SHA-256 so results can be tied to the exact inputs. The files are not
committed (see .gitignore) because they are generated with a third-party voice model.
"""

import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.services.audio import pcm16_to_wav  # noqa: E402
from app.services.tts import PiperTTS  # noqa: E402

PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Can you explain in a couple of sentences how a rainbow forms?",
    "long": (
        "I am planning a three day trip to Rome next month. "
        "What are the most important things I should see, and what should I avoid?"
    ),
}
OUTPUT_DIR = Path("benchmarks/inputs")


async def synthesize_prompts() -> dict[str, tuple[bytes, int]]:
    tts = PiperTTS(get_settings())
    await tts.load()
    try:
        return {
            name: (b"".join([c.pcm16 async for c in tts.synthesize(text, threading.Event())]), tts.sample_rate)
            for name, text in PROMPTS.items()
        }
    finally:
        await tts.aclose()


def main() -> None:
    audio = asyncio.run(synthesize_prompts())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (pcm, sample_rate) in audio.items():
        path = OUTPUT_DIR / f"{name}.wav"
        path.write_bytes(pcm16_to_wav(pcm, sample_rate))
        print(f"{path}: {len(pcm) / 2 / sample_rate:.1f} s  {PROMPTS[name]!r}")


if __name__ == "__main__":
    main()
