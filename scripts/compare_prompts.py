"""Compare how long Ollama's replies are under different system prompts (needs Ollama running).

    python scripts/compare_prompts.py

Edit PROMPTS and QUESTIONS below to try your own. Each cell is 5 samples, so read the spread,
not just the median. The result that led to the current default prompt is saved in
benchmarks/results/prompt-length-2026-10-26.txt.
"""

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.services.llm import ChatMessage, LLMUsage, OllamaLLM  # noqa: E402
from app.services.speech_text import clean_for_speech  # noqa: E402

SAMPLES = 5
QUESTIONS = [
    "What is the capital of France?",
    "Can you explain in a couple of sentences how a rainbow forms?",
    "I am planning a three day trip to Rome next month. What are the most important things I should see, and what should I avoid?",
]
PROMPTS = {
    "A original": (
        "You are a voice assistant. Your reply will be spoken aloud, so answer in one to "
        "three short sentences of plain, conversational English. Never use markdown, "
        "bullet points, emoji or code blocks. If you do not know something, say so briefly."
    ),
    "B strict (current default)": (
        "You are a voice assistant. Your reply is spoken aloud. Reply in at most two short sentences, "
        "under 40 words in total. Use plain spoken English with no lists, markdown, or emoji. "
        "Never give more than two sentences, even if the question is broad."
    ),
}


async def main() -> None:
    llm = OllamaLLM(Settings(llm_max_output_tokens=160))
    try:
        for name, prompt in PROMPTS.items():
            print(f"=== prompt {name} ===")
            for question in QUESTIONS:
                words = []
                for _ in range(SAMPLES):
                    messages = [ChatMessage("system", prompt), ChatMessage("user", question)]
                    text = "".join([delta async for delta in llm.stream(messages, LLMUsage())])
                    words.append(len(clean_for_speech(text).split()))
                print(f"  {question[:42]:<44} words per reply: {sorted(words)}  median {statistics.median(words)}")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
