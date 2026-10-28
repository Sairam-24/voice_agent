"""Language-model adapters.

Two providers behind one small interface: Ollama (local) and Gemini (hosted API).
Both stream plain-text deltas, so the pipeline can start speaking before the
model has finished. Neither adapter has a retry: a voice turn that is already late
is better failed fast and shown to the user than silently retried. FallbackLLM asks
the local model instead when Gemini refuses before answering (quota, a bad key).
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings
from app.errors import ProviderError

log = logging.getLogger("voice_agent.llm")


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMUsage:
    """Filled in by stream() when the provider reports token counts."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    fallback_reason: str | None = None  # set by FallbackLLM when the fallback model answered


class LanguageModel(Protocol):
    description: str
    info: dict

    async def check(self) -> tuple[bool, str]:
        """Cheap reachability/configuration check used by /health. Returns (ok, detail)."""
        ...

    async def warm_up(self) -> None: ...

    def stream(self, messages: list[ChatMessage], usage: LLMUsage) -> AsyncIterator[str]:
        """Yield text deltas; fill `usage` before finishing."""
        ...

    async def aclose(self) -> None: ...


def provider_error_message(response_body: bytes) -> str:
    """Pull a human-readable message out of a provider's JSON error body."""
    try:
        data = json.loads(response_body)
    except ValueError:
        return response_body[:200].decode("utf-8", "replace")
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)


class OllamaLLM:
    """Ollama's native /api/chat endpoint, which streams newline-delimited JSON."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._keep_alive = settings.ollama_keep_alive
        self._num_thread = settings.ollama_num_thread
        self._max_tokens = settings.llm_max_output_tokens
        self._temperature = settings.llm_temperature
        self._reachable_before = False
        # The pipeline enforces the real deadlines; this read timeout is only a backstop.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=3.0, read=settings.llm_timeout_s)
        )
        self.description = f"ollama {self._model} @ {self._base}"
        self.info = {
            "provider": "ollama",
            "model": self._model,
            "max_output_tokens": self._max_tokens,
            "temperature": self._temperature,
            "num_thread": self._num_thread,
        }

    async def check(self) -> tuple[bool, str]:
        try:
            response = await self._client.get(f"{self._base}/api/tags", timeout=2.0)
            response.raise_for_status()
        except httpx.TimeoutException:
            # A small CPU that is busy generating a reply can be slow to answer even a cheap request.
            # That is "busy", not "down": reporting it as unreachable would warn users mid-conversation.
            if self._reachable_before:
                return True, f"{self.description} (busy, slow to answer)"
            return False, f"Ollama at {self._base} did not answer in time."
        except httpx.HTTPError:
            return False, f"Ollama is not reachable at {self._base}. Start it with `ollama serve`."
        self._reachable_before = True
        installed = {m.get("name"): m for m in response.json().get("models", [])}
        wanted = self._model if ":" in self._model else f"{self._model}:latest"
        if wanted not in installed:
            return False, f"Model '{self._model}' is not installed. Run: ollama pull {self._model}"
        details = installed[wanted].get("details", {})
        self.info.update(
            digest=installed[wanted].get("digest", "")[:12],
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
        )
        return True, self.description

    async def warm_up(self) -> None:
        """Load the model into memory so the first real turn does not pay for it."""
        usage = LLMUsage()
        async for _ in self._stream([ChatMessage("user", "Hi")], usage, max_tokens=1):
            pass

    def stream(self, messages: list[ChatMessage], usage: LLMUsage) -> AsyncIterator[str]:
        return self._stream(messages, usage, self._max_tokens)

    async def _stream(self, messages: list[ChatMessage], usage: LLMUsage, max_tokens: int) -> AsyncIterator[str]:
        options = {"num_predict": max_tokens, "temperature": self._temperature}
        if self._num_thread:
            options["num_thread"] = self._num_thread
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": options,
        }
        finished = False
        try:
            async with self._client.stream("POST", f"{self._base}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    detail = provider_error_message(await response.aread())
                    raise ProviderError("llm", f"Ollama returned {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise ProviderError("llm", f"Ollama error: {event['error']}")
                    delta = event.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if event.get("done"):
                        usage.prompt_tokens = event.get("prompt_eval_count")
                        usage.output_tokens = event.get("eval_count")
                        finished = True
        except httpx.ConnectError as exc:
            raise ProviderError("llm", f"Cannot connect to Ollama at {self._base}. Is it running?") from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("llm", "Ollama stopped responding.") from exc
        except (httpx.HTTPError, ValueError) as exc:  # ValueError: a line that is not JSON
            raise ProviderError("llm", f"Ollama stream failed: {exc}") from exc
        if not finished:
            raise ProviderError("llm", "Ollama closed the stream before finishing.")

    async def aclose(self) -> None:
        await self._client.aclose()


class GeminiLLM:
    """Gemini REST API, streamed as server-sent events (`:streamGenerateContent?alt=sse`)."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        key = settings.gemini_api_key
        self._key = key.get_secret_value() if key else ""
        self._model = settings.gemini_model
        self._url = f"{settings.gemini_base_url.rstrip('/')}/models/{self._model}:streamGenerateContent"
        self._max_tokens = settings.llm_max_output_tokens
        self._temperature = settings.llm_temperature
        self._thinking_level = settings.gemini_thinking_level
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0, read=settings.llm_timeout_s)
        )
        self.description = f"gemini {self._model}"
        self.info = {
            "provider": "gemini",
            "model": self._model,
            "max_output_tokens": self._max_tokens,
            "temperature": self._temperature,
            "thinking_level": self._thinking_level,
        }

    async def check(self) -> tuple[bool, str]:
        # No network call: health is polled often and every call would use API quota.
        if not self._key:
            return False, "GEMINI_API_KEY is not set."
        return True, self.description

    async def warm_up(self) -> None:
        return  # nothing to load locally, and a warm-up call would cost money

    async def stream(self, messages: list[ChatMessage], usage: LLMUsage) -> AsyncIterator[str]:
        if not self._key:
            raise ProviderError("llm", "GEMINI_API_KEY is not set.")

        system = "\n".join(m.content for m in messages if m.role == "system")
        generation_config: dict = {"maxOutputTokens": self._max_tokens, "temperature": self._temperature}
        if self._thinking_level:
            generation_config["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        payload: dict = {
            "contents": [
                {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
                for m in messages
                if m.role != "system"
            ],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        # The key goes in a header, not the query string, so it cannot end up in access logs.
        headers = {"x-goog-api-key": self._key}
        produced_text = False
        finish_reason = None
        block_reason = None
        try:
            async with self._client.stream(
                "POST", self._url, params={"alt": "sse"}, headers=headers, json=payload
            ) as response:
                if response.status_code != 200:
                    detail = provider_error_message(await response.aread())
                    raise ProviderError("llm", f"Gemini returned {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    block_reason = event.get("promptFeedback", {}).get("blockReason", block_reason)
                    meta = event.get("usageMetadata")
                    if meta:
                        usage.prompt_tokens = meta.get("promptTokenCount", usage.prompt_tokens)
                        usage.output_tokens = meta.get("candidatesTokenCount", usage.output_tokens)
                    for candidate in event.get("candidates", []):
                        finish_reason = candidate.get("finishReason", finish_reason)
                        for part in candidate.get("content", {}).get("parts", []):
                            if part.get("thought"):  # reasoning summaries are not meant to be spoken
                                continue
                            text = part.get("text", "")
                            if text:
                                produced_text = True
                                yield text
        except httpx.TimeoutException as exc:
            raise ProviderError("llm", "Gemini stopped responding.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("llm", f"Gemini stream failed: {exc}") from exc

        if not produced_text:
            if block_reason or finish_reason == "SAFETY":
                raise ProviderError("llm", f"Gemini blocked the request ({block_reason or finish_reason}).")
            if finish_reason == "MAX_TOKENS":
                # Thinking tokens count toward maxOutputTokens, so a small cap can leave no room for text.
                raise ProviderError(
                    "llm",
                    "Gemini hit the output token limit before producing text. "
                    "Raise LLM_MAX_OUTPUT_TOKENS or lower GEMINI_THINKING_LEVEL.",
                )
            raise ProviderError("llm", f"Gemini returned no text (finishReason={finish_reason}).")

    async def aclose(self) -> None:
        await self._client.aclose()


class FallbackLLM:
    """Answer with `primary`; if it fails before producing any text, answer with `fallback` and record
    why in `usage.fallback_reason`. If the fallback fails too, the primary's error is reported.

    With `shared_fallback` the fallback belongs to someone else (the local model every Gemini key
    shares): it is neither warmed up nor closed here.
    """

    def __init__(self, primary: LanguageModel, fallback: LanguageModel, shared_fallback: bool = False):
        self._primary = primary
        self._fallback = fallback
        self._shared = shared_fallback
        self.description = f"{primary.description}; fallback {fallback.description}"
        self.info = {**primary.info, "fallback": fallback.info}

    async def check(self) -> tuple[bool, str]:
        return await self._primary.check()

    async def warm_up(self) -> None:
        await self._primary.warm_up()

    async def stream(self, messages: list[ChatMessage], usage: LLMUsage) -> AsyncIterator[str]:
        produced = False
        try:
            async with aclosing(self._primary.stream(messages, usage)) as deltas:
                async for delta in deltas:
                    produced = True
                    yield delta
            return
        except ProviderError as exc:
            if produced:
                raise
            primary_error = exc
        log.warning("primary language model failed, using the fallback: %s", primary_error)
        try:
            async with aclosing(self._fallback.stream(messages, usage)) as deltas:
                async for delta in deltas:
                    usage.fallback_reason = str(primary_error)
                    yield delta
        except ProviderError as exc:
            if usage.fallback_reason:  # the fallback had started answering: its own failure is the news
                raise
            raise ProviderError("llm", f"{primary_error} (the local model could not answer either: {exc})") from exc

    async def aclose(self) -> None:
        await self._primary.aclose()
        if not self._shared:
            await self._fallback.aclose()


async def check_gemini_key(settings: Settings, key: str, client: httpx.AsyncClient | None = None) -> tuple[bool, str]:
    """Ask Google whether `key` works. Listing one model uses no generation quota."""
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
    try:
        response = await client.get(
            f"{settings.gemini_base_url.rstrip('/')}/models", params={"pageSize": 1}, headers={"x-goog-api-key": key}
        )
    except httpx.HTTPError as exc:
        return False, f"Could not reach Google to check the key: {exc}"
    finally:
        if own_client:
            await client.aclose()
    if response.status_code == 200:
        return True, "The key works."
    if response.status_code == 429:
        return True, "The key works, but it is rate-limited right now."
    return False, f"Google did not accept this key ({response.status_code}): {provider_error_message(response.content)}"
