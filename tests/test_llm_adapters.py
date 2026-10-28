"""LLM adapter contract tests against mocked HTTP transports (no network, no model)."""

import json

import httpx
import pytest

from app.errors import ProviderError
from app.services.llm import ChatMessage, GeminiLLM, LLMUsage, OllamaLLM
from tests.conftest import make_settings

MESSAGES = [
    ChatMessage("system", "Be brief."),
    ChatMessage("user", "Hi"),
    ChatMessage("assistant", "Hello!"),
    ChatMessage("user", "Capital of France?"),
]


def ndjson(*events: dict) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


def sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in events)


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def collect(llm, messages=MESSAGES) -> tuple[str, LLMUsage]:
    usage = LLMUsage()
    parts = [delta async for delta in llm.stream(messages, usage)]
    return "".join(parts), usage


# ----------------------------------------------------------------------------- Ollama

OLLAMA_OK = ndjson(
    {"message": {"role": "assistant", "content": "Paris"}, "done": False},
    {"message": {"role": "assistant", "content": " it is."}, "done": False},
    {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
     "prompt_eval_count": 31, "eval_count": 4},
)


async def test_ollama_streams_deltas_and_reports_usage_and_sends_expected_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, content=OLLAMA_OK)

    llm = OllamaLLM(make_settings(ollama_model="llama3.2:1b", llm_max_output_tokens=77), client_for(handler))
    text, usage = await collect(llm)

    assert text == "Paris it is."
    assert (usage.prompt_tokens, usage.output_tokens) == (31, 4)
    assert seen["url"] == "http://localhost:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "llama3.2:1b"
    assert body["stream"] is True
    assert body["options"]["num_predict"] == 77
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize("num_thread,expected", [(None, None), (1, 1)])
async def test_ollama_num_thread_is_sent_only_when_configured(num_thread, expected):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=OLLAMA_OK)

    await collect(OllamaLLM(make_settings(ollama_num_thread=num_thread), client_for(handler)))
    assert seen["options"].get("num_thread") == expected
    assert ("num_thread" in seen["options"]) == (expected is not None)


async def test_ollama_error_status_becomes_provider_error_with_the_servers_message():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    with pytest.raises(ProviderError, match="model 'nope' not found") as info:
        await collect(OllamaLLM(make_settings(), client_for(handler)))
    assert info.value.code == "llm_error"


async def test_ollama_error_event_inside_a_200_stream_is_raised():
    def handler(request):
        return httpx.Response(200, content=ndjson({"error": "out of memory"}))

    with pytest.raises(ProviderError, match="out of memory"):
        await collect(OllamaLLM(make_settings(), client_for(handler)))


async def test_ollama_stream_that_ends_without_done_is_an_error():
    def handler(request):
        return httpx.Response(200, content=ndjson({"message": {"content": "partial"}, "done": False}))

    with pytest.raises(ProviderError, match="before finishing"):
        await collect(OllamaLLM(make_settings(), client_for(handler)))


async def test_ollama_connection_failure_gives_an_actionable_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="Cannot connect to Ollama"):
        await collect(OllamaLLM(make_settings(), client_for(handler)))


async def test_ollama_garbage_line_is_a_provider_error_not_a_crash():
    def handler(request):
        return httpx.Response(200, content=b"<html>proxy error</html>\n")

    with pytest.raises(ProviderError, match="stream failed"):
        await collect(OllamaLLM(make_settings(), client_for(handler)))


@pytest.mark.parametrize(
    "installed,model,expected_ok",
    [
        (["llama3.2:1b"], "llama3.2:1b", True),
        (["llama3.2:latest"], "llama3.2", True),  # bare name means :latest
        (["qwen2.5:0.5b"], "llama3.2:1b", False),
        ([], "llama3.2:1b", False),
    ],
)
async def test_ollama_check_looks_for_the_configured_model(installed, model, expected_ok):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": n} for n in installed]})

    ok, detail = await OllamaLLM(make_settings(ollama_model=model), client_for(handler)).check()
    assert ok is expected_ok
    if not expected_ok:
        assert f"ollama pull {model}" in detail


async def test_ollama_check_reports_unreachable_server():
    def handler(request):
        raise httpx.ConnectError("refused")

    ok, detail = await OllamaLLM(make_settings(), client_for(handler)).check()
    assert not ok and "ollama serve" in detail


async def test_ollama_that_was_reachable_but_is_slow_to_answer_is_busy_not_down():
    answers = iter([True, False])

    def handler(request):
        if next(answers):
            return httpx.Response(200, json={"models": [{"name": "llama3.2:1b"}]})
        raise httpx.ReadTimeout("busy generating")

    llm = OllamaLLM(make_settings(ollama_model="llama3.2:1b"), client_for(handler))
    assert (await llm.check())[0] is True
    ok, detail = await llm.check()
    assert ok and "busy" in detail


async def test_ollama_that_never_answered_is_not_reported_ready_on_a_timeout():
    def handler(request):
        raise httpx.ReadTimeout("no answer")

    ok, detail = await OllamaLLM(make_settings(), client_for(handler)).check()
    assert not ok and "did not answer in time" in detail


async def test_ollama_warm_up_asks_for_a_single_token():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=OLLAMA_OK)

    await OllamaLLM(make_settings(), client_for(handler)).warm_up()
    assert seen["options"]["num_predict"] == 1


# ----------------------------------------------------------------------------- Gemini

def gemini_settings(**overrides):
    return make_settings(llm_provider="gemini", gemini_api_key="test-key-123", **overrides)


GEMINI_OK = sse(
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "Paris"}]}}]},
    {"candidates": [{"content": {"role": "model", "parts": [{"text": " is it."}]}, "finishReason": "STOP"}],
     "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5}},
)


async def test_gemini_streams_text_and_builds_the_expected_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = request.url, request.headers, json.loads(request.content)
        return httpx.Response(200, content=GEMINI_OK)

    llm = GeminiLLM(gemini_settings(gemini_model="gemini-test", llm_max_output_tokens=99), client_for(handler))
    text, usage = await collect(llm)

    assert text == "Paris is it."
    assert (usage.prompt_tokens, usage.output_tokens) == (20, 5)
    assert seen["url"].path.endswith("/models/gemini-test:streamGenerateContent")
    assert seen["url"].params["alt"] == "sse"
    # The key must travel in a header, never in the URL where logs would capture it.
    assert seen["headers"]["x-goog-api-key"] == "test-key-123"
    assert "test-key-123" not in str(seen["url"])
    body = seen["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]  # assistant -> model
    assert body["generationConfig"]["maxOutputTokens"] == 99
    assert "thinkingConfig" not in body["generationConfig"]


async def test_gemini_thinking_level_is_sent_only_when_configured():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=GEMINI_OK)

    await collect(GeminiLLM(gemini_settings(gemini_thinking_level="minimal"), client_for(handler)))
    assert seen["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


async def test_gemini_skips_thought_parts():
    events = sse({"candidates": [{"content": {"parts": [
        {"text": "internal reasoning", "thought": True}, {"text": "Visible answer."}]}, "finishReason": "STOP"}]})
    llm = GeminiLLM(gemini_settings(), client_for(lambda r: httpx.Response(200, content=events)))
    text, _ = await collect(llm)
    assert text == "Visible answer."


async def test_gemini_without_a_key_fails_clearly_and_makes_no_request():
    def handler(request):
        raise AssertionError("must not call the API without a key")

    llm = GeminiLLM(make_settings(llm_provider="gemini"), client_for(handler))
    assert await llm.check() == (False, "GEMINI_API_KEY is not set.")
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        await collect(llm)


async def test_gemini_api_error_message_is_surfaced_without_leaking_the_key():
    def handler(request):
        return httpx.Response(400, json={"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(ProviderError, match="API key not valid") as info:
        await collect(GeminiLLM(gemini_settings(), client_for(handler)))
    assert "test-key-123" not in str(info.value)


async def test_gemini_empty_reply_from_token_limit_explains_the_cause():
    events = sse({"candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}]})
    with pytest.raises(ProviderError, match="LLM_MAX_OUTPUT_TOKENS"):
        await collect(GeminiLLM(gemini_settings(), client_for(lambda r: httpx.Response(200, content=events))))


async def test_gemini_safety_block_is_reported():
    events = sse({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ProviderError, match="blocked"):
        await collect(GeminiLLM(gemini_settings(), client_for(lambda r: httpx.Response(200, content=events))))


async def test_gemini_check_makes_no_network_call():
    def handler(request):
        raise AssertionError("health checks must not spend API quota")

    ok, detail = await GeminiLLM(gemini_settings(), client_for(handler)).check()
    assert ok and "gemini" in detail
