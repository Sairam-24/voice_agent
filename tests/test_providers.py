"""Local and cloud components: the app works with no key, with the server's key, and with a user's own key."""

import json

import httpx
import pytest

from app.errors import ProviderError
from app.services.llm import FallbackLLM, LLMUsage, check_gemini_key
from app.services.stt import FallbackSTT, GeminiSTT
from app.services.tts import FallbackTTS
from app.startup import Providers, Services
from tests.conftest import make_settings, sine_wav
from tests.fakes import FakeLLM, FakeSTT, FakeTTS

# Nothing listens here, so every Gemini call fails at once, without touching the network.
UNREACHABLE = "http://127.0.0.1:9/v1beta"


def local_fakes() -> Services:
    return Services(FakeSTT(), FakeLLM(), FakeTTS())


# ----------------------------------------------------------------------------- choosing components


def test_without_any_key_everything_is_local():
    local = local_fakes()
    providers = Providers(make_settings(), local)
    assert providers.default is local and not providers.server_key
    assert providers.for_key(None) is local


def test_the_servers_key_makes_gemini_the_default_with_the_local_models_as_fallbacks():
    local = local_fakes()
    providers = Providers(make_settings(gemini_api_key="server-key"), local)
    assert providers.server_key and providers.default.cloud == {"stt", "llm", "tts"}
    assert isinstance(providers.default.stt, FallbackSTT)
    assert isinstance(providers.default.llm, FallbackLLM)
    assert isinstance(providers.default.tts, FallbackTTS)


def test_a_users_key_gets_its_own_cached_set_and_local_stages_stay_local():
    local = local_fakes()
    providers = Providers(make_settings(stt_provider="whisper"), local)
    mine = providers.for_key("user-key-1")
    assert mine is providers.for_key("user-key-1"), "built once per key"
    assert mine is not providers.for_key("user-key-2")
    assert mine.stt is local.stt, "STT_PROVIDER=whisper keeps speech recognition local"
    assert isinstance(mine.llm, FallbackLLM) and mine.cloud == {"llm", "tts"}
    assert providers.default is local, "other users are unaffected"


def test_with_every_stage_local_a_key_changes_nothing():
    local = local_fakes()
    providers = Providers(make_settings(stt_provider="whisper", llm_provider="ollama", tts_provider="local"), local)
    assert providers.for_key("user-key") is local


async def test_closing_closes_each_shared_local_model_once_and_never_through_a_cloud_set():
    local = local_fakes()
    providers = Providers(make_settings(gemini_api_key="server-key"), local)
    providers.for_key("user-key")
    await providers.aclose()
    assert local.stt.closed and local.llm.closed and local.tts.closed


async def test_shared_fallbacks_are_not_loaded_again_by_a_cloud_set():
    local_stt = FakeSTT()
    await FallbackSTT(GeminiSTT(make_settings(gemini_api_key="k")), local_stt, shared_fallback=True).load()
    assert not local_stt.loaded, "the local recogniser is loaded once, at startup"


# ----------------------------------------------------------------------------- the language-model fallback


async def collect(llm, usage):
    return "".join([delta async for delta in llm.stream([], usage)])


async def test_the_local_model_answers_when_the_cloud_model_refuses():
    usage = LLMUsage()
    llm = FallbackLLM(FakeLLM(error=ProviderError("llm", "quota used up")), FakeLLM(reply="Local answer."))
    assert await collect(llm, usage) == "Local answer."
    assert usage.fallback_reason == "quota used up"


async def test_no_fallback_once_the_cloud_model_has_answered():
    usage = LLMUsage()
    llm = FallbackLLM(FakeLLM(reply="Cloud answer."), FakeLLM(reply="Local answer."))
    assert await collect(llm, usage) == "Cloud answer." and usage.fallback_reason is None


async def test_when_both_fail_the_error_explains_both():
    llm = FallbackLLM(FakeLLM(error=ProviderError("llm", "bad key")), FakeLLM(error=ProviderError("llm", "Ollama is not running")))
    with pytest.raises(ProviderError, match="bad key.*Ollama is not running"):
        await collect(llm, LLMUsage())


# ----------------------------------------------------------------------------- checking a key


async def test_a_key_is_checked_by_listing_models_with_the_key_in_a_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"] = str(request.url), request.headers["x-goog-api-key"]
        if seen["key"] == "good":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(400, json={"error": {"message": "API key not valid. Please pass a valid API key."}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await check_gemini_key(make_settings(), "good", client) == (True, "The key works.")
    valid, message = await check_gemini_key(make_settings(), "bad", client)
    assert not valid and "API key not valid" in message
    assert seen["url"].endswith("/models?pageSize=1") and "bad" not in seen["url"]


def test_the_key_check_endpoint_never_echoes_the_key(make_client, monkeypatch):
    async def fake_check(settings, key, client=None):
        return key == "good-key", "The key works." if key == "good-key" else "Google did not accept this key (400)."

    monkeypatch.setattr("app.main.check_gemini_key", fake_check)
    client = make_client()
    assert client.post("/api/gemini-key/check", json={"key": " good-key "}).json() == {"valid": True, "message": "The key works."}
    refused = client.post("/api/gemini-key/check", json={"key": "wrong-key"})
    assert refused.json()["valid"] is False and "wrong-key" not in refused.text
    assert client.post("/api/gemini-key/check", json={"key": ""}).status_code == 422


def test_info_says_whether_the_server_has_a_key_without_revealing_it(make_client):
    info = make_client(gemini_api_key="server-secret").get("/api/info")
    assert info.json()["cloud"] == {"server_key": True, "stages": ["llm", "stt", "tts"]}
    assert "server-secret" not in info.text
    assert make_client().get("/api/info").json()["cloud"]["server_key"] is False


# ----------------------------------------------------------------------------- a user's key, end to end


def test_a_turn_with_a_users_key_uses_gemini_and_falls_back_locally_when_it_fails(make_client):
    """The key reaches the Gemini adapters (which fail here: nothing listens at UNREACHABLE), and each
    stage falls back to the local model, so the turn still succeeds and says why."""
    client = make_client(gemini_base_url=UNREACHABLE)
    response = client.post(
        "/api/turn", files={"audio": ("speech.wav", sine_wav(), "audio/wav")}, headers={"X-Gemini-Key": "user-key"}
    )
    assert response.status_code == 200, response.text
    metrics = response.json()["metrics"]
    assert metrics["stt_fallback"] and metrics["llm_fallback"] and metrics["tts_fallback"]
    assert "user-key" not in response.text

    plain = client.post("/api/turn", files={"audio": ("speech.wav", sine_wav(), "audio/wav")}).json()["metrics"]
    assert plain["stt_fallback"] is None and plain["llm_fallback"] is None, "without a key: local, no fallback"


def test_websocket_configure_switches_this_connection_to_the_users_key(make_client):
    client = make_client(gemini_base_url=UNREACHABLE)
    with client.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "configure", "gemini_key": "user-key"})
        assert ws.receive_json() == {"type": "configured", "cloud": ["llm", "stt", "tts"]}
        ws.send_json({"type": "start"})
        assert ws.receive_json()["state"] == "listening"
        ws.send_bytes(b"\x00\x10" * 16000)
        ws.send_json({"type": "stop"})
        while (message := ws.receive_json())["type"] != "done":
            if message["type"] == "audio":
                ws.receive_bytes()
        assert message["metrics"]["llm_fallback"], "the user's key was used (and failed over to the local model)"
        assert ws.receive_json() == {"type": "status", "state": "idle"}

        ws.send_json({"type": "configure", "gemini_key": None})
        assert ws.receive_json() == {"type": "configured", "cloud": []}
        ws.send_json({"type": "configure", "gemini_key": 42})
        assert ws.receive_json()["code"] == "invalid_request"
    assert json.dumps(message).count("user-key") == 0
