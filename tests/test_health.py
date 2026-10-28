import time

from app.startup import Services
from tests.fakes import FakeLLM, FakeSTT, FakeTTS


def test_health_ok_when_everything_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["components"]) == {"stt", "tts", "llm"}
    assert all(c["state"] == "ready" for c in body["components"].values())
    assert body["version"]
    assert body["uptime_s"] >= 0
    assert client.get("/ready").status_code == 200


def test_health_is_200_but_starting_while_a_model_loads(make_client):
    client = make_client(Services(FakeSTT(load_delay=1.0), FakeLLM(), FakeTTS()), wait=False)
    time.sleep(0.1)
    body = client.get("/health").json()
    assert body["status"] == "starting"
    assert body["components"]["stt"]["state"] == "loading"
    assert client.get("/ready").status_code == 503


def test_turn_is_refused_with_503_while_models_load(make_client):
    client = make_client(Services(FakeSTT(load_delay=1.0), FakeLLM(), FakeTTS()), wait=False)
    time.sleep(0.1)
    response = client.post("/api/turn", files={"audio": ("a.wav", b"x", "audio/wav")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "models_not_ready"


def test_health_reports_a_model_that_failed_to_load(make_client):
    stt = FakeSTT(load_error=RuntimeError("model file is corrupt"))
    client = make_client(Services(stt, FakeLLM(), FakeTTS()), wait=False)
    time.sleep(0.2)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["stt"]["state"] == "error"
    assert "model file is corrupt" in body["components"]["stt"]["detail"]
    assert client.get("/ready").status_code == 503


def test_health_reports_an_unreachable_llm(make_client):
    client = make_client(Services(FakeSTT(), FakeLLM(healthy=False), FakeTTS()), wait=False)
    time.sleep(0.2)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["llm"] == {"state": "error", "detail": "fake llm is down"}
    assert client.get("/ready").status_code == 503


def test_request_id_is_generated_and_returned(client):
    assert len(client.get("/health").headers["x-request-id"]) == 12


def test_safe_client_request_id_is_reused(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})
    assert response.headers["x-request-id"] == "trace-abc.123"


def test_unsafe_client_request_id_is_replaced(client):
    response = client.get("/health", headers={"X-Request-ID": "bad id with spaces\t"})
    assert len(response.headers["x-request-id"]) == 12


def test_unknown_route_is_404_in_the_standard_error_shape(client):
    response = client.get("/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_404"
    assert error["request_id"] == response.headers["x-request-id"]


def test_info_reports_versions_and_settings_but_never_secrets(make_client):
    client = make_client(gemini_api_key="super-secret-key")
    response = client.get("/api/info")
    body = response.json()
    # With the server's key the default model is Gemini, with the local one as its fallback.
    assert body["components"]["llm"]["provider"] == "gemini"
    assert body["components"]["llm"]["fallback"] == {"provider": "fake-llm"}
    assert body["packages"]["faster-whisper"] and body["python"]
    assert body["settings"]["max_inflight_turns"] == 4
    assert "super-secret-key" not in response.text
    assert "gemini_api_key" not in body["settings"]


def test_metrics_endpoint_starts_empty(client):
    body = client.get("/api/metrics").json()
    assert body["recent_turns"] == []
    assert body["summary"]["window_size"] == 0
    assert body["load"] == {"inflight_turns": 0, "max_inflight_turns": 4}
