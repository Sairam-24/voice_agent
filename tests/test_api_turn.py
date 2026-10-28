import base64
import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.errors import ProviderError
from app.startup import Services
from tests.conftest import sine_wav
from tests.fakes import FakeLLM, FakeSTT, FakeTTS


def post_audio(client, data: bytes | None = None, history=None, filename="speech.wav"):
    files = {"audio": (filename, sine_wav() if data is None else data, "audio/wav")}
    form = {"history": json.dumps(history)} if history is not None else {}
    return client.post("/api/turn", files=files, data=form)


def error_code(response) -> str:
    return response.json()["error"]["code"]


# ------------------------------------------------------------------------ success

def test_successful_turn_returns_transcript_reply_playable_wav_and_metrics(client):
    response = post_audio(client)
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["transcript"] == "what is the capital of france"
    assert body["reply"].startswith("The capital of France is Paris.")
    assert body["request_id"] == response.headers["x-request-id"]

    audio = body["audio"]
    assert audio["format"] == "wav" and audio["sample_rate"] == 22050
    wav_bytes = base64.b64decode(audio["data_base64"])
    assert wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav_bytes)) as wav:  # the stdlib parser accepting it = it is playable
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 22050)
        assert wav.getnframes() / wav.getframerate() == pytest.approx(audio["duration_s"], abs=0.001)

    metrics = body["metrics"]
    assert metrics["transport"] == "http" and metrics["status"] == "ok"
    for stage in ("upload_ms", "decode_ms", "stt_ms", "llm_ttft_ms", "llm_ms", "tts_first_chunk_ms", "tts_ms", "total_ms"):
        assert metrics[stage] is not None, stage


def test_finished_turns_show_up_in_the_metrics_endpoint(client):
    post_audio(client)
    post_audio(client)
    body = client.get("/api/metrics").json()
    assert body["summary"]["turns_since_start"] == {"ok": 2}
    assert body["summary"]["stages_ms"]["stt_ms"]["n"] == 2
    assert len(body["recent_turns"]) == 2


def test_conversation_history_reaches_the_llm(make_client):
    llm = FakeLLM()
    client = make_client(Services(FakeSTT(), llm, FakeTTS()))
    history = [{"role": "user", "content": "my name is Sam"}, {"role": "assistant", "content": "Nice to meet you, Sam."}]
    assert post_audio(client, history=history).status_code == 200
    assert [m.content for m in llm.last_messages][1:3] == ["my name is Sam", "Nice to meet you, Sam."]


def test_silence_returns_no_speech_with_no_audio(make_client):
    client = make_client(Services(FakeSTT(text=""), FakeLLM(), FakeTTS()))
    body = post_audio(client).json()
    assert body["status"] == "no_speech"
    assert body["audio"] is None and body["reply"] == ""


# ------------------------------------------------------------------------ invalid input

def test_garbage_bytes_are_415_unsupported_audio(client):
    response = post_audio(client, b"definitely not audio" * 100)
    assert response.status_code == 415
    assert error_code(response) == "unsupported_audio"
    assert client.get("/api/metrics").json()["summary"]["turns_since_start"] == {"error": 1}


def test_empty_file_is_422_empty_audio(client):
    response = post_audio(client, b"")
    assert response.status_code == 422
    assert error_code(response) == "empty_audio"


def test_audio_over_the_duration_limit_is_413(make_client):
    client = make_client(max_audio_seconds=0.5)
    response = post_audio(client, sine_wav(2.0))
    assert response.status_code == 413
    assert error_code(response) == "audio_too_long"


def test_upload_over_the_size_limit_is_413(make_client):
    client = make_client(max_upload_bytes=1000)
    response = post_audio(client, sine_wav(1.0))
    assert response.status_code == 413
    assert error_code(response) == "upload_too_large"


def test_oversized_request_is_refused_from_content_length_before_the_body_is_read(make_client):
    stt = FakeSTT()
    client = make_client(Services(stt, FakeLLM(), FakeTTS()), max_upload_bytes=1000)
    response = post_audio(client, b"\x00" * 200_000)  # far past 1000 bytes plus multipart overhead
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "upload_too_large"
    assert error["message"].startswith("Request body is larger")  # the middleware's message, not the route's
    assert error["request_id"] == response.headers["x-request-id"]
    assert stt.probe.total == 0


def test_missing_audio_field_is_422_invalid_request(client):
    response = client.post("/api/turn", data={"history": "[]"})
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


@pytest.mark.parametrize("bad_history", ["not json", '{"role": "user"}', '[{"role": "system", "content": "x"}]'])
def test_malformed_history_is_422(client, bad_history):
    response = client.post("/api/turn", files={"audio": ("a.wav", sine_wav(), "audio/wav")}, data={"history": bad_history})
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_get_on_the_turn_endpoint_is_405(client):
    assert client.get("/api/turn").status_code == 405


# ------------------------------------------------------------------------ provider failures and timeouts

@pytest.mark.parametrize(
    "services,status,code",
    [
        (Services(FakeSTT(error=ProviderError("stt", "x")), FakeLLM(), FakeTTS()), 502, "stt_error"),
        (Services(FakeSTT(), FakeLLM(error=ProviderError("llm", "Ollama down")), FakeTTS()), 502, "llm_error"),
    ],
)
def test_provider_failures_map_to_502_with_a_specific_code(make_client, services, status, code):
    response = post_audio(make_client(services))
    assert response.status_code == status
    assert error_code(response) == code


def test_a_failed_voice_still_returns_the_reply_as_text(make_client):
    response = post_audio(make_client(Services(FakeSTT(), FakeLLM(), FakeTTS(error=ProviderError("tts", "quota used up")))))
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] and body["audio"] is None
    assert body["metrics"]["tts_fallback"] == "Text only: quota used up"
    assert body["language"] == "en"


def test_slow_llm_maps_to_504_llm_timeout(make_client):
    client = make_client(Services(FakeSTT(), FakeLLM(first_token_delay=2), FakeTTS()), llm_first_token_timeout_s=0.1)
    response = post_audio(client)
    assert response.status_code == 504
    assert error_code(response) == "llm_timeout"
    assert client.get("/api/metrics").json()["summary"]["turns_since_start"] == {"timeout": 1}


def test_error_responses_carry_the_request_id(client):
    response = post_audio(client, b"junk" * 50)
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


# ------------------------------------------------------------------------ real-server behaviour

def test_concurrent_requests_beyond_the_limit_get_429_and_the_rest_succeed(live_server):
    base = live_server(Services(FakeSTT(delay=0.6), FakeLLM(), FakeTTS()), max_inflight_turns=1)

    def one_request(_):
        with httpx.Client(timeout=10) as http:
            return http.post(f"{base}/api/turn", files={"audio": ("a.wav", sine_wav(), "audio/wav")})

    with ThreadPoolExecutor(3) as pool:
        responses = list(pool.map(one_request, range(3)))

    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 429, 429]
    busy = next(r for r in responses if r.status_code == 429)
    assert busy.json()["error"]["code"] == "server_busy"


def test_client_disconnect_cancels_the_running_turn(live_server):
    llm = FakeLLM(first_token_delay=30)  # would hang for 30 s if nothing cancelled it
    base = live_server(Services(FakeSTT(), llm, FakeTTS()))

    # Give up after 0.5 s: httpx closes the connection, which is what a closed browser tab does.
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{base}/api/turn", files={"audio": ("a.wav", sine_wav(), "audio/wav")}, timeout=0.5)

    deadline = time.monotonic() + 5
    turns = []
    while time.monotonic() < deadline:
        turns = httpx.get(f"{base}/api/metrics").json()["recent_turns"]
        if turns:
            break
        time.sleep(0.05)

    assert turns, "the server never recorded the abandoned turn"
    assert turns[0]["status"] == "cancelled" and turns[0]["error_code"] == "cancelled"
    assert llm.cancelled, "the LLM stream was not cancelled"
    assert httpx.get(f"{base}/api/metrics").json()["load"]["inflight_turns"] == 0
