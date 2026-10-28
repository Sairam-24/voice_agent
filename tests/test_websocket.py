"""WebSocket protocol tests against a real uvicorn server (so disconnects and framing are real)."""

import json
import time

import httpx
import numpy as np
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from app.errors import ProviderError
from app.startup import Services
from tests.fakes import FakeLLM, FakeSTT, FakeTTS

FRAME_SECONDS = 0.1
TERMINAL = ("done", "error", "cancelled")


def tone_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2").tobytes()


class Session:
    """Small synchronous client for the voice protocol."""

    def __init__(self, base_url: str, **connect_kwargs):
        self.base = base_url
        # websockets' sync client is meant to be used as a context manager; enter it by hand
        # because a Session outlives a single `with` block.
        self._connection = connect(base_url.replace("http://", "ws://") + "/ws/voice", open_timeout=5, **connect_kwargs)
        self.ws = self._connection.__enter__()
        self.ready = self.next_event()

    def send(self, **message) -> None:
        self.ws.send(json.dumps(message))

    def speak(self, seconds: float = 1.0, realtime: bool = False) -> None:
        pcm = tone_pcm(seconds)
        step = int(16000 * FRAME_SECONDS) * 2
        for i in range(0, len(pcm), step):
            self.ws.send(pcm[i : i + step])
            if realtime:
                time.sleep(FRAME_SECONDS)

    def next_event(self, timeout: float = 10) -> dict:
        message = self.ws.recv(timeout=timeout)
        if isinstance(message, bytes):
            return {"type": "binary", "data": message, "at": time.monotonic()}
        return {**json.loads(message), "at": time.monotonic()}

    def events_until(self, terminal=TERMINAL, timeout: float = 15) -> list[dict]:
        events = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self.next_event(timeout=max(0.1, deadline - time.monotonic()))
            events.append(event)
            if event["type"] in terminal:
                return events
        raise AssertionError(f"no {terminal} event; got {[e['type'] for e in events]}")

    def turn(self, seconds: float = 1.0) -> list[dict]:
        self.send(type="start")
        self.speak(seconds)
        self.send(type="stop")
        return self.events_until()

    def close(self) -> None:
        self._connection.__exit__(None, None, None)


@pytest.fixture
def session_for(live_server):
    sessions = []

    def start(services: Services | None = None, **settings) -> Session:
        session = Session(live_server(services, **settings))
        sessions.append(session)
        return session

    yield start
    for s in sessions:
        s.close()


def types(events):
    return [e["type"] for e in events]


# ------------------------------------------------------------------------ happy path

def test_connect_sends_ready_with_the_input_format(session_for):
    ready = session_for().ready
    assert ready["type"] == "ready"
    assert ready["input"] == {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    assert ready["models_ready"] is True
    assert ready["max_audio_seconds"] == 30


def test_a_full_streaming_turn_follows_the_documented_protocol(session_for):
    events = session_for().turn(1.0)

    # "idle" follows "done", so it is not part of this list
    assert [e["state"] for e in events if e["type"] == "status"] == [
        "listening", "transcribing", "thinking", "speaking",
    ]

    transcript = next(e for e in events if e["type"] == "transcript")
    assert (transcript["text"], transcript["final"]) == ("what is the capital of france", True)

    deltas = "".join(e["text"] for e in events if e["type"] == "reply_delta")
    assert deltas == "The capital of France is Paris. It is a lovely city."

    # every audio header is immediately followed by exactly one binary frame
    headers = [i for i, e in enumerate(events) if e["type"] == "audio"]
    assert len(headers) == 2
    for i in headers:
        assert events[i + 1]["type"] == "binary"
        assert events[i]["sample_rate"] == 22050
        assert len(events[i + 1]["data"]) % 2 == 0 and len(events[i + 1]["data"]) > 0
    assert [events[i]["index"] for i in headers] == [0, 1]

    done = events[-1]
    assert done["type"] == "done" and done["status"] == "ok"
    assert done["metrics"]["transport"] == "ws"
    assert done["metrics"]["first_audio_ms"] <= done["metrics"]["total_ms"]
    assert types(events).index("transcript") < types(events).index("reply_delta") < types(events).index("audio")


def test_results_really_stream_instead_of_arriving_in_one_burst(session_for):
    reply = "First sentence is right here. " + "Then plenty more words follow along. " * 6
    session = session_for(Services(FakeSTT(), FakeLLM(reply=reply, token_delay=0.05), FakeTTS()))
    events = session.turn(0.5)

    first_delta = next(e["at"] for e in events if e["type"] == "reply_delta")
    first_audio = next(e["at"] for e in events if e["type"] == "audio")
    done_at = events[-1]["at"]
    assert first_audio < done_at - 0.5, "audio only arrived at the very end"
    assert first_delta < done_at - 0.5, "text only arrived at the very end"


def test_history_carries_across_turns_until_reset(session_for):
    llm = FakeLLM()
    session = session_for(Services(FakeSTT(), llm, FakeTTS()))
    session.turn()
    session.turn()
    roles = [m.role for m in llm.last_messages]
    assert roles == ["system", "user", "assistant", "user"]

    session.send(type="reset")
    session.turn()
    assert [m.role for m in llm.last_messages] == ["system", "user"]


def test_silence_gives_done_no_speech_and_no_audio(session_for):
    events = session_for(Services(FakeSTT(text=""), FakeLLM(), FakeTTS())).turn()
    assert events[-1]["type"] == "done" and events[-1]["status"] == "no_speech"
    assert "audio" not in types(events) and "reply_delta" not in types(events)


def test_partial_transcripts_arrive_before_the_final_one_when_enabled(session_for):
    session = session_for(Services(FakeSTT(delay=0.05), FakeLLM(), FakeTTS()), partial_transcripts=True, partial_interval_s=0.3)
    assert session.ready["partial_transcripts"] is True
    session.send(type="start")
    session.speak(2.0, realtime=True)  # real-time pacing so the server can see audio accumulate
    session.send(type="stop")
    events = session.events_until()

    flags = [e["final"] for e in events if e["type"] == "transcript"]
    assert flags[-1] is True and False in flags[:-1], f"expected partial then final, got {flags}"


def test_no_partials_by_default(session_for):
    session = session_for()
    session.send(type="start")
    session.speak(2.0, realtime=True)
    session.send(type="stop")
    flags = [e["final"] for e in session.events_until() if e["type"] == "transcript"]
    assert flags == [True]


# ------------------------------------------------------------------------ cancellation and disconnects

def test_cancel_aborts_a_running_turn_and_the_connection_stays_usable(session_for):
    llm = FakeLLM(first_token_delay=30)
    session = session_for(Services(FakeSTT(), llm, FakeTTS()))
    session.send(type="start")
    session.speak(0.5)
    session.send(type="stop")
    while session.next_event()["type"] != "transcript":
        pass
    session.send(type="cancel")
    rest = session.events_until(terminal=("cancelled",))
    assert rest[-1]["type"] == "cancelled"
    assert session.events_until(terminal=("status",))[-1]["state"] == "idle"
    assert llm.cancelled

    llm.first_token_delay = 0  # same connection, next turn works
    assert session.turn()[-1]["type"] == "done"


def test_cancel_while_recording_discards_the_audio(session_for):
    stt = FakeSTT()
    session = session_for(Services(stt, FakeLLM(), FakeTTS()))
    session.send(type="start")
    session.speak(0.5)
    session.send(type="cancel")
    events = session.events_until(terminal=("cancelled",))
    assert events[-1]["type"] == "cancelled"
    assert stt.probe.total == 0, "a cancelled recording must never be transcribed"
    assert session.turn()[-1]["type"] == "done"


def test_closing_the_socket_mid_turn_cancels_the_server_side_work(session_for):
    llm = FakeLLM(first_token_delay=30)
    session = session_for(Services(FakeSTT(), llm, FakeTTS()))
    session.send(type="start")
    session.speak(0.5)
    session.send(type="stop")
    while session.next_event()["type"] != "transcript":
        pass
    session.close()  # what a closed tab does

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not llm.cancelled:
        time.sleep(0.05)
    assert llm.cancelled, "the LLM stream kept running after the client left"
    metrics = httpx.get(f"{session.base}/api/metrics").json()
    assert metrics["load"]["inflight_turns"] == 0
    assert metrics["summary"]["turns_since_start"].get("cancelled") == 1


# ------------------------------------------------------------------------ errors keep the connection open

def test_provider_error_is_reported_and_the_connection_survives(session_for):
    llm = FakeLLM(error=ProviderError("llm", "Ollama is not reachable"))
    session = session_for(Services(FakeSTT(), llm, FakeTTS()))
    events = session.turn()
    error = events[-1]
    assert (error["type"], error["code"]) == ("error", "llm_error")
    assert "Ollama" in error["message"]
    llm.error = None
    assert session.turn()[-1]["type"] == "done"


def test_slow_llm_reports_a_timeout(session_for):
    session = session_for(
        Services(FakeSTT(), FakeLLM(first_token_delay=2), FakeTTS()), llm_first_token_timeout_s=0.2
    )
    assert session.turn()[-1]["code"] == "llm_timeout"


def test_stop_without_start_is_an_error(session_for):
    session = session_for()
    session.send(type="stop")
    assert session.next_event()["code"] == "invalid_request"


def test_unknown_and_malformed_messages_are_errors_not_crashes(session_for):
    session = session_for()
    session.send(type="dance")
    assert session.next_event()["code"] == "invalid_request"
    session.ws.send("this is not json")
    assert session.next_event()["code"] == "invalid_request"
    assert session.turn()[-1]["type"] == "done"  # still alive


def test_audio_frames_outside_a_recording_are_ignored(session_for):
    session = session_for()
    session.speak(0.3)  # no "start"
    assert session.turn()[-1]["type"] == "done"


def test_recording_longer_than_the_limit_is_rejected(session_for):
    session = session_for(max_audio_seconds=1.0)
    session.send(type="start")
    session.speak(2.0)
    errors = [e for e in session.events_until(terminal=("error",)) if e["type"] == "error"]
    assert errors[0]["code"] == "audio_too_long"
    session.send(type="stop")  # the recording was discarded, so there is nothing to stop
    assert session.events_until(terminal=("error",))[-1]["code"] == "invalid_request"


def test_starting_while_a_turn_runs_is_refused(session_for):
    session = session_for(Services(FakeSTT(), FakeLLM(first_token_delay=30), FakeTTS()))
    session.send(type="start")
    session.speak(0.3)
    session.send(type="stop")
    while session.next_event()["type"] != "transcript":
        pass
    session.send(type="start")
    refusal = session.events_until(terminal=("error",))[-1]
    assert refusal["code"] == "invalid_request" and "in progress" in refusal["message"]


def test_models_still_loading_is_reported_on_connect_and_on_start(live_server):
    base = live_server(Services(FakeSTT(load_delay=3), FakeLLM(), FakeTTS()), wait_ready=False)
    session = Session(base)
    try:
        assert session.ready["models_ready"] is False
        session.send(type="start")
        assert session.next_event()["code"] == "models_not_ready"
    finally:
        session.close()


def test_server_busy_is_reported_to_the_second_connection(live_server):
    base = live_server(Services(FakeSTT(delay=1.0), FakeLLM(), FakeTTS()), max_inflight_turns=1)
    first, second = Session(base), Session(base)
    try:
        first.send(type="start")
        first.speak(0.3)
        first.send(type="stop")
        time.sleep(0.3)  # first turn is now inside its (slow) STT
        second.send(type="start")
        second.speak(0.3)
        second.send(type="stop")
        assert second.events_until()[-1]["code"] == "server_busy"
        assert first.events_until()[-1]["type"] == "done"
    finally:
        first.close()
        second.close()


# ------------------------------------------------------------------------ origin check

def test_cross_site_websocket_is_refused(live_server):
    base = live_server()
    with pytest.raises(InvalidStatus) as info:
        Session(base, origin="http://evil.example")
    assert info.value.response.status_code == 403


def test_same_origin_websocket_is_accepted(live_server):
    base = live_server()
    assert Session(base, origin=base).ready["type"] == "ready"


def test_extra_allowed_origin_is_accepted(live_server):
    base = live_server(allowed_origins="http://localhost:3000, http://other.example")
    assert Session(base, origin="http://localhost:3000").ready["type"] == "ready"
