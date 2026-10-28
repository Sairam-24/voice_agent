import os
import threading
import time
from contextlib import ExitStack

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.audio import pcm16_to_wav
from app.startup import Services
from tests.fakes import FakeLLM, FakeSTT, FakeTTS

SETTING_NAMES = {name.upper() for name in Settings.model_fields}


@pytest.fixture(autouse=True)
def isolated_from_shell_settings(request, monkeypatch):
    """Unit tests must not depend on the developer's shell: a GEMINI_API_KEY or LLM_PROVIDER exported
    there would otherwise change the defaults they test. Integration tests read the key on purpose."""
    if request.node.get_closest_marker("integration"):
        return
    for name in [n for n in os.environ if n.upper() in SETTING_NAMES]:
        monkeypatch.delenv(name)


def make_settings(**overrides) -> Settings:
    base = dict(web_dir="does-not-exist", log_format="text", log_level="WARNING", warmup_on_start=False)
    return Settings(_env_file=None, **{**base, **overrides})


def sine_wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    pcm = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2").tobytes()
    return pcm16_to_wav(pcm, sample_rate)


def wait_until_ready(client: TestClient, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/ready").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError(f"app not ready: {client.get('/health').json()}")


@pytest.fixture
def live_server():
    """Factory: start_server(services, **settings) -> base URL of a real uvicorn server on a free port.

    Needed where TestClient cannot help: client disconnects and real WebSocket traffic.
    """
    running = []

    def start(services: Services | None = None, wait_ready: bool = True, **overrides) -> str:
        app = create_app(make_settings(**overrides), services or Services(FakeSTT(), FakeLLM(), FakeTTS()))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        running.append((server, thread))
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)
        base_url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        while wait_ready and httpx.get(f"{base_url}/ready").status_code != 200:
            assert time.monotonic() < deadline, "app did not become ready"
            time.sleep(0.02)
        return base_url

    yield start
    for server, thread in running:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def fakes() -> Services:
    return Services(stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS())


@pytest.fixture
def make_client():
    """Factory: make_client(services=..., **settings_overrides) -> TestClient with lifespan running."""
    stack = ExitStack()

    def factory(services: Services | None = None, wait: bool = True, **overrides) -> TestClient:
        app = create_app(make_settings(**overrides), services or Services(FakeSTT(), FakeLLM(), FakeTTS()))
        client = stack.enter_context(TestClient(app))
        if wait:
            wait_until_ready(client)
        return client

    yield factory
    stack.close()


@pytest.fixture
def client(make_client, fakes) -> TestClient:
    return make_client(services=fakes)
