"""Graceful shutdown of a real server process that is in the middle of a turn."""

import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from tests.test_websocket import tone_pcm

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_termination_signal_closes_connections_and_exits_cleanly_with_a_turn_in_flight(tmp_path):
    port, log_path = free_port(), tmp_path / "server.log"
    windows = sys.platform == "win32"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.serve_fakes:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(port), "--timeout-graceful-shutdown", "3"],
        cwd=ROOT, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,  # needed to send CTRL_BREAK
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/ready", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline and server.poll() is None, log_path.read_text()
            time.sleep(0.2)

        with connect(f"ws://127.0.0.1:{port}/ws/voice", open_timeout=5) as ws:
            ws.recv(timeout=5)  # ready
            ws.send(json.dumps({"type": "start"}))
            ws.send(tone_pcm(0.5))
            ws.send(json.dumps({"type": "stop"}))
            while json.loads(ws.recv(timeout=10)).get("type") != "transcript":
                pass  # the turn is now blocked waiting for the (never-answering) LLM

            server.send_signal(signal.CTRL_BREAK_EVENT if windows else signal.SIGTERM)

            with pytest.raises(ConnectionClosed):  # the server must hang up on us, not leave us waiting
                while True:
                    ws.recv(timeout=15)

        exit_code = server.wait(timeout=15)
    finally:
        if server.poll() is None:
            server.kill()

    log = log_path.read_text()
    assert "shutting down" in log, log  # our lifespan shutdown handler ran
    assert "Application shutdown complete" in log, log  # uvicorn finished its own shutdown, not killed mid-way
    # After a graceful, signal-triggered shutdown uvicorn re-raises the signal with its default action
    # (see uvicorn/server.py), so the exit status reports the signal: 3 for CTRL_BREAK on Windows,
    # -SIGTERM on POSIX. A kill() or a crash would show up as something else.
    expected = 3 if windows else -signal.SIGTERM
    assert exit_code == expected, f"exit code {exit_code}\n{log}"
