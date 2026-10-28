"""Talk to a running server without a browser.

    python scripts/voice_client.py benchmarks/inputs/short.wav
    python scripts/voice_client.py benchmarks/inputs/short.wav --transport http
    python scripts/voice_client.py question.wav --save-audio reply.wav

Also imported by scripts/benchmark.py, which is why the measuring code lives here.

What is timed (all by this client, with time.perf_counter, relative to "end of user input"):
  * websocket: the moment the final "stop" message has been sent. Audio frames are sent
    before that, as a browser would while the user is still talking.
  * http: the moment the request starts, because the upload only begins after the user stops.
"""

import argparse
import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

from app.services.audio import decode_audio_capped, pcm16_to_wav  # noqa: E402

FRAME_SAMPLES = 1600  # 100 ms at 16 kHz, what the browser sends


@dataclass
class TurnTrace:
    transport: str
    ok: bool = False
    error_code: str | None = None
    error_message: str | None = None
    transcript: str = ""
    reply: str = ""
    # seconds after end of user input; None = never happened
    t_transcript: float | None = None
    t_first_text: float | None = None
    t_first_audio: float | None = None
    t_done: float | None = None
    audio_seconds: float = 0.0
    audio_wav: bytes = b""
    server_metrics: dict = field(default_factory=dict)
    wer: float | None = None  # filled in by the benchmark when the spoken text is known


def load_pcm16(path: Path) -> bytes:
    """Any audio file -> 16 kHz mono signed 16-bit PCM bytes (what the websocket expects)."""
    samples = decode_audio_capped(path.read_bytes(), max_seconds=60)
    return np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()


async def ws_turn(base_url: str, pcm: bytes, *, realtime: bool = False, timeout: float = 120) -> TurnTrace:
    trace = TurnTrace(transport="ws")
    url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/voice"
    pending_audio: dict | None = None
    chunks: list[bytes] = []
    async with connect(url, max_size=None, open_timeout=10) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected first message: {ready}")
        await ws.send(json.dumps({"type": "start"}))
        step = FRAME_SAMPLES * 2
        for offset in range(0, len(pcm), step):
            await ws.send(pcm[offset : offset + step])
            if realtime:
                await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "stop"}))
        t0 = time.perf_counter()

        async with asyncio.timeout(timeout):
            async for message in ws:
                now = time.perf_counter() - t0
                if isinstance(message, bytes):
                    chunks.append(message)
                    continue
                event = json.loads(message)
                kind = event["type"]
                if kind == "transcript" and event["final"]:
                    trace.transcript, trace.t_transcript = event["text"], now
                elif kind == "reply_delta":
                    trace.t_first_text = trace.t_first_text if trace.t_first_text is not None else now
                    trace.reply += event["text"]
                elif kind == "audio":
                    pending_audio = event
                    trace.t_first_audio = trace.t_first_audio if trace.t_first_audio is not None else now
                elif kind == "done":
                    trace.ok, trace.t_done, trace.server_metrics = True, now, event["metrics"]
                    break
                elif kind in ("error", "cancelled"):
                    trace.error_code = event.get("code", kind)
                    trace.error_message = event.get("message")
                    trace.t_done = now
                    break
    if chunks and pending_audio:
        pcm_out = b"".join(chunks)
        trace.audio_seconds = len(pcm_out) / 2 / pending_audio["sample_rate"]
        trace.audio_wav = pcm16_to_wav(pcm_out, pending_audio["sample_rate"])
    return trace


async def http_turn(client: httpx.AsyncClient, base_url: str, wav_bytes: bytes) -> TurnTrace:
    trace = TurnTrace(transport="http")
    t0 = time.perf_counter()
    response = await client.post(f"{base_url}/api/turn", files={"audio": ("input.wav", wav_bytes, "audio/wav")})
    trace.t_done = time.perf_counter() - t0
    body = response.json()
    if response.status_code != 200:
        trace.error_code = body["error"]["code"]
        trace.error_message = body["error"]["message"]
        return trace
    trace.ok = True
    trace.transcript, trace.reply = body["transcript"], body["reply"]
    trace.server_metrics = body["metrics"]
    # Nothing is available to the user before the whole response arrives.
    trace.t_transcript = trace.t_first_text = trace.t_first_audio = trace.t_done
    if body["audio"]:
        trace.audio_wav = base64.b64decode(body["audio"]["data_base64"])
        trace.audio_seconds = body["audio"]["duration_s"]
    return trace


def seconds(value: float | None) -> str:
    return "never" if value is None else f"+{value:.2f} s"


def print_trace(trace: TurnTrace) -> None:
    print(f"transport     : {trace.transport}")
    if not trace.ok:
        print(f"FAILED        : {trace.error_code}: {trace.error_message}")
        return
    print(f"transcript    : {trace.transcript!r}  ({seconds(trace.t_transcript)})")
    print(f"reply         : {trace.reply!r}  (first text {seconds(trace.t_first_text)})")
    print(f"first audio   : {seconds(trace.t_first_audio)}")
    print(f"turn complete : {seconds(trace.t_done)}   ({trace.audio_seconds:.1f} s of speech returned)")
    print("server metrics:")
    for key, value in trace.server_metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s} {value}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="WAV/WebM/MP3/... file to send as the user's utterance")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--transport", choices=["ws", "http"], default="ws")
    parser.add_argument("--realtime", action="store_true", help="ws: pace audio frames at real time")
    parser.add_argument("--save-audio", type=Path, help="write the spoken reply to this WAV file")
    args = parser.parse_args()

    if args.transport == "ws":
        trace = await ws_turn(args.url, load_pcm16(args.audio), realtime=args.realtime)
    else:
        wav = pcm16_to_wav(load_pcm16(args.audio), 16000)
        async with httpx.AsyncClient(timeout=180) as client:
            trace = await http_turn(client, args.url, wav)
    print_trace(trace)
    if args.save_audio and trace.audio_wav:
        args.save_audio.write_bytes(trace.audio_wav)
        print(f"saved reply audio to {args.save_audio}")
    return 0 if trace.ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
