"""Benchmark a voice server with fixed spoken prompts.

Warm, steady-state numbers against a server you already started (both transports,
concurrency 1 and 2, three runs of six requests each, two unmeasured warm-up requests):

    python scripts/benchmark.py --transport both --concurrency 1,2 --requests 6 --runs 3 --warmup 2 --label baseline

Cold start: this script starts its own server (WARMUP_ON_START=false) and, for Ollama, unloads
the model first, so the first request really is the first one. Extra server settings go in --env:

    python scripts/benchmark.py --spawn-server --cold --transport ws --requests 6 --runs 3 --label cold --env STT_MODEL=tiny.en

Inputs are the WAV files made by scripts/make_test_audio.py. Results (JSON + CSV, including every
raw request) are written to benchmarks/results/. Nothing here is hard-coded or simulated: every
number comes from a request that was actually sent.

Latency definitions (client-side, seconds after the end of user input; see voice_client.py):
  first audio  the first synthesized audio reached the client (what a listener perceives)
  turn done    the whole reply has arrived
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from make_test_audio import PROMPTS  # noqa: E402  (the text each input WAV was synthesized from)
from voice_client import TurnTrace, http_turn, load_pcm16, ws_turn  # noqa: E402

from app.services.audio import decode_audio_capped, pcm16_to_wav  # noqa: E402
from app.services.metrics import percentile  # noqa: E402

DEFAULT_INPUTS = ["benchmarks/inputs/short.wav", "benchmarks/inputs/medium.wav", "benchmarks/inputs/long.wav"]
# Server-side stage timings copied from each response's metrics.
STAGE_KEYS = [
    "stt_queue_ms", "stt_ms", "llm_queue_ms", "llm_ttft_ms", "llm_ms",
    "tts_queue_ms", "tts_first_chunk_ms", "tts_ms", "first_audio_ms", "total_ms",
]
MIN_RUNS_FOR_COMPARISON = 3


# ---------------------------------------------------------------------------- statistics

def _words(text: str) -> list[str]:
    return "".join(c if c.isalnum() or c == "'" else " " for c in text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the number of reference words (0.0 = perfect).

    Case and punctuation are ignored; hyphens count as spaces ("three-day" == "three day").
    """
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        raise ValueError("empty reference")
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1] / len(ref)


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_run(traces: list[TurnTrace], wall_seconds: float) -> dict:
    """Reduce one run (a batch of requests at one concurrency level) to its statistics."""
    ok = [t for t in traces if t.ok]
    failed = [t for t in traces if not t.ok]
    stages = {}
    for key in STAGE_KEYS:
        stats = describe([t.server_metrics[key] for t in ok if t.server_metrics.get(key) is not None])
        if stats:
            stages[key] = stats
    return {
        "requests": len(traces),
        "ok": len(ok),
        "failed": len(failed),
        "error_rate": len(failed) / len(traces) if traces else 0.0,
        "errors": dict(Counter(t.error_code for t in failed)),
        "wall_seconds": wall_seconds,
        "throughput_rps": len(ok) / wall_seconds if wall_seconds > 0 else 0.0,
        "first_audio_s": describe([t.t_first_audio for t in ok if t.t_first_audio is not None]),
        "turn_done_s": describe([t.t_done for t in ok if t.t_done is not None]),
        "stt_wer": describe([t.wer for t in ok if t.wer is not None]),
        "stages_ms": stages,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Combine repeated runs of one configuration: spread of the per-run medians, and totals."""

    def across(path: str, stat: str = "median") -> dict | None:
        values = [r[path][stat] for r in runs if r.get(path)]
        return {"median": percentile(values, 50), "min": min(values), "max": max(values)} if values else None

    total = sum(r["requests"] for r in runs)
    failed = sum(r["failed"] for r in runs)
    return {
        "runs": len(runs),
        "requests": total,
        "ok": total - failed,
        "failed": failed,
        "error_rate": failed / total if total else 0.0,
        "errors": dict(sum((Counter(r["errors"]) for r in runs), Counter())),
        "first_audio_s_median_of_runs": across("first_audio_s"),
        "first_audio_s_p95_of_runs": across("first_audio_s", "p95"),
        "turn_done_s_median_of_runs": across("turn_done_s"),
        "turn_done_s_p95_of_runs": across("turn_done_s", "p95"),
        "throughput_rps": {
            "median": percentile([r["throughput_rps"] for r in runs], 50),
            "min": min(r["throughput_rps"] for r in runs),
            "max": max(r["throughput_rps"] for r in runs),
        },
        "stt_wer_mean": (lambda w: sum(w) / len(w) if w else None)([r["stt_wer"]["mean"] for r in runs if r.get("stt_wer")]),
        "stage_medians_ms": {
            key: percentile([r["stages_ms"][key]["median"] for r in runs if key in r["stages_ms"]], 50)
            for key in STAGE_KEYS
            if any(key in r["stages_ms"] for r in runs)
        },
    }


def csv_rows(label: str, runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        for index, trace in enumerate(run["traces"]):
            rows.append({
                "label": label, "transport": run["transport"], "concurrency": run["concurrency"], "run": run["run"],
                "index": index, "input": trace["input"], "ok": trace["ok"], "error_code": trace["error_code"],
                "t_transcript": trace["t_transcript"], "t_first_text": trace["t_first_text"],
                "t_first_audio": trace["t_first_audio"], "t_done": trace["t_done"],
                "audio_seconds": trace["audio_seconds"],
                **{key: trace["server_metrics"].get(key) for key in STAGE_KEYS},
            })
    return rows


def trace_to_dict(trace: TurnTrace, input_name: str) -> dict:
    return {
        "input": input_name, "ok": trace.ok, "error_code": trace.error_code, "error_message": trace.error_message,
        "transcript": trace.transcript, "reply": trace.reply, "wer": trace.wer,
        "t_transcript": trace.t_transcript, "t_first_text": trace.t_first_text,
        "t_first_audio": trace.t_first_audio, "t_done": trace.t_done,
        "audio_seconds": trace.audio_seconds, "server_metrics": trace.server_metrics,
    }


# ---------------------------------------------------------------------------- environment description

def _powershell(command: str) -> str:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=20
    ).stdout.strip()


def machine_info() -> dict:
    info = {
        "os": platform.platform(), "python": platform.python_version(), "arch": platform.machine(),
        "logical_cpus": os.cpu_count(), "cpu": "unknown", "ram_gb": None, "gpu": "unknown",
    }
    try:
        if sys.platform == "win32":
            info["cpu"] = _powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
            info["physical_cores"] = int(_powershell("(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"))
            info["ram_gb"] = round(int(_powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")) / 2**30, 1)
            info["gpu"] = _powershell("(Get-CimInstance Win32_VideoController).Name") or "unknown"
        elif sys.platform == "darwin":
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            info["ram_gb"] = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30, 1)
        else:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            info["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), "unknown")
            meminfo = Path("/proc/meminfo").read_text()
            info["ram_gb"] = round(int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith("MemTotal"))) / 2**20, 1)
    except (OSError, ValueError, subprocess.SubprocessError, StopIteration):
        pass  # leave the "unknown" placeholders: better than guessing
    return info


async def cpu_busy_percent(seconds: float = 5.0) -> float | None:
    """Overall CPU utilisation over a short window, or None if this platform is not supported.

    Measured while the benchmark is NOT sending requests, so it estimates background load
    (browsers, editors, ...) that competes with the models for the same cores.
    """
    try:
        if sys.platform == "win32":
            command = f"(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples {int(seconds)}).CounterSamples.CookedValue | Measure-Object -Average | Select-Object -ExpandProperty Average"
            out = await asyncio.to_thread(_powershell, command)
            return round(float(out), 1)
        if sys.platform.startswith("linux"):
            def read() -> tuple[int, int]:
                values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
                return values[3] + values[4], sum(values)  # idle + iowait, total

            idle_a, total_a = read()
            await asyncio.sleep(seconds)
            idle_b, total_b = read()
            return round(100 * (1 - (idle_b - idle_a) / (total_b - total_a)), 1)
    except (OSError, ValueError, subprocess.SubprocessError, ZeroDivisionError):
        pass
    return None


def describe_inputs(paths: list[Path]) -> dict:
    return {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "seconds": round(len(decode_audio_capped(p.read_bytes(), 60)) / 16000, 2),
        }
        for p in paths
    }


# ---------------------------------------------------------------------------- running requests

class Target:
    """Everything needed to send one request: the server URL, preloaded inputs and a shared HTTP client."""

    def __init__(self, base_url: str, inputs: list[Path], timeout: float, realtime: bool = False):
        self.base_url, self.timeout, self.realtime = base_url, timeout, realtime
        self.names = [p.name for p in inputs]
        self.references = [PROMPTS.get(p.stem) for p in inputs]  # None for files we did not generate
        self.pcm = [load_pcm16(p) for p in inputs]
        self.wav = [pcm16_to_wav(pcm, 16000) for pcm in self.pcm]
        self.client: httpx.AsyncClient | None = None

    async def one(self, transport: str, index: int) -> tuple[str, TurnTrace]:
        i = index % len(self.names)
        try:
            if transport == "ws":
                trace = await ws_turn(self.base_url, self.pcm[i], realtime=self.realtime, timeout=self.timeout)
            else:
                trace = await http_turn(self.client, self.base_url, self.wav[i])
        except Exception as exc:  # a failed request is data, not a reason to abort the whole benchmark
            return self.names[i], TurnTrace(transport=transport, error_code=type(exc).__name__, error_message=str(exc))
        if trace.ok and self.references[i]:
            trace.wer = word_error_rate(self.references[i], trace.transcript)
        return self.names[i], trace


async def run_load(target: Target, transport: str, concurrency: int, requests: int) -> tuple[list[tuple[str, TurnTrace]], float]:
    """Closed-loop load: `concurrency` workers each send their next request as soon as the last one finishes."""
    next_index = 0
    results: list[tuple[str, TurnTrace]] = []

    async def worker() -> None:
        nonlocal next_index
        while next_index < requests:
            index, next_index = next_index, next_index + 1
            results.append(await target.one(transport, index))

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, time.perf_counter() - started


# ---------------------------------------------------------------------------- spawning a server (for cold starts)

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_server(env_overrides: dict[str, str], log_path: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0  # lets us send CTRL_BREAK later
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "LOG_FORMAT": "text", **env_overrides},
        stdout=log_path.open("w"), stderr=subprocess.STDOUT, creationflags=flags,
        cwd=Path(__file__).resolve().parents[1],
    )
    return process, f"http://127.0.0.1:{port}"


def stop_server(process: subprocess.Popen) -> None:
    process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()


async def wait_until_ready(base_url: str, timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{base_url}/ready", timeout=3)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"{base_url} did not become ready within {timeout:g}s")


async def unload_ollama_model(info: dict, env_overrides: dict[str, str]) -> bool:
    """Evict the model from Ollama's memory so a cold request pays the model-load cost."""
    llm = info["components"]["llm"]
    if llm.get("provider") != "ollama":
        return False
    base = env_overrides.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base}/api/generate", json={"model": llm["model"], "keep_alive": 0}, timeout=30)
    return response.status_code == 200


# ---------------------------------------------------------------------------- reporting

def fmt(value: float | None, unit: str = "s") -> str:
    return "—" if value is None else f"{value:.2f}{unit}"


def span(stats: dict | None) -> str:
    return "—" if stats is None else f"{stats['median']:.2f}s [{stats['min']:.2f}–{stats['max']:.2f}]"


def print_summary(summary: dict, cold: dict | None) -> None:
    if cold:
        print("\nCOLD start (first request on a freshly started server; a single sample, not a statistic)")
        for transport, trace in cold.items():
            print(f"  {transport}: first audio {fmt(trace['t_first_audio'])}, done {fmt(trace['t_done'])}"
                  f"  ok={trace['ok']} {trace['error_code'] or ''}")
    print("\nWARM results  (median of per-run medians, [min–max across runs])")
    header = f"{'config':<12}{'ok/total':>10}{'err%':>7}{'first audio':>22}{'turn done':>22}{'p95 done':>10}{'req/s':>8}{'WER%':>7}"
    print(header + "\n" + "-" * len(header))
    for name, agg in summary.items():
        fa, td = agg["first_audio_s_median_of_runs"], agg["turn_done_s_median_of_runs"]
        p95 = agg["turn_done_s_p95_of_runs"]
        wer = "—" if agg["stt_wer_mean"] is None else f"{agg['stt_wer_mean'] * 100:.1f}"
        print(f"{name:<12}{agg['ok']:>4}/{agg['requests']:<5}{agg['error_rate'] * 100:>6.1f}%{span(fa):>22}{span(td):>22}"
              f"{fmt(p95['median'] if p95 else None):>10}{agg['throughput_rps']['median']:>8.2f}{wer:>7}")
    print("\nStage medians in ms (server-side, median across runs)")
    stages = ["stt_queue_ms", "stt_ms", "llm_ttft_ms", "llm_ms", "tts_queue_ms", "tts_first_chunk_ms", "tts_ms"]
    print(f"{'config':<12}" + "".join(f"{s.replace('_ms', ''):>16}" for s in stages))
    for name, agg in summary.items():
        print(f"{name:<12}" + "".join(f"{agg['stage_medians_ms'].get(s, float('nan')):>16.0f}" for s in stages))
    if any(a["errors"] for a in summary.values()):
        print("\nErrors by code:", {n: a["errors"] for n, a in summary.items() if a["errors"]})


# ---------------------------------------------------------------------------- main

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server to test (ignored with --spawn-server)")
    p.add_argument("--transport", choices=["ws", "http", "both"], default="both")
    p.add_argument("--concurrency", default="1", help="comma-separated levels, e.g. 1,2,4")
    p.add_argument("--requests", type=int, default=6, help="requests per run")
    p.add_argument("--runs", type=int, default=3, help="repetitions of every configuration (use at least 3)")
    p.add_argument("--warmup", type=int, default=2, help="unmeasured sequential requests per transport before measuring")
    p.add_argument("--inputs", nargs="+", type=Path, default=[Path(x) for x in DEFAULT_INPUTS])
    p.add_argument("--label", default="run")
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--timeout", type=float, default=180, help="per-request timeout in seconds")
    p.add_argument("--realtime", action="store_true",
                   help="ws: send audio frames at speaking speed instead of all at once (needed to exercise partial transcripts)")
    p.add_argument("--spawn-server", action="store_true", help="start (and stop) a private server for this benchmark")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="server env override (with --spawn-server)")
    p.add_argument("--cold", action="store_true", help="measure the first request after start (implies --spawn-server)")
    args = p.parse_args(argv)
    args.concurrency = [int(c) for c in args.concurrency.split(",")]
    args.cold = args.cold or False
    args.spawn_server = args.spawn_server or args.cold
    args.env_overrides = dict(e.split("=", 1) for e in args.env)
    if args.requests < 1 or args.runs < 1 or min(args.concurrency) < 1:
        p.error("--requests, --runs and every --concurrency level must be >= 1")
    return args


async def main(args: argparse.Namespace) -> int:
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        print(f"Missing input audio {missing}. Run: python scripts/make_test_audio.py", file=sys.stderr)
        return 2
    if args.runs < MIN_RUNS_FOR_COMPARISON:
        print(f"WARNING: --runs {args.runs} < {MIN_RUNS_FOR_COMPARISON}; not enough to compare configurations.\n")

    process = None
    base_url = args.url.rstrip("/")
    started_at = datetime.now(UTC)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = dict(args.env_overrides)
        if args.cold:
            env["WARMUP_ON_START"] = "false"
        if args.spawn_server:
            process, base_url = spawn_server(env, args.output_dir / f"{stamp}-{args.label}-server.log")
            print(f"started private server at {base_url} (pid {process.pid}); waiting for it to be ready…")
        await wait_until_ready(base_url)
        async with httpx.AsyncClient(timeout=30) as client:
            info = (await client.get(f"{base_url}/api/info")).json()
        print(f"server: stt={info['components']['stt']}\n        llm={info['components']['llm']}\n        tts={info['components']['tts']}")

        background_before = await cpu_busy_percent()
        print(f"background CPU use before the benchmark (server idle): {background_before}%")
        target = Target(base_url, args.inputs, args.timeout, args.realtime)
        target.client = httpx.AsyncClient(timeout=args.timeout)
        transports = ["ws", "http"] if args.transport == "both" else [args.transport]

        cold = None
        if args.cold:
            evicted = await unload_ollama_model(info, args.env_overrides)
            print(f"cold start: LLM evicted from memory: {evicted}")
            cold = {}
            for transport in transports[:1]:  # only the very first request on the server is truly cold
                name, trace = await target.one(transport, 0)
                cold[transport] = trace_to_dict(trace, name)
                print(f"  cold {transport} request: first audio {fmt(trace.t_first_audio)}, done {fmt(trace.t_done)}")

        if args.warmup:
            print(f"warming up ({args.warmup} requests per transport, not measured)…")
            for transport in transports:
                await run_load(target, transport, 1, args.warmup)

        configs = [(t, c) for t in transports for c in args.concurrency]
        runs: list[dict] = []
        # Interleave configurations (run 1 of everything, then run 2, ...) so slow drift such as
        # thermal throttling or background load hits every configuration equally.
        for run_number in range(1, args.runs + 1):
            for transport, concurrency in configs:
                results, wall = await run_load(target, transport, concurrency, args.requests)
                summary = summarize_run([t for _, t in results], wall)
                summary.update(
                    transport=transport, concurrency=concurrency, run=run_number,
                    traces=[trace_to_dict(t, n) for n, t in results],
                )
                runs.append(summary)
                print(f"run {run_number}/{args.runs} {transport:<4} c={concurrency}: ok {summary['ok']}/{summary['requests']}"
                      f"  first audio median {fmt(summary['first_audio_s'] and summary['first_audio_s']['median'])}"
                      f"  done median {fmt(summary['turn_done_s'] and summary['turn_done_s']['median'])}"
                      f"  {summary['throughput_rps']:.2f} req/s")
        await target.client.aclose()
        background_after = await cpu_busy_percent()
        print(f"background CPU use after the benchmark (server idle): {background_after}%")

        summary = {
            f"{t}-c{c}": aggregate_runs([r for r in runs if (r["transport"], r["concurrency"]) == (t, c)])
            for t, c in configs
        }
        print_summary(summary, cold)

        result = {
            "schema": 1, "label": args.label, "started_at": started_at.isoformat(),
            "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "inputs"},
            "machine": machine_info(),
            "background_cpu_percent": {"before": background_before, "after": background_after},
            "server": info, "inputs": describe_inputs(args.inputs),
            "server_was_spawned_by_benchmark": args.spawn_server, "cold": cold, "summary": summary, "runs": runs,
        }
        json_path = args.output_dir / f"{stamp}-{args.label}.json"
        json_path.write_text(json.dumps(result, indent=2, default=str))
        rows = csv_rows(args.label, runs)
        csv_path = json_path.with_suffix(".csv")
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved {json_path}\nsaved {csv_path}")
        return 0
    finally:
        if process:
            stop_server(process)


if __name__ == "__main__":
    # Windows consoles and pipes default to a legacy code page that cannot print some of the symbols above.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(main(parse_args())))
