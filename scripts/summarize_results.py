"""Turn benchmark result files into markdown tables, so README numbers are copied from data.

    python scripts/summarize_results.py benchmarks/results/*-baseline-warm.json
    python scripts/summarize_results.py --experiment stt-tiny benchmarks/results/*-stt-tiny.json

Prints the machine and server description of each file first, then one table of latency and one
of server-side stage medians.
"""

import argparse
import json
import sys
from pathlib import Path


def span(stats: dict | None, digits: int = 2) -> str:
    if not stats:
        return "—"
    return f"{stats['median']:.{digits}f} [{stats['min']:.{digits}f}–{stats['max']:.{digits}f}]"


def describe_file(result: dict) -> str:
    machine, server = result["machine"], result["server"]
    comps = server["components"]
    bg = result.get("background_cpu_percent", {})
    lines = [
        f"**{result['label']}** — started {result['started_at'][:19]}Z",
        f"- machine: {machine['cpu']} ({machine.get('physical_cores', '?')} cores / {machine['logical_cpus']} threads), "
        f"{machine['ram_gb']} GB RAM, {machine['os']}, Python {machine['python']}; GPU listed: {machine['gpu']} (not used)",
        f"- background CPU use with the server idle: before {bg.get('before')}%, after {bg.get('after')}%",
        f"- STT {comps['stt']}",
        f"- LLM {comps['llm']}",
        f"- TTS {comps['tts']}",
        "- server settings: " + ", ".join(
            f"{k}={server['settings'][k]}"
            for k in ("stt_concurrency", "llm_concurrency", "tts_concurrency", "max_inflight_turns",
                      "warmup_on_start", "partial_transcripts", "tts_first_clause_chars")
        ),
        f"- {result['arguments']['runs']} runs x {result['arguments']['requests']} requests per config, "
        f"{result['arguments']['warmup']} warm-up requests, realtime audio pacing: {result['arguments'].get('realtime', False)}",
    ]
    if result.get("cold"):
        for transport, trace in result["cold"].items():
            lines.append(f"- cold first request ({transport}, single sample): first audio {trace['t_first_audio']:.2f} s, "
                         f"done {trace['t_done']:.2f} s")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.files)]

    for result in results:
        print(describe_file(result) + "\n")

    print("| experiment | config | ok / total | first audio, s (median [min–max of run medians]) | turn done, s | p95 done, s | req/s | STT WER |")
    print("|---|---|---|---|---|---|---|---|")
    for result in results:
        for config, agg in result["summary"].items():
            wer = "—" if agg.get("stt_wer_mean") is None else f"{agg['stt_wer_mean'] * 100:.1f}%"
            p95 = agg["turn_done_s_p95_of_runs"]
            print(f"| {result['label']} | {config} | {agg['ok']} / {agg['requests']} | {span(agg['first_audio_s_median_of_runs'])} "
                  f"| {span(agg['turn_done_s_median_of_runs'])} | {p95['median']:.2f} | {agg['throughput_rps']['median']:.2f} | {wer} |"
                  if p95 else f"| {result['label']} | {config} | {agg['ok']} / {agg['requests']} | — | — | — | — | — |")

    stages = ["stt_queue_ms", "stt_ms", "llm_ttft_ms", "llm_ms", "tts_queue_ms", "tts_first_chunk_ms", "tts_ms"]
    print("\n| experiment | config | " + " | ".join(s.removesuffix("_ms") for s in stages) + " |")
    print("|---|---|" + "---|" * len(stages))
    for result in results:
        for config, agg in result["summary"].items():
            cells = [f"{agg['stage_medians_ms'][s]:.0f}" if s in agg["stage_medians_ms"] else "—" for s in stages]
            print(f"| {result['label']} | {config} | " + " | ".join(cells) + " |")
    print("\n(stage columns: server-side milliseconds, median across runs)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
