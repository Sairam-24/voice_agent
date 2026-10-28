"""Compare configurations that were benchmarked in rotating rounds, using within-round differences.

Why: laptops drift. Background load, thermals and cache state change over minutes, so comparing a
variant measured at 10:20 with a baseline measured at 10:00 mixes the effect with the drift. In a
rotation every variant is measured once per round, in a different order each round, and each variant
is compared with the baseline *of the same round*.

    python scripts/compare_rotation.py ab        # reads benchmarks/results/*-ab-r<round>-<variant>.json

A change is only called an improvement or a regression when the paired difference has the same sign
in every round. With three rounds that is a modest bar, not statistical proof; it mainly protects
against reading drift as an effect.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

METRICS = {
    "first audio (s)": lambda s: s["first_audio_s_median_of_runs"]["median"],
    "turn done (s)": lambda s: s["turn_done_s_median_of_runs"]["median"],
    "stt (ms)": lambda s: s["stage_medians_ms"].get("stt_ms"),
    "llm total (ms)": lambda s: s["stage_medians_ms"].get("llm_ms"),
    "tts first chunk (ms)": lambda s: s["stage_medians_ms"].get("tts_first_chunk_ms"),
}


def load(prefix: str, results_dir: Path) -> dict[str, dict[int, dict]]:
    """variant -> round -> summary of its single configuration (ws, concurrency 1)."""
    pattern = re.compile(rf"-{re.escape(prefix)}-r(\d+)-(.+)\.json$")
    found: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in results_dir.glob(f"*-{prefix}-r*.json"):
        match = pattern.search(path.name)
        if not match:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = next(iter(result["summary"].values()))
        summary["_wer"] = summary.get("stt_wer_mean")
        summary["_background"] = result.get("background_cpu_percent", {})
        found[match.group(2)][int(match.group(1))] = summary
    return found


def verdict(deltas: list[float]) -> str:
    if len(deltas) < 3:
        return f"only {len(deltas)} paired rounds"
    if all(d < 0 for d in deltas):
        return "lower in every round"
    if all(d > 0 for d in deltas):
        return "higher in every round"
    return "inconclusive (sign changes between rounds)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="label prefix used for the rotation, e.g. 'ab'")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    data = load(args.prefix, args.results_dir)
    if args.baseline not in data:
        sys.exit(f"no '{args.baseline}' results found for prefix '{args.prefix}'")
    base = data[args.baseline]
    rounds = sorted(base)

    print(f"Rounds with a baseline: {rounds}.  Each cell is that round's median; Δ = variant minus the same round's baseline.\n")
    for metric, getter in METRICS.items():
        print(f"### {metric}")
        print("| variant | " + " | ".join(f"round {r}" for r in rounds) + " | median of rounds | paired Δ per round | reading |")
        print("|---|" + "---|" * (len(rounds) + 3))
        for variant in [args.baseline, *sorted(v for v in data if v != args.baseline)]:
            cells, deltas = [], []
            for r in rounds:
                value = getter(data[variant][r]) if r in data[variant] else None
                cells.append("—" if value is None else f"{value:.2f}" if "(s)" in metric else f"{value:.0f}")
                base_value = getter(base[r])
                if value is not None and base_value is not None and variant != args.baseline:
                    deltas.append(value - base_value)
            values = [getter(data[variant][r]) for r in rounds if r in data[variant] and getter(data[variant][r]) is not None]
            mid = "—" if not values else (f"{median(values):.2f}" if "(s)" in metric else f"{median(values):.0f}")
            if variant == args.baseline:
                print(f"| {variant} | " + " | ".join(cells) + f" | {mid} | — | reference |")
            else:
                fmt = (lambda d: f"{d:+.2f}") if "(s)" in metric else (lambda d: f"{d:+.0f}")
                print(f"| {variant} | " + " | ".join(cells) + f" | {mid} | {', '.join(fmt(d) for d in deltas)} | {verdict(deltas)} |")
        print()

    print("### transcription error rate (clean synthetic speech only; not representative of real microphones)")
    for variant in sorted(data):
        wers = [data[variant][r]["_wer"] for r in sorted(data[variant]) if data[variant][r]["_wer"] is not None]
        print(f"- {variant}: " + (", ".join(f"{w * 100:.1f}%" for w in wers) if wers else "—"))
    print("\n### background CPU use with the server idle, before each window (%)")
    for variant in sorted(data):
        print(f"- {variant}: " + ", ".join(str(data[variant][r]["_background"].get("before")) for r in sorted(data[variant])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
