"""Run a large replication study and export summary statistics.

Usage:
    python3 scripts/run_large_replication_study.py --samples 200000

By default this runs many independent baseline matches with distinct seeds,
stores one row per run, and writes aggregate confidence interval statistics
for the resulting summary metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from statistics import NormalDist
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fps_pvp_abm import FpsPvpModel, SimulationConfig

OUT_DIR = ROOT / "out" / "experiments" / "large_replication_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a large batch of FPS PvP simulations.")
    parser.add_argument("--samples", type=int, default=200000, help="Number of simulation runs to execute.")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Worker processes.")
    parser.add_argument("--max-ticks", type=int, default=200, help="Maximum ticks per simulation.")
    parser.add_argument("--n-agents", type=int, default=30, help="Number of agents per team setup.")
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed; each run adds its index.")
    return parser.parse_args()


def run_one(args: tuple[int, int, int, int]) -> dict[str, float | int]:
    idx, seed_base, max_ticks, n_agents = args
    config = SimulationConfig(max_ticks=max_ticks, n_agents=n_agents, seed=seed_base + idx)
    model = FpsPvpModel(config)
    summary = model.run()
    return {
        "sample_index": idx,
        "seed": seed_base + idx,
        "ticks": summary.get("ticks", 0.0),
        "objective_progress": summary.get("objective_progress", 0.0),
        "objective_controller": summary.get("objective_controller", -1.0),
    }


def iter_results(sample_count: int, seed_base: int, max_ticks: int, n_agents: int, workers: int) -> Iterable[dict[str, float | int]]:
    tasks = ((idx, seed_base, max_ticks, n_agents) for idx in range(sample_count))
    with mp.Pool(processes=workers) as pool:
        yield from pool.imap_unordered(run_one, tasks, chunksize=max(1, sample_count // (workers * 50) or 1))


def summarize(rows: list[dict[str, float | int]]) -> dict[str, float]:
    metrics = ["ticks", "objective_progress", "objective_controller"]
    z_score = NormalDist().inv_cdf(0.975)
    summary: dict[str, float] = {"samples": float(len(rows))}

    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0
        stderr = stddev / math.sqrt(len(values)) if values else 0.0
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_stddev"] = stddev
        summary[f"{metric}_stderr"] = stderr
        summary[f"{metric}_ci95_low"] = mean - z_score * stderr
        summary[f"{metric}_ci95_high"] = mean + z_score * stderr

    return summary


def main() -> None:
    args = parse_args()
    raw_csv = OUT_DIR / "large_replication_runs.csv"
    summary_json = OUT_DIR / "large_replication_summary.json"
    summary_csv = OUT_DIR / "large_replication_summary.csv"

    rows: list[dict[str, float | int]] = []
    fieldnames = ["sample_index", "seed", "ticks", "objective_progress", "objective_controller"]

    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in iter_results(args.samples, args.seed_base, args.max_ticks, args.n_agents, args.workers):
            writer.writerow(result)
            rows.append(result)
            if len(rows) % 1000 == 0 or len(rows) == args.samples:
                print(f"Completed {len(rows)}/{args.samples} runs")

    summary = summarize(rows)
    summary.update(
        {
            "samples": args.samples,
            "max_ticks": args.max_ticks,
            "n_agents": args.n_agents,
            "seed_base": args.seed_base,
            "workers": args.workers,
        }
    )

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "stddev", "stderr", "ci95_low", "ci95_high"])
        for metric in ("ticks", "objective_progress", "objective_controller"):
            writer.writerow(
                [
                    metric,
                    summary[f"{metric}_mean"],
                    summary[f"{metric}_stddev"],
                    summary[f"{metric}_stderr"],
                    summary[f"{metric}_ci95_low"],
                    summary[f"{metric}_ci95_high"],
                ]
            )

    print(f"Raw runs written to: {raw_csv}")
    print(f"Aggregate summary written to: {summary_json}")
    print(f"Aggregate CSV written to: {summary_csv}")


if __name__ == "__main__":
    main()