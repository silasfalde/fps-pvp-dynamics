"""Plot match metrics exported by `scripts/run_baseline.py`.

Usage:
    python3 scripts/plot_metrics.py metrics.csv

Writes a PNG plot next to the CSV input.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 scripts/plot_metrics.py <metrics.csv>")

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    ticks = [int(row["tick"]) for row in rows]
    objective_progress = [float(row["objective_progress"]) for row in rows]
    alive_team_0 = [int(row.get("alive_team_0", 0)) for row in rows]
    alive_team_1 = [int(row.get("alive_team_1", 0)) for row in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ticks, objective_progress, color="#2a6fdb", linewidth=2, label="Objective progress")
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Objective progress", color="#2a6fdb")
    ax1.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2a6fdb")

    ax2 = ax1.twinx()
    ax2.plot(ticks, alive_team_0, color="#1f77b4", alpha=0.55, label="Team 0 alive")
    ax2.plot(ticks, alive_team_1, color="#d62728", alpha=0.55, label="Team 1 alive")
    ax2.set_ylabel("Alive agents", color="#444444")
    ax2.tick_params(axis="y", labelcolor="#444444")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title("FPS PvP Match Metrics")
    fig.tight_layout()

    outpath = csv_path.with_suffix(".png")
    fig.savefig(outpath, dpi=160)
    print(f"Saved plot to: {outpath}")


if __name__ == "__main__":
    main()