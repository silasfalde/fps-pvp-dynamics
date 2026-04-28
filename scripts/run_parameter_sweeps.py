"""Run small parameter sweeps and export summary metrics to CSV.

Usage:
    python3 scripts/run_parameter_sweeps.py

Produces files under `out/experiments/`.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fps_pvp_abm import FpsPvpModel, SimulationConfig

OUT_DIR = ROOT / "out" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_sweep() -> Path:
    rows: list[dict[str, float | int | str]] = []
    sweep_values = {
        "objective_radius": [5, 7, 9],
        "wall_prob": [0.03, 0.07, 0.15],
        "weapon_cooldown_ticks": [1, 2, 3],
    }
    seed = 531

    for param_name, values in sweep_values.items():
        for value in values:
            config_kwargs = {"max_ticks": 160, "n_agents": 30, "seed": seed}
            config_kwargs[param_name] = value
            config = SimulationConfig(**config_kwargs)
            model = FpsPvpModel(config)
            summary = model.run()
            row = {"parameter": param_name, "value": value}
            row.update(summary)
            rows.append(row)

    outpath = OUT_DIR / "parameter_sweeps.csv"
    with outpath.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["parameter", "value", "ticks", "objective_progress", "objective_controller"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return outpath


def main() -> None:
    outpath = run_sweep()
    print(f"Sweep results written to: {outpath}")


if __name__ == "__main__":
    main()