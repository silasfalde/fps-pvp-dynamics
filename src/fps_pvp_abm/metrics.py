from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(slots=True)
class TickMetrics:
    tick: int
    alive_by_team: Dict[int, int]
    kills_by_team: Dict[int, int]
    objective_controller: int | None
    objective_progress: float


@dataclass(slots=True)
class MetricStore:
    ticks: List[TickMetrics] = field(default_factory=list)

    def add(self, metric: TickMetrics) -> None:
        self.ticks.append(metric)

    def summary(self) -> Dict[str, float]:
        if not self.ticks:
            return {}
        last = self.ticks[-1]
        return {
            "ticks": float(len(self.ticks)),
            "objective_progress": last.objective_progress,
            "objective_controller": float(last.objective_controller) if last.objective_controller is not None else -1.0,
        }

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ticks:
            path.write_text("tick\n", encoding="utf-8")
            return

        team_ids = sorted(
            {
                team_id
                for metric in self.ticks
                for team_id in metric.alive_by_team
            }
            | {
                team_id
                for metric in self.ticks
                for team_id in metric.kills_by_team
            }
        )
        fieldnames = ["tick", "objective_controller", "objective_progress"]
        fieldnames.extend(f"alive_team_{team_id}" for team_id in team_ids)
        fieldnames.extend(f"kills_team_{team_id}" for team_id in team_ids)

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.ticks:
                row = {
                    "tick": metric.tick,
                    "objective_controller": metric.objective_controller if metric.objective_controller is not None else -1,
                    "objective_progress": metric.objective_progress,
                }
                for team_id in team_ids:
                    row[f"alive_team_{team_id}"] = metric.alive_by_team.get(team_id, 0)
                    row[f"kills_team_{team_id}"] = metric.kills_by_team.get(team_id, 0)
                writer.writerow(row)
