from __future__ import annotations

from dataclasses import dataclass, field
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
