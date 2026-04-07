from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Terrain

Coord = Tuple[int, int]


@dataclass(slots=True)
class Cell:
    terrain: Terrain = Terrain.OPEN
    occupant_ids: List[int] = field(default_factory=list)
    objective_progress: float = 0.0
    objective_controller: Optional[int] = None
    combat_heat: float = 0.0


class GridEnvironment:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid = [[Cell() for _ in range(height)] for _ in range(width)]

    def in_bounds(self, pos: Coord) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors_moore(self, pos: Coord) -> List[Coord]:
        x, y = pos
        neighbors: List[Coord] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (x + dx, y + dy)
                if self.in_bounds(nxt):
                    neighbors.append(nxt)
        return neighbors

    def traversable(self, pos: Coord) -> bool:
        cell = self.cell(pos)
        return cell.terrain != Terrain.WALL

    def cell(self, pos: Coord) -> Cell:
        return self.grid[pos[0]][pos[1]]

    def set_terrain(self, pos: Coord, terrain: Terrain) -> None:
        self.cell(pos).terrain = terrain

    def clear_occupancy(self) -> None:
        for col in self.grid:
            for cell in col:
                cell.occupant_ids.clear()

    def rebuild_occupancy(self, positions: Dict[int, Coord]) -> None:
        self.clear_occupancy()
        for agent_id, pos in positions.items():
            self.cell(pos).occupant_ids.append(agent_id)

    def decay_combat_heat(self, factor: float = 0.95) -> None:
        for col in self.grid:
            for cell in col:
                cell.combat_heat *= factor
