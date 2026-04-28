from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import Action

Coord = Tuple[int, int]


@dataclass(slots=True)
class PlayerAgent:
    agent_id: int
    team_id: int
    pos: Coord
    skill: float
    aggression: float
    risk_tolerance: float
    facing: Coord = (0, 1)
    health: float = 100.0
    alive: bool = True
    respawn_timer: int = 0
    cooldown_timer: int = 0
    kills: int = 0
    deaths: int = 0

    def tick_timers(self) -> None:
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        if not self.alive and self.respawn_timer > 0:
            self.respawn_timer -= 1

    def visible_enemies(self, enemy_positions: List[Coord], detection_radius: int) -> bool:
        px, py = self.pos
        for ex, ey in enemy_positions:
            if abs(px - ex) + abs(py - ey) <= detection_radius:
                return True
        return False

    def decide_action(self, has_visible_enemy: bool, on_objective: bool) -> Action:
        if not self.alive:
            return Action.HOLD_DEFEND
        if has_visible_enemy and self.cooldown_timer == 0:
            return Action.ENGAGE
        if on_objective:
            return Action.HOLD_DEFEND
        if self.aggression > 0.5:
            return Action.PUSH_OBJECTIVE
        return Action.REPOSITION

    def reset_for_respawn(self, spawn_pos: Coord) -> None:
        self.pos = spawn_pos
        self.health = 100.0
        self.alive = True
        self.cooldown_timer = 0
        self.facing = (0, 1)
