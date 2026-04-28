from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .agent import PlayerAgent
from .combat import damage_roll, hit_probability
from .config import SimulationConfig
from .environment import GridEnvironment
from .metrics import MetricStore, TickMetrics
from .types import Action, Terrain

Coord = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class WeaponProfile:
    name: str
    damage: float
    range: float
    cooldown_ticks: int


class FpsPvpModel:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.tick = 0
        self.objective_radius = config.objective_radius

        self.env = GridEnvironment(config.width, config.height)
        self._init_map()

        self.objective_pos = (config.width // 2, config.height // 2)
        self._init_objective_zone()
        self._objective_distance_field = self._build_objective_distance_field()
        self.weapon_profiles = self._build_weapon_profiles()

        self.spawn_points: Dict[int, Coord] = {
            0: (1, 1),
            1: (config.width - 2, config.height - 2),
        }
        self._init_spawns()

        self.agents = self._init_agents()
        self.metrics = MetricStore()
        # In-memory per-tick trace (list of frames)
        self.trace: List[Dict] = []

    def _init_map(self) -> None:
        # Sparse random walls, cover, and choke points.
        for x in range(self.config.width):
            for y in range(self.config.height):
                r = self.rng.random()
                # walls have baseline density
                if r < self.config.wall_prob:
                    self.env.set_terrain((x, y), Terrain.WALL)
                else:
                    # cover and choke are rarer
                    if r < 0.07 + self.config.cover_prob:
                        self.env.set_terrain((x, y), Terrain.COVER)
                    elif r < 0.07 + self.config.cover_prob + self.config.choke_prob:
                        self.env.set_terrain((x, y), Terrain.CHOKE)

    def _init_spawns(self) -> None:
        for pos in self.spawn_points.values():
            self.env.set_terrain(pos, Terrain.SPAWN)

    def _init_objective_zone(self) -> None:
        ox, oy = self.objective_pos
        radius_sq = self.objective_radius * self.objective_radius
        for x in range(self.config.width):
            for y in range(self.config.height):
                if (x - ox) * (x - ox) + (y - oy) * (y - oy) <= radius_sq:
                    self.env.set_terrain((x, y), Terrain.OBJECTIVE)

    def _init_agents(self) -> List[PlayerAgent]:
        agents: List[PlayerAgent] = []
        per_team = self.config.n_agents // self.config.n_teams
        next_id = 0
        weapon_names = list(self.weapon_profiles)
        for team_id in range(self.config.n_teams):
            spawn = self.spawn_points[team_id]
            for _ in range(per_team):
                agents.append(
                    PlayerAgent(
                        agent_id=next_id,
                        team_id=team_id,
                        pos=spawn,
                        weapon_name=self.rng.choice(weapon_names),
                        skill=self.rng.uniform(0.2, 0.95),
                        aggression=self.rng.uniform(0.0, 1.0),
                        risk_tolerance=self.rng.uniform(0.0, 1.0),
                    )
                )
                next_id += 1
        return agents

    def run(self) -> Dict[str, float]:
        for _ in range(self.config.max_ticks):
            self.step()
        return self.metrics.summary()

    def step(self) -> None:
        self.tick += 1

        # reset per-tick events list for visualization and analysis
        self._current_events: List[Dict] = []

        self._sense_decide_move()
        self._resolve_combat()
        self._update_objective()
        self._adapt_and_respawn()
        self._record_metrics()

    def _sense_decide_move(self) -> None:
        for agent in self.agents:
            agent.tick_timers()
            if not agent.alive:
                continue
            enemies = [a for a in self.agents if a.team_id != agent.team_id and a.alive]
            # visible enemy means within detection radius and (if enabled) line-of-sight
            has_enemy = False
            for e in enemies:
                dist = abs(agent.pos[0] - e.pos[0]) + abs(agent.pos[1] - e.pos[1])
                if dist <= self.config.detection_radius:
                    if self.config.los_enabled:
                        if self._has_line_of_sight(agent.pos, e.pos):
                            has_enemy = True
                            break
                    else:
                        has_enemy = True
                        break
            # update cover flag from environment
            agent.in_cover = self.env.cell(agent.pos).terrain == Terrain.COVER

            on_objective = self._is_on_objective(agent.pos)
            action = agent.decide_action(has_enemy, on_objective, agent.in_cover)

            if action == Action.ENGAGE:
                continue
            self._move_agent(agent, push=(action == Action.PUSH_OBJECTIVE), retreat=(action == Action.RETREAT))

        self.env.rebuild_occupancy({a.agent_id: a.pos for a in self.agents if a.alive})

    def _move_agent(self, agent: PlayerAgent, push: bool, retreat: bool = False) -> None:
        candidates = [p for p in self.env.neighbors_moore(agent.pos) if self.env.traversable(p)]
        if not candidates:
            return

        old_pos = agent.pos
        if push:
            new_pos = min(candidates, key=self._objective_candidate_key)
            agent.pos = new_pos
        elif retreat:
            new_pos = max(candidates, key=self._retreat_candidate_key)
            agent.pos = new_pos
        else:
            new_pos = min(candidates, key=self._reposition_candidate_key)
            agent.pos = new_pos

        # Update facing as simple grid delta (new - old)
        dx = agent.pos[0] - old_pos[0]
        dy = agent.pos[1] - old_pos[1]
        if dx == 0 and dy == 0:
            # keep existing facing
            return
        agent.facing = (dx, dy)

    def _is_on_objective(self, pos: Coord) -> bool:
        ox, oy = self.objective_pos
        return (pos[0] - ox) * (pos[0] - ox) + (pos[1] - oy) * (pos[1] - oy) <= self.objective_radius * self.objective_radius

    def _build_objective_distance_field(self) -> Dict[Coord, int]:
        distances: Dict[Coord, int] = {}
        queue = deque([self.objective_pos])
        distances[self.objective_pos] = 0
        while queue:
            pos = queue.popleft()
            base_distance = distances[pos]
            for neighbor in self.env.neighbors_moore(pos):
                if neighbor in distances:
                    continue
                if not self.env.traversable(neighbor):
                    continue
                distances[neighbor] = base_distance + 1
                queue.append(neighbor)
        return distances

    def _build_weapon_profiles(self) -> Dict[str, WeaponProfile]:
        return {
            "rifle": WeaponProfile(
                name="rifle",
                damage=self.config.base_damage,
                range=self.config.weapon_range,
                cooldown_ticks=self.config.weapon_cooldown_ticks,
            ),
            "smg": WeaponProfile(
                name="smg",
                damage=self.config.base_damage * 0.85,
                range=self.config.weapon_range * 0.8,
                cooldown_ticks=max(1, self.config.weapon_cooldown_ticks - 1),
            ),
            "dmr": WeaponProfile(
                name="dmr",
                damage=self.config.base_damage * 1.1,
                range=self.config.weapon_range * 1.25,
                cooldown_ticks=self.config.weapon_cooldown_ticks + 1,
            ),
        }

    def _objective_distance(self, pos: Coord) -> int:
        return self._objective_distance_field.get(pos, abs(pos[0] - self.objective_pos[0]) + abs(pos[1] - self.objective_pos[1]))

    def _objective_candidate_key(self, pos: Coord) -> tuple[int, float, int]:
        cell = self.env.cell(pos)
        return (
            self._objective_distance(pos),
            cell.combat_heat,
            0 if cell.terrain == Terrain.COVER else 1,
        )

    def _reposition_candidate_key(self, pos: Coord) -> tuple[int, float, int, int]:
        cell = self.env.cell(pos)
        return (
            self._objective_distance(pos),
            cell.combat_heat,
            0 if cell.terrain == Terrain.COVER else 1,
            len(cell.occupant_ids),
        )

    def _retreat_candidate_key(self, pos: Coord) -> tuple[int, int, float, int]:
        cell = self.env.cell(pos)
        return (
            self._objective_distance(pos),
            0 if cell.terrain == Terrain.COVER else 1,
            cell.combat_heat,
            -len(cell.occupant_ids),
        )

    def _has_line_of_sight(self, a: Coord, b: Coord) -> bool:
        # Bresenham-like integer line between a and b — block on WALL
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx // 2
            while x != x1:
                if self.env.cell((x, y)).terrain == Terrain.WALL:
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy // 2
            while y != y1:
                if self.env.cell((x, y)).terrain == Terrain.WALL:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        # check final cell
        if self.env.cell((x1, y1)).terrain == Terrain.WALL:
            return False
        return True

    def _resolve_combat(self) -> None:
        for attacker in self.agents:
            if not attacker.alive or attacker.cooldown_timer > 0:
                continue
            enemies = [a for a in self.agents if a.team_id != attacker.team_id and a.alive]
            if not enemies:
                continue
            profile = self.weapon_profiles[attacker.weapon_name]

            # target nearest enemy
            target = min(enemies, key=lambda e: abs(e.pos[0] - attacker.pos[0]) + abs(e.pos[1] - attacker.pos[1]))
            distance = abs(target.pos[0] - attacker.pos[0]) + abs(target.pos[1] - attacker.pos[1])

            # check LOS if enabled
            if self.config.los_enabled and not self._has_line_of_sight(attacker.pos, target.pos):
                # cannot fire if no LOS
                continue

            # mark shot attempt
            p_hit = hit_probability(attacker.skill, float(distance), profile.range, self.config.stochasticity)
            hit = False
            if self.rng.random() <= p_hit:
                dmg = damage_roll(profile.damage, self.config.stochasticity)
                target.health -= dmg
                self.env.cell(target.pos).combat_heat += 1.0
                hit = True
                if target.health <= 0:
                    target.alive = False
                    target.deaths += 1
                    target.respawn_timer = self.config.respawn_delay
                    attacker.kills += 1
                    target.last_death_tick = self.tick

            # record shot event
            self._current_events.append(
                {
                    "type": "shot",
                    "attacker": attacker.agent_id,
                    "target": target.agent_id,
                    "from": attacker.pos,
                    "to": target.pos,
                    "hit": hit,
                    "weapon": attacker.weapon_name,
                }
            )

            if hit and not target.alive:
                self._current_events.append({"type": "death", "agent": target.agent_id, "pos": target.pos})

            attacker.last_shot_tick = self.tick

            # simple retreat behavior on miss for low risk tolerance
            if not hit and attacker.risk_tolerance < 0.3:
                # choose neighbor that maximizes distance to target
                candidates = [p for p in self.env.neighbors_moore(attacker.pos) if self.env.traversable(p)]
                if candidates:
                    candidates.sort(key=lambda p: -(abs(p[0] - target.pos[0]) + abs(p[1] - target.pos[1])))
                    attacker.pos = candidates[0]

            attacker.cooldown_timer = profile.cooldown_ticks

    def _update_objective(self) -> None:
        team_counts: Dict[int, int] = {}
        for agent in self.agents:
            if agent.alive and self._is_on_objective(agent.pos):
                team_counts[agent.team_id] = team_counts.get(agent.team_id, 0) + 1

        obj_cell = self.env.cell(self.objective_pos)
        if not team_counts:
            obj_cell.objective_progress = max(0.0, obj_cell.objective_progress - 0.01)
            self.env.decay_combat_heat()
            return

        leader_team = max(team_counts, key=lambda team_id: team_counts[team_id])
        if len(team_counts) > 1:
            obj_cell.objective_progress = max(0.0, obj_cell.objective_progress - 0.02)
        else:
            if obj_cell.objective_controller == leader_team:
                obj_cell.objective_progress = min(1.0, obj_cell.objective_progress + self.config.objective_capture_rate)
            else:
                obj_cell.objective_progress -= self.config.objective_capture_rate
                if obj_cell.objective_progress <= 0.0:
                    obj_cell.objective_controller = leader_team
                    obj_cell.objective_progress = self.config.objective_capture_rate

        self.env.decay_combat_heat()

    def _adapt_and_respawn(self) -> None:
        for agent in self.agents:
            if agent.alive:
                if agent.kills > agent.deaths:
                    agent.aggression = min(1.0, agent.aggression + self.config.adaptation_rate)
                continue
            if agent.respawn_timer == 0:
                spawn = self.spawn_points[agent.team_id]
                agent.reset_for_respawn(spawn)
                agent.weapon_name = self._choose_weapon_for_agent(agent)

    def _choose_weapon_for_agent(self, agent: PlayerAgent) -> str:
        if agent.skill >= 0.75:
            return "dmr"
        if agent.aggression >= 0.65:
            return "smg"
        return "rifle"

    def _record_metrics(self) -> None:
        alive_by_team: Dict[int, int] = {}
        kills_by_team: Dict[int, int] = {}
        for a in self.agents:
            alive_by_team[a.team_id] = alive_by_team.get(a.team_id, 0) + int(a.alive)
            kills_by_team[a.team_id] = kills_by_team.get(a.team_id, 0) + a.kills

        obj = self.env.cell(self.objective_pos)
        self.metrics.add(
            TickMetrics(
                tick=self.tick,
                alive_by_team=alive_by_team,
                kills_by_team=kills_by_team,
                objective_controller=obj.objective_controller,
                objective_progress=obj.objective_progress,
            )
        )
        # Append compact trace frame for visualization/export
        frame = {
            "tick": self.tick,
            "agents": [
                {
                    "id": a.agent_id,
                    "team": a.team_id,
                    "pos": a.pos,
                    "facing": a.facing,
                    "weapon": a.weapon_name,
                    "alive": a.alive,
                }
                for a in self.agents
            ],
            "events": getattr(self, "_current_events", []),
        }
        self.trace.append(frame)

    def export_trace_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trace, f)
