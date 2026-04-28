from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

from .agent import PlayerAgent
from .combat import damage_roll, hit_probability
from .config import SimulationConfig
from .environment import GridEnvironment
from .metrics import MetricStore, TickMetrics
from .types import Action, Terrain

Coord = Tuple[int, int]


class FpsPvpModel:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.tick = 0

        self.env = GridEnvironment(config.width, config.height)
        self._init_map()

        self.objective_pos = (config.width // 2, config.height // 2)
        self.env.set_terrain(self.objective_pos, Terrain.OBJECTIVE)

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
        # Sparse random walls as a placeholder for map templates.
        for x in range(self.config.width):
            for y in range(self.config.height):
                if self.rng.random() < 0.07:
                    self.env.set_terrain((x, y), Terrain.WALL)

    def _init_spawns(self) -> None:
        for pos in self.spawn_points.values():
            self.env.set_terrain(pos, Terrain.SPAWN)

    def _init_agents(self) -> List[PlayerAgent]:
        agents: List[PlayerAgent] = []
        per_team = self.config.n_agents // self.config.n_teams
        next_id = 0
        for team_id in range(self.config.n_teams):
            spawn = self.spawn_points[team_id]
            for _ in range(per_team):
                agents.append(
                    PlayerAgent(
                        agent_id=next_id,
                        team_id=team_id,
                        pos=spawn,
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

            enemies = [a.pos for a in self.agents if a.team_id != agent.team_id and a.alive]
            has_enemy = agent.visible_enemies(enemies, self.config.detection_radius)
            on_objective = agent.pos == self.objective_pos
            action = agent.decide_action(has_enemy, on_objective)

            if action == Action.ENGAGE:
                continue
            self._move_agent(agent, push=(action == Action.PUSH_OBJECTIVE))

        self.env.rebuild_occupancy({a.agent_id: a.pos for a in self.agents if a.alive})

    def _move_agent(self, agent: PlayerAgent, push: bool) -> None:
        candidates = [p for p in self.env.neighbors_moore(agent.pos) if self.env.traversable(p)]
        if not candidates:
            return

        old_pos = agent.pos
        if push:
            candidates.sort(key=lambda p: abs(p[0] - self.objective_pos[0]) + abs(p[1] - self.objective_pos[1]))
            new_pos = candidates[0]
            agent.pos = new_pos
        else:
            new_pos = self.rng.choice(candidates)
            agent.pos = new_pos

        # Update facing as simple grid delta (new - old)
        dx = agent.pos[0] - old_pos[0]
        dy = agent.pos[1] - old_pos[1]
        if dx == 0 and dy == 0:
            # keep existing facing
            return
        agent.facing = (dx, dy)

    def _resolve_combat(self) -> None:
        for attacker in self.agents:
            if not attacker.alive or attacker.cooldown_timer > 0:
                continue

            enemies = [a for a in self.agents if a.team_id != attacker.team_id and a.alive]
            if not enemies:
                continue

            target = min(enemies, key=lambda e: abs(e.pos[0] - attacker.pos[0]) + abs(e.pos[1] - attacker.pos[1]))
            distance = abs(target.pos[0] - attacker.pos[0]) + abs(target.pos[1] - attacker.pos[1])

            p_hit = hit_probability(attacker.skill, float(distance), self.config.weapon_range, self.config.stochasticity)
            if self.rng.random() <= p_hit:
                dmg = damage_roll(self.config.base_damage, self.config.stochasticity)
                target.health -= dmg
                self.env.cell(target.pos).combat_heat += 1.0
                if target.health <= 0:
                    target.alive = False
                    target.deaths += 1
                    target.respawn_timer = self.config.respawn_delay
                    attacker.kills += 1

            attacker.cooldown_timer = self.config.weapon_cooldown_ticks

    def _update_objective(self) -> None:
        team_counts: Dict[int, int] = {}
        for agent in self.agents:
            if agent.alive and agent.pos == self.objective_pos:
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
                    "alive": a.alive,
                }
                for a in self.agents
            ],
        }
        self.trace.append(frame)

    def export_trace_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trace, f)
