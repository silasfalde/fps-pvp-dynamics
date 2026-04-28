from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fps_pvp_abm import FpsPvpModel, SimulationConfig
from fps_pvp_abm.agent import PlayerAgent
from fps_pvp_abm.types import Terrain


class DummyRng:
    def random(self) -> float:
        return 0.0


class ModelBehaviorTests(unittest.TestCase):
    def test_combat_resolves_with_shot_and_death_events(self) -> None:
        config = SimulationConfig(
            width=10,
            height=10,
            n_agents=2,
            n_teams=2,
            max_ticks=1,
            respawn_delay=3,
            objective_radius=2,
            detection_radius=10,
            base_damage=200.0,
            weapon_range=20.0,
            weapon_cooldown_ticks=1,
            stochasticity=0.0,
            seed=7,
        )
        model = FpsPvpModel(config)
        model.env.set_terrain((1, 1), Terrain.OPEN)
        model.env.set_terrain((2, 1), Terrain.OPEN)
        attacker = PlayerAgent(0, 0, (1, 1), "rifle", 1.0, 0.5, 0.5)
        target = PlayerAgent(1, 1, (2, 1), "rifle", 1.0, 0.5, 0.5)
        model.agents = [attacker, target]
        model.rng = DummyRng()
        model._current_events = []

        model._resolve_combat()

        self.assertFalse(target.alive)
        self.assertEqual(attacker.kills, 1)
        self.assertEqual(target.deaths, 1)
        self.assertTrue(any(event["type"] == "shot" for event in model._current_events))
        self.assertTrue(any(event["type"] == "death" for event in model._current_events))

    def test_respawn_restores_agent_state(self) -> None:
        config = SimulationConfig(width=10, height=10, n_agents=2, n_teams=2, seed=11)
        model = FpsPvpModel(config)
        agent = model.agents[0]
        agent.alive = False
        agent.respawn_timer = 0
        agent.health = 0.0
        agent.weapon_name = "smg"

        model._adapt_and_respawn()

        self.assertTrue(agent.alive)
        self.assertEqual(agent.pos, model.spawn_points[agent.team_id])
        self.assertEqual(agent.health, 100.0)
        self.assertEqual(agent.cooldown_timer, 0)
        self.assertIn(agent.weapon_name, {"rifle", "smg", "dmr"})

    def test_objective_progress_increases_when_controlled(self) -> None:
        config = SimulationConfig(width=12, height=12, n_agents=2, n_teams=2, seed=13, objective_radius=4)
        model = FpsPvpModel(config)
        on_objective = model.objective_pos
        model.agents = [
            PlayerAgent(0, 0, on_objective, "rifle", 1.0, 0.5, 0.5),
            PlayerAgent(1, 1, (0, 0), "rifle", 1.0, 0.5, 0.5),
        ]
        cell = model.env.cell(model.objective_pos)
        cell.objective_controller = 0
        cell.objective_progress = 0.5

        model._update_objective()

        self.assertGreater(cell.objective_progress, 0.5)


if __name__ == "__main__":
    unittest.main()