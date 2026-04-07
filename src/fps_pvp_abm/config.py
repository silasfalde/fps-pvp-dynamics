from dataclasses import dataclass


@dataclass(slots=True)
class SimulationConfig:
    width: int = 40
    height: int = 30
    n_agents: int = 40
    n_teams: int = 2
    max_ticks: int = 500
    respawn_delay: int = 5
    objective_capture_rate: float = 0.04
    detection_radius: int = 7
    base_damage: float = 35.0
    weapon_range: float = 8.0
    weapon_cooldown_ticks: int = 2
    stochasticity: float = 0.20
    adaptation_rate: float = 0.05
    seed: int = 531
