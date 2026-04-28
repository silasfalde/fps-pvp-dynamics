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
    objective_radius: int = 5
    detection_radius: int = 7
    base_damage: float = 35.0
    weapon_range: float = 8.0
    weapon_cooldown_ticks: int = 2
    stochasticity: float = 0.20
    adaptation_rate: float = 0.05
    seed: int = 531
    # New behavior/environment tunables
    los_enabled: bool = True
    wall_prob: float = 0.07
    cover_prob: float = 0.04
    choke_prob: float = 0.02
    shot_flash_duration: int = 2
    death_marker_duration: int = 10
