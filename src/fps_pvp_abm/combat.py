from __future__ import annotations

import random


def hit_probability(skill: float, distance: float, weapon_range: float, stochasticity: float) -> float:
    range_penalty = max(0.0, (distance - weapon_range) / max(weapon_range, 1e-6))
    base = 0.25 + 0.65 * skill - 0.35 * range_penalty
    noisy = base + random.uniform(-stochasticity, stochasticity)
    return max(0.05, min(0.95, noisy))


def damage_roll(base_damage: float, stochasticity: float) -> float:
    return max(1.0, base_damage * (1.0 + random.uniform(-stochasticity, stochasticity)))
