# FPS PvP ABM Implementation Plan

This document translates the proposal in `proposal/proposal.tex` into an executable implementation roadmap.

## 1. MVP Scope (First Valid Model)

- 2D grid map with terrain tags: `open`, `cover`, `choke`, `spawn`, `objective`, `wall`.
- Two teams of heterogeneous player agents.
- Tick schedule:
  1. Sense
  2. Decide
  3. Move
  4. Resolve Combat
  5. Update Objectives + Environment Fields
  6. Adapt + Respawn
  7. Record Metrics
- Stochastic combat outcome from skill, weapon profile, and distance.
- Objective control and win-condition progress tracking.
- Per-tick metrics logging and per-match summary output.

## 2. Package Layout

- `src/fps_pvp_abm/config.py`: dataclasses for global parameters.
- `src/fps_pvp_abm/environment.py`: map/grid representation and dynamic fields.
- `src/fps_pvp_abm/agent.py`: player state and decision/action hooks.
- `src/fps_pvp_abm/combat.py`: hit and damage probability utilities.
- `src/fps_pvp_abm/metrics.py`: metric collection and match summaries.
- `src/fps_pvp_abm/model.py`: scheduler and simulation orchestration.
- `scripts/run_baseline.py`: baseline run entrypoint.

## 3. Phase Plan

### Phase A: Core Loop (implemented in scaffold)

- Build discrete tick loop and state containers.
- Implement movement constraints, basic target selection, and elimination/respawn.
- Add objective progress updates and minimal metric recording.

### Phase B: Tactical Behavior

- Add risk-aware movement using cover and combat pressure fields.
- Add richer action scoring (`engage`, `push`, `hold`, `retreat`).
- Implement weapon selection adaptation from recent outcomes.

### Phase C: Experiments and Analysis

- Add parameter sweep runner (one-factor and factorial settings).
- Add replication support with deterministic seeds.
- Export metric tables and summary plots.

### Phase D: Calibration and Validation

- Connect parameter priors to empirical FPS stats.
- Validate macro-patterns: fairness, tempo, dominance, and contest intensity.

## 4. Experiment Backlog

- Team assignment: random vs skill-balanced.
- Map variants: cover/choke density and spawn layouts.
- Weapon multipliers: damage/range/cooldown sweeps.
- Objective pacing: capture rate and respawn delay.
- Adaptation rate sweeps with stability checks.

## 5. Near-Term Tasks

1. Replace random movement heuristic with shortest-path objective utility.
2. Add line-of-sight checks from wall/cover geometry.
3. Add per-weapon stats and cooldown windows.
4. Add CSV logging and plotting notebook/script.
5. Add tests for combat, respawn, and objective updates.
