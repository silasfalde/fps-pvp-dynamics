# fps-pvp-dynamics

Agent-based simulation and analysis of first-person shooter player-versus-player dynamics for balancing, experimentation, and visualization.

This project models a compact multiplayer FPS match on a grid map, then captures enough state to analyze combat, objective control, movement choices, and tuning sensitivity. The codebase is structured to support both simulation research and presentation-ready outputs such as traces, metrics, GIFs, and plots.

## What This Project Does

The simulator runs a two-team match on a 2D grid with the following behavior:

- agents spawn in team-specific locations and carry one of several weapon profiles
- each tick, agents sense nearby enemies, decide whether to engage, push, retreat, reposition, or hold
- combat is stochastic and depends on skill, distance, weapon profile, and random variation
- a central objective zone tracks capture progress and controller state
- the environment records occupancy, terrain, combat heat, and per-tick traces for later analysis
- agents respawn after a delay and adapt aggression based on performance

The repository also includes experiment scripts for sweeps, large replications, Latin hypercube sampling, and Sobol sensitivity analysis.

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a baseline simulation and export the main artifacts:

```bash
python3 scripts/run_baseline.py
```

Visualize the generated trace interactively:

```bash
python3 scripts/visualize_trace.py trace.json
```

Render example matches to GIFs and JSON traces:

```bash
python3 scripts/render_examples.py
```

Plot exported CSV metrics:

```bash
python3 scripts/plot_metrics.py metrics.csv
```

## Installation

The project is a standard Python source tree and does not require a package install step beyond dependencies.

Recommended setup:

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run scripts from the repository root so the local `src/` path is available.

The scripts add `src/` to `sys.path` automatically, so the project can be run directly without packaging it first.

## Usage Guide

### Baseline match

`scripts/run_baseline.py` runs a single simulation using the default configuration, prints a summary, and exports:

- `trace.json` at the repository root
- `metrics.csv` at the repository root

This is the fastest way to validate the simulator and generate a representative artifact set.

### Interactive trace visualization

`scripts/visualize_trace.py` loads a trace and opens a Matplotlib animation showing:

- agent positions as colored markers by team
- facing direction as arrows
- tick progression in the plot title

If the trace contains a larger map than the default configuration, the viewer infers the bounds from the data.

### Headless example rendering

`scripts/render_examples.py` runs several matches and writes both traces and GIFs to `out/videos/`.

This is useful for sharing the simulation visually without requiring a live plotting window.

### Metrics plotting

`scripts/plot_metrics.py` converts a metrics CSV into a PNG chart that overlays:

- objective progress
- alive agents per team

The output image is written next to the input CSV.

### Experiments and sensitivity analysis

The `scripts/` folder also includes analysis workflows that scale beyond a single match:

- `scripts/run_parameter_sweeps.py` runs targeted one-factor sweeps and exports a comparison CSV
- `scripts/run_large_replication_study.py` runs many independent replications and aggregates confidence intervals
- `scripts/experiments_pipeline.py` performs Latin hypercube sampling, parallel execution, and Sobol analysis

These scripts are designed for parameter exploration and balancing studies rather than just demo runs.

## Configuration

Simulation settings live in `src/fps_pvp_abm/config.py` and are exposed through `SimulationConfig`.

Important knobs include:

- map size and match length
- number of agents and teams
- respawn delay and objective capture rate
- detection radius, weapon range, and weapon cooldown
- stochasticity and adaptation rate
- line-of-sight toggling and terrain densities

Example:

```python
from fps_pvp_abm import FpsPvpModel, SimulationConfig

config = SimulationConfig(
	max_ticks=200,
	n_agents=30,
	seed=531,
	wall_prob=0.07,
	los_enabled=True,
)

model = FpsPvpModel(config)
summary = model.run()
print(summary)
```

## Outputs

The simulator produces several useful artifacts:

- `trace.json`: compact per-tick frames containing agent state and events
- `metrics.csv`: per-tick metrics with team counts and objective state
- GIF animations in `out/videos/`
- experiment summaries in `out/experiments/`
- analysis outputs in `out/analysis/`

The trace format is designed for visualization, while the metrics CSV is structured for plotting and downstream analysis.

## Implementation Details

### Core model

The main simulation loop is implemented in `src/fps_pvp_abm/model.py`.

Each tick executes a full update pipeline:

1. tick timers and evaluate agent perception
2. decide movement or engagement based on nearby enemies, objective state, and cover
3. resolve combat using weapon profile, distance, skill, and stochastic hit/damage logic
4. update objective control and progress
5. adapt agent behavior and respawn dead agents when timers expire
6. record metrics and append a trace frame

The model also maintains an in-memory event log per tick so renderers can show shots and deaths in context.

### Agent behavior

`src/fps_pvp_abm/agent.py` defines the `PlayerAgent` dataclass, including:

- team and position
- weapon selection and combat stats
- movement-facing state
- health, cooldowns, respawn timers, kills, and deaths
- simple decision logic for engagement, retreat, pushing, and holding

This keeps the behavioral policy separate from the environment and combat resolution.

### Environment representation

`src/fps_pvp_abm/environment.py` stores the map as a grid of cells with:

- terrain type
- occupant IDs
- combat heat
- objective progress
- objective controller

The environment tracks traversability, neighbor lookup, occupancy rebuilding, and combat-heat decay.

### Combat model

`src/fps_pvp_abm/combat.py` implements lightweight stochastic helpers:

- `hit_probability(...)` computes a hit chance from skill, distance, weapon range, and noise
- `damage_roll(...)` computes variable damage around a base value

These functions keep the combat math isolated and easy to tune.

### Metrics and trace capture

`src/fps_pvp_abm/metrics.py` collects per-tick summary data and can export it to CSV.

The recorded metrics include:

- alive counts by team
- kills by team
- objective controller
- objective progress

Trace frames store compact agent snapshots and event lists, which makes them suitable for animation and analysis without having to rerun the simulation.

### Experiment tooling

The analysis scripts are intentionally separated from the simulator so that the model can be reused across:

- baseline runs
- repeated replications
- targeted sweeps
- broader variance and sensitivity studies

The experiments pipeline uses multiprocessing to parallelize runs, then emits CSV and JSON summaries for downstream plotting or statistical review.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── metrics.csv
├── trace.json
├── proposal/
│   ├── model.py
│   ├── proposal.tex
│   └── supporting diagrams
├── report/
│   └── report.tex
├── scripts/
│   ├── experiments_pipeline.py
│   ├── plot_metrics.py
│   ├── render_examples.py
│   ├── run_baseline.py
│   ├── run_large_replication_study.py
│   ├── run_parameter_sweeps.py
│   └── visualize_trace.py
├── src/
│   └── fps_pvp_abm/
│       ├── __init__.py
│       ├── agent.py
│       ├── combat.py
│       ├── config.py
│       ├── environment.py
│       ├── metrics.py
│       ├── model.py
│       └── types.py
└── tests/
	└── test_model.py
```

## Testing

The current test suite uses `unittest` and focuses on core simulation behavior:

- combat resolves with shot and death events
- respawn restores agent state
- objective progress increases when a team controls the point

Run the tests with:

```bash
python3 -m unittest discover -s tests
```

## Skills Demonstrated

This project highlights practical experience with:

- agent-based modeling and simulation design
- game-system balancing and parameterization
- stochastic combat logic and stateful entity simulation
- grid-based environment modeling and occupancy tracking
- trace generation for visualization and debugging
- metrics capture and CSV export
- plotting, animation, and headless rendering
- multiprocessing for experiment workloads
- sensitivity analysis and reproducible simulation studies

## Notes

The repository currently emphasizes a readable and extensible research prototype. The next obvious improvements are richer pathfinding, stronger line-of-sight and cover modeling, more differentiated weapons, and additional downstream analysis of the experiment outputs.
