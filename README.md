# fps-pvp-dynamics

Modeling and analysis of First Person Shooter (FPS), Player vs. Player (PvP), large-scale video game balancing.

## Current Workflow

The project now includes a runnable baseline simulation, per-tick match traces, and headless example renderers.

### Run a baseline match

```bash
python3 scripts/run_baseline.py
```

This runs a full match and writes `trace.json` at the repository root.

### Visualize a trace interactively

```bash
python3 scripts/visualize_trace.py trace.json
```

This opens a Matplotlib animation that shows each agent as a dot with an arrow for facing direction.

### Render 3 example matches to files

```bash
python3 scripts/render_examples.py
```

This runs three example matches and writes trace files plus GIFs under `out/videos/`:

- `out/videos/trace_1.json`
- `out/videos/trace_2.json`
- `out/videos/trace_3.json`
- `out/videos/match_1.gif`
- `out/videos/match_2.gif`
- `out/videos/match_3.gif`

## Project Status

Implemented pieces include:

- `src/fps_pvp_abm/model.py`: core simulation loop, movement, combat, objective updates, metrics, and trace capture
- `src/fps_pvp_abm/agent.py`: player state, action selection, cooldowns, and facing direction
- `src/fps_pvp_abm/combat.py`: stochastic hit and damage helpers
- `src/fps_pvp_abm/environment.py`: grid map, occupancy, and terrain helpers
- `src/fps_pvp_abm/metrics.py`: per-tick metric collection
- `scripts/run_baseline.py`: baseline runner that exports `trace.json`
- `scripts/visualize_trace.py`: interactive trace viewer
- `scripts/render_examples.py`: headless renderer for three example matches

The next major implementation steps are more realistic movement/pathing, line-of-sight checks, richer weapon profiles, and experiment/analysis tooling.
