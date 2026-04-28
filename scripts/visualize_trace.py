"""Simple Matplotlib animator for trace.json produced by the model.

Usage:
    python scripts/visualize_trace.py trace.json

This will open a Matplotlib window animating agent positions and facings.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_trace(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(vx, vy):
    mag = math.hypot(vx, vy)
    if mag == 0:
        return 0.0, 0.0
    return vx / mag, vy / mag


def animate(trace, width=40, height=30):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    ax.set_aspect('equal')
    scatter = ax.scatter([], [], s=60)
    arrows: list = []

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return (scatter,)

    def update(frame):
        agents = frame['agents']
        xs = [a['pos'][0] for a in agents]
        ys = [a['pos'][1] for a in agents]
        colors = [a['team'] for a in agents]
        us = []
        vs = []
        for a in agents:
            dx, dy = a.get('facing', (0, 1))
            nx, ny = normalize(dx, dy)
            us.append(nx)
            vs.append(ny)

        scatter.set_offsets(np.column_stack((xs, ys)))
        scatter.set_array(np.array(colors))

        # remove old arrows
        nonlocal arrows
        for a in arrows:
            try:
                a.remove()
            except Exception:
                pass
        arrows = []

        # draw new arrows for alive agents
        for x, y, u, v, agent in zip(xs, ys, us, vs, agents):
            if not agent.get('alive', True):
                continue
            # scale arrow for visibility
            ax_a = ax.arrow(x, y, u * 0.6, v * 0.6, head_width=0.3, head_length=0.3, fc='k', ec='k')
            arrows.append(ax_a)

        ax.set_title(f"Tick: {frame['tick']}")
        return (scatter, *arrows)

    ani = animation.FuncAnimation(fig, update, frames=trace, init_func=init, blit=False, interval=120)
    plt.colorbar(scatter, ax=ax, label='team')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path, help='Path to trace.json')
    args = parser.parse_args()
    trace = load_trace(args.trace)
    # infer grid size from trace if possible
    width = 40
    height = 30
    if trace:
        # try to find max coords
        maxx = max((a['pos'][0] for f in trace for a in f['agents']), default=width)
        maxy = max((a['pos'][1] for f in trace for a in f['agents']), default=height)
        width = max(width, maxx + 2)
        height = max(height, maxy + 2)
    animate(trace, width=width, height=height)


if __name__ == '__main__':
    main()
