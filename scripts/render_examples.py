"""Run 3 example matches and save headless GIF animations.

Usage:
    python3 scripts/render_examples.py

Produces files under `out/videos/`.
"""
from pathlib import Path
import json
import math
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fps_pvp_abm import FpsPvpModel, SimulationConfig

OUT_DIR = ROOT / "out" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 960
CANVAS_H = 720
PADDING = 40
TEAM_COLORS = {
    0: (31, 119, 180),
    1: (214, 39, 40),
}


def normalize(vx: float, vy: float) -> tuple[float, float]:
    magnitude = math.hypot(vx, vy)
    if magnitude == 0:
        return 0.0, 0.0
    return vx / magnitude, vy / magnitude


def world_to_canvas(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    scale_x = (CANVAS_W - 2 * PADDING) / max(width - 1, 1)
    scale_y = (CANVAS_H - 2 * PADDING) / max(height - 1, 1)
    scale = min(scale_x, scale_y)
    canvas_x = PADDING + x * scale
    canvas_y = CANVAS_H - (PADDING + y * scale)
    return canvas_x, canvas_y


def draw_frame(frame: dict, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), (246, 244, 240))
    draw = ImageDraw.Draw(image)

    # Grid and axes
    for gx in range(width):
        x0, y0 = world_to_canvas(gx, 0, width, height)
        x1, y1 = world_to_canvas(gx, height - 1, width, height)
        draw.line((x0, y0, x1, y1), fill=(225, 222, 216), width=1)
    for gy in range(height):
        x0, y0 = world_to_canvas(0, gy, width, height)
        x1, y1 = world_to_canvas(width - 1, gy, width, height)
        draw.line((x0, y0, x1, y1), fill=(225, 222, 216), width=1)

    # Match header
    font = ImageFont.load_default()
    draw.text((20, 12), f"Tick {frame['tick']}", fill=(20, 20, 20), font=font)

    for agent in frame["agents"]:
        x, y = agent["pos"]
        fx, fy = agent.get("facing", (0, 1))
        fx, fy = normalize(float(fx), float(fy))
        cx, cy = world_to_canvas(float(x), float(y), width, height)
        color = TEAM_COLORS.get(agent["team"], (80, 80, 80))

        radius = 10 if agent.get("alive", True) else 7
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(30, 30, 30), width=2)

        if agent.get("alive", True) and (fx != 0.0 or fy != 0.0):
            arrow_len = 24
            end_x = cx + fx * arrow_len
            end_y = cy - fy * arrow_len
            draw.line((cx, cy, end_x, end_y), fill=(25, 25, 25), width=3)
            head_size = 6
            angle = math.atan2(cy - end_y, end_x - cx)
            left = (end_x - head_size * math.cos(angle - math.pi / 6), end_y + head_size * math.sin(angle - math.pi / 6))
            right = (end_x - head_size * math.cos(angle + math.pi / 6), end_y + head_size * math.sin(angle + math.pi / 6))
            draw.polygon((end_x, end_y, left[0], left[1], right[0], right[1]), fill=(25, 25, 25))

    return image


def save_gif(frames: list[Image.Image], outpath: Path, duration_ms: int = 90) -> None:
    first, *rest = frames
    first.save(
        outpath,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def run_and_render(seed: int, idx: int) -> None:
    config = SimulationConfig(max_ticks=200, n_agents=30, seed=seed)
    model = FpsPvpModel(config)
    print(f"Running match {idx} (seed={seed})...")
    model.run()
    trace = model.trace

    trace_path = OUT_DIR / f"trace_{idx}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)
    print(f"Saved trace to {trace_path}")

    frames = [draw_frame(frame, config.width, config.height) for frame in trace]
    outpath = OUT_DIR / f"match_{idx}.gif"
    save_gif(frames, outpath)
    print(f"Saved animation to {outpath}")


def main() -> None:
    seeds = [531, 532, 533]
    for idx, seed in enumerate(seeds, start=1):
        run_and_render(seed, idx)


if __name__ == "__main__":
    main()
