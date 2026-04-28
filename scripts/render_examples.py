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

TERRAIN_COLORS = {
    "wall": (70, 70, 74),
    "cover": (149, 167, 95),
    "choke": (203, 164, 108),
    "spawn": (203, 209, 222),
    "objective": (184, 220, 245),
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


def grid_scale(width: int, height: int) -> float:
    scale_x = (CANVAS_W - 2 * PADDING) / max(width - 1, 1)
    scale_y = (CANVAS_H - 2 * PADDING) / max(height - 1, 1)
    return min(scale_x, scale_y)


def cell_rect(x: int, y: int, width: int, height: int) -> tuple[float, float, float, float]:
    scale = grid_scale(width, height)
    cx, cy = world_to_canvas(float(x), float(y), width, height)
    half = scale / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def draw_frame(frame: dict, width: int, height: int, terrain_map: list[list[str]]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), (246, 244, 240))
    draw = ImageDraw.Draw(image)

    for x in range(width):
        for y in range(height):
            terrain = terrain_map[x][y]
            fill = TERRAIN_COLORS.get(terrain)
            if fill is None:
                continue
            draw.rectangle(cell_rect(x, y, width, height), fill=fill)

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


def run_and_render(seed: int, idx: int, wall_prob: float | None = None) -> None:
    cfg_kwargs = {"max_ticks": 200, "n_agents": 30, "seed": seed}
    if wall_prob is not None:
        cfg_kwargs["wall_prob"] = wall_prob
    config = SimulationConfig(**cfg_kwargs)
    model = FpsPvpModel(config)
    print(f"Running match {idx} (seed={seed})...")
    model.run()
    trace = model.trace
    terrain_map = [[cell.terrain.value for cell in col] for col in model.env.grid]

    trace_path = OUT_DIR / f"trace_{idx}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)
    print(f"Saved trace to {trace_path}")

    # build frames while tracking persistent death markers
    death_markers: list[dict] = []  # each: {pos:(x,y), ttl:int, color:(r,g,b)}
    shot_markers: list[dict] = []  # each: {pos:(x,y), ttl:int, color:(r,g,b)}
    frames_images = []
    # objective position in world coords
    obj_pos = (config.width // 2, config.height // 2)
    for fr in trace:
        # build agent id -> team map for this frame
        id2team = {a['id']: a['team'] for a in fr.get('agents', [])}

        # collect events
        for ev in fr.get("events", []):
            if ev.get("type") == "death":
                aid = ev.get("agent")
                team = id2team.get(aid, 0)
                color = TEAM_COLORS.get(team, (0, 0, 0))
                death_markers.append({"pos": tuple(ev.get("pos")), "ttl": config.death_marker_duration, "color": color})
            if ev.get("type") == "shot":
                attacker = ev.get("attacker")
                team = id2team.get(attacker, 0)
                # flash at attacker position; green for hit, red for miss but muted
                if ev.get("hit"):
                    color = (255, 220, 50)
                else:
                    color = (200, 100, 100)
                shot_markers.append({"pos": tuple(ev.get("from")), "ttl": config.shot_flash_duration, "color": color})

        # decrement TTL and filter death markers
        for m in list(death_markers):
            if m["ttl"] <= 0:
                death_markers.remove(m)
        # decrement TTL and filter shots
        for s in list(shot_markers):
            if s["ttl"] <= 0:
                shot_markers.remove(s)

        # draw frame with markers
        img = draw_frame(fr, config.width, config.height, terrain_map)
        draw = ImageDraw.Draw(img)

        # draw objective boundary
        ox, oy = obj_pos
        ocx, ocy = world_to_canvas(ox + 0.5, oy + 0.5, config.width, config.height)
        radius = grid_scale(config.width, config.height) * 5.5
        draw.ellipse((ocx - radius, ocy - radius, ocx + radius, ocy + radius), outline=(80, 80, 200), width=3)

        # draw persistent X markers (team-colored)
        for m in death_markers:
            cx, cy = world_to_canvas(m["pos"][0], m["pos"][1], config.width, config.height)
            size = 12
            draw.line((cx - size, cy - size, cx + size, cy + size), fill=m.get('color', (0, 0, 0)), width=3)
            draw.line((cx - size, cy + size, cx + size, cy - size), fill=m.get('color', (0, 0, 0)), width=3)
            m["ttl"] -= 1

        # draw muzzle/shot flashes
        for s in shot_markers:
            sx, sy = world_to_canvas(s["pos"][0], s["pos"][1], config.width, config.height)
            r = 6 + s["ttl"] * 2
            draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=s.get('color', (255, 220, 50)))
            s["ttl"] -= 1

        frames_images.append(img)

    outpath = OUT_DIR / f"match_{idx}.gif"
    save_gif(frames_images, outpath)
    print(f"Saved animation to {outpath}")


def main() -> None:
    # baseline runs
    seeds = [531, 532, 533]
    for idx, seed in enumerate(seeds, start=1):
        run_and_render(seed, idx)

    # obstacle-heavy runs
    run_and_render(600, 4, wall_prob=0.18)
    run_and_render(601, 5, wall_prob=0.25)


if __name__ == "__main__":
    main()
