from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fps_pvp_abm import FpsPvpModel, SimulationConfig


def main() -> None:
    config = SimulationConfig(max_ticks=200, n_agents=30)
    model = FpsPvpModel(config)
    summary = model.run()
    print("Baseline summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
