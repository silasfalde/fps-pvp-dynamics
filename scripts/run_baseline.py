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

    # Export trace for visualization
    out = Path(ROOT) / "trace.json"
    model.export_trace_json(str(out))
    print(f"Trace exported to: {out}")

    metrics_csv = Path(ROOT) / "metrics.csv"
    model.metrics.export_csv(metrics_csv)
    print(f"Metrics exported to: {metrics_csv}")


if __name__ == "__main__":
    main()
