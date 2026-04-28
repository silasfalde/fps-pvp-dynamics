"""Experiments pipeline: LHS sampling, parallel runs, Sobol analysis, and plotting.

Usage:
    python3 scripts/experiments_pipeline.py

Outputs:
    - out/samples/*.json (per-sample traces)
    - out/samples/metrics_*.csv (per-sample metrics)
    - out/analysis/lhs_uniformity.json (Kolmogorov-Smirnov uniformity statistics)
    - out/analysis/sobol_indices.png
    - out/analysis/sobol_results.json
    - out/experiments/lhs_summary.csv (LHS run results)
"""
from __future__ import annotations

import json
import math
import os
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Dict, List
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fps_pvp_abm import FpsPvpModel, SimulationConfig

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze
except Exception:
    saltelli = None
    sobol_analyze = None

OUT_SAMPLES = ROOT / "out" / "samples"
OUT_ANALYSIS = ROOT / "out" / "analysis"
OUT_EXPERIMENTS = ROOT / "out" / "experiments"
OUT_SAMPLES.mkdir(parents=True, exist_ok=True)
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
OUT_EXPERIMENTS.mkdir(parents=True, exist_ok=True)


PARAMETERS = {
    "objective_radius": (3.0, 12.0),
    "wall_prob": (0.0, 0.25),
    "cover_prob": (0.0, 0.2),
    "choke_prob": (0.0, 0.1),
    "stochasticity": (0.0, 0.5),
    "adaptation_rate": (0.0, 0.2),
    "weapon_cooldown_ticks": (1.0, 3.0),
}


def lhs_sampling(n_samples: int, param_bounds: Dict[str, tuple]) -> np.ndarray:
    k = len(param_bounds)
    rng = np.random.default_rng()
    # Stratified intervals
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random((n_samples, k))
    lhs = np.zeros((n_samples, k))
    for j in range(k):
        idx = rng.permutation(n_samples)
        lhs[:, j] = cut[:n_samples] + u[:, j] * (1.0 / n_samples)
        lhs[:, j] = lhs[idx, j]

    # map to bounds
    samples = np.zeros_like(lhs)
    names = list(param_bounds.keys())
    for j, name in enumerate(names):
        lo, hi = param_bounds[name]
        samples[:, j] = lhs[:, j] * (hi - lo) + lo
    return samples


def map_sample_to_config(sample: np.ndarray, param_names: List[str], seed_base: int, idx: int) -> dict:
    cfg = {
        "max_ticks": 200,
        "n_agents": 30,
        "seed": int(seed_base + idx),
    }
    for j, name in enumerate(param_names):
        val = sample[j]
        if name == "weapon_cooldown_ticks":
            cfg[name] = int(round(val))
        elif name in ("objective_radius",):
            cfg[name] = int(round(val))
        else:
            cfg[name] = float(val)
    return cfg


def worker_run(args):
    idx, sample, param_names, seed_base = args
    cfg_kwargs = map_sample_to_config(sample, param_names, seed_base, idx)
    config = SimulationConfig(**cfg_kwargs)
    model = FpsPvpModel(config)
    model.run()

    trace_path = OUT_SAMPLES / f"trace_{idx}.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(model.trace, f)

    metrics_path = OUT_SAMPLES / f"metrics_{idx}.csv"
    model.metrics.export_csv(metrics_path)

    summary = model.metrics.summary()
    summary.update(cfg_kwargs)
    summary_path = OUT_SAMPLES / f"summary_{idx}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f)

    return summary


def export_lhs_plot_csvs(samples: np.ndarray, param_names: List[str]) -> None:
    for column_index, name in enumerate(param_names[:3]):
        csv_name = f"lhs_{name}.csv"
        bins = 10 if name in ("wall_prob", "cover_prob") else 20
        counts, edges = np.histogram(samples[:, column_index], bins=bins)

        csv_path = OUT_ANALYSIS / csv_name
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["edge", "count"])
            for edge, count in zip(edges[:-1], counts):
                writer.writerow([float(edge), int(count)])
            writer.writerow([float(edges[-1]), 0])


def export_sobol_plot_csv(sobol_result: dict) -> None:
    labels = [
        "Adaptation Rate",
        "Choke Probability",
        "Cover Probability",
        "Objective Radius",
        "Stochasticity",
        "Wall Probability",
        "Weapon Cooldown Ticks",
    ]
    csv_path = OUT_ANALYSIS / "sobol_indices.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "s1", "s1_conf"])
        for label, value, conf in zip(labels, sobol_result["S1"], sobol_result["S1_conf"]):
            writer.writerow([label, float(value), float(conf)])


def run_lhs_and_save(n_samples: int = 100, n_workers: int = None, seed_base: int = 1000):
    param_names = list(PARAMETERS.keys())
    samples = lhs_sampling(n_samples, PARAMETERS)
    np.savetxt(OUT_ANALYSIS / "lhs_samples.csv", samples, delimiter=",", header=",".join(param_names), comments="")

    # Parallel run
    args = [(i, samples[i], param_names, seed_base) for i in range(len(samples))]
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(worker_run, args))

    keys = sorted(results[0].keys())
    out_csv = OUT_EXPERIMENTS / "lhs_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    export_lhs_plot_csvs(samples, param_names)

    # Compute uniformity statistics for each parameter using Kolmogorov-Smirnov test
    # KS statistic measures how well samples match uniform distribution (0 = perfect uniform, 1 = no overlap)
    uniformity_stats = {}
    for j, name in enumerate(param_names):
        lo, hi = PARAMETERS[name]
        # Normalize samples to [0, 1]
        normalized = (samples[:, j] - lo) / (hi - lo)
        ks_stat, ks_pval = stats.ks_1samp(normalized, stats.uniform.cdf)
        uniformity_stats[name] = {"ks_statistic": float(ks_stat), "ks_pvalue": float(ks_pval)}

    # Write uniformity statistics
    uniformity_json = OUT_ANALYSIS / "lhs_uniformity.json"
    with uniformity_json.open("w", encoding="utf-8") as f:
        json.dump(uniformity_stats, f, indent=2)

    return out_csv


def run_sobol_analysis(base_N: int = 64, n_workers: int = None, seed_base: int = 2000):
    if saltelli is None or sobol_analyze is None:
        raise RuntimeError("SALib is required for Sobol analysis. Install SALib in the environment.")

    names = list(PARAMETERS.keys())
    bounds = [list(PARAMETERS[n]) for n in names]
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    # Saltelli sampling
    param_values = saltelli.sample(problem, base_N, calc_second_order=False)
    # run model for each sample
    args = []
    for i in range(param_values.shape[0]):
        args.append((i, param_values[i], names, seed_base))

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(worker_run, args))

    # collect Y as objective_progress
    Y = np.array([r.get("objective_progress", 0.0) for r in results])

    Si = sobol_analyze.analyze(problem, Y, print_to_console=False)

    # save sobol results
    out_json = OUT_ANALYSIS / "sobol_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({k: v.tolist() if hasattr(v, "tolist") else v for k, v in Si.items()}, f)

    export_sobol_plot_csv(Si)

    # plot first-order indices
    S1 = Si.get("S1")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, S1)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylabel("S1 (first-order Sobol index)")
    ax.set_title("Sobol sensitivity (objective_progress)")
    fig.tight_layout()
    fig.savefig(OUT_ANALYSIS / "sobol_indices.png", dpi=150)

    return out_json


def main() -> None:
    start = time.time()
    print("Running LHS samples (n=1024)...")
    lhs_csv = run_lhs_and_save(n_samples=1024)
    print(f"LHS summary: {lhs_csv}")

    # Run Sobol analysis with larger base_N
    try:
        print("Running Sobol analysis (base_N=128)...")
        sobol_json = run_sobol_analysis(base_N=128)
        print(f"Sobol results: {sobol_json}")
    except RuntimeError as e:
        print("Skipping Sobol analysis:", e)

    elapsed = time.time() - start
    print(f"Pipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
