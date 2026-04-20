"""Plot ablation study learning curves from results.json.

Usage:
    python scripts/plot_ablation.py --results experiments/exp_003_ablation_s6/results.json
    python scripts/plot_ablation.py --results experiments/exp_003_ablation_s6/results.json --output figures/ablation_curves.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["conditions"]


def _aggregate(conditions: list[dict]) -> dict[str, dict[str, list]]:
    """Group by condition name, collect timesteps and win rates across seeds."""
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in conditions:
        name = c["condition"]
        for entry in c.get("win_rate_history", []):
            groups[name]["timesteps"].append(entry["timestep"])
            groups[name]["win_rates"].append(entry["win_rate_vs_random"])
            groups[name]["avg_score"].append(entry["avg_score_p0"])
    return groups


def _mean_std_per_timestep(
    timesteps: list[int], values: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique_timesteps, means, stds) aggregated across seeds."""
    from collections import defaultdict
    buckets: dict[int, list[float]] = defaultdict(list)
    for t, v in zip(timesteps, values):
        buckets[t].append(v)
    ts = np.array(sorted(buckets.keys()))
    means = np.array([np.mean(buckets[t]) for t in ts])
    stds = np.array([np.std(buckets[t]) for t in ts])
    return ts, means, stds


CONDITION_LABELS = {
    "baseline": "PPO (baseline)",
    "bc_ppo":   "BC → PPO (ours)",
    "rag":      "PPO + RAG",
    "full":     "BC → PPO + RAG (full)",
}

CONDITION_COLORS = {
    "baseline": "#4878CF",
    "bc_ppo":   "#D65F5F",
    "rag":      "#6ACC65",
    "full":     "#B47CC7",
}


def plot_ablation(results_path: Path, output_path: Path) -> None:
    conditions = _load_results(results_path)
    groups = _aggregate(conditions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TabletopOracle — Ablation Study (Wingspan, 1M steps)", fontsize=13, fontweight="bold")

    # --- Panel 1: Win rate vs RandomAgent ---
    ax1 = axes[0]
    for name, data in groups.items():
        ts, means, stds = _mean_std_per_timestep(data["timesteps"], data["win_rates"])
        color = CONDITION_COLORS.get(name, "gray")
        label = CONDITION_LABELS.get(name, name)
        ax1.plot(ts / 1_000, means, color=color, label=label, linewidth=2, marker="o", markersize=5)
        ax1.fill_between(ts / 1_000, means - stds, means + stds, color=color, alpha=0.15)

    ax1.axhline(0.55, color="gray", linestyle="--", linewidth=1, label="Target (0.55)")
    ax1.set_xlabel("Training timesteps (×1 000)", fontsize=11)
    ax1.set_ylabel("Win rate vs. RandomAgent", fontsize=11)
    ax1.set_title("Sample Efficiency", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Average score P0 ---
    ax2 = axes[1]
    for name, data in groups.items():
        ts, means, stds = _mean_std_per_timestep(data["timesteps"], data["avg_score"])
        color = CONDITION_COLORS.get(name, "gray")
        label = CONDITION_LABELS.get(name, name)
        ax2.plot(ts / 1_000, means, color=color, label=label, linewidth=2, marker="o", markersize=5)
        ax2.fill_between(ts / 1_000, means - stds, means + stds, color=color, alpha=0.15)

    ax2.set_xlabel("Training timesteps (×1 000)", fontsize=11)
    ax2.set_ylabel("Average score (P0)", fontsize=11)
    ax2.set_title("Scoring Progression", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # --- Print summary table ---
    print("\n=== Final win rate (mean ± std across seeds) ===")
    print(f"{'Condition':<20}  {'WR vs Random':>14}  {'Avg Score P0':>14}")
    print("-" * 52)
    for name, data in groups.items():
        _, wr_means, wr_stds = _mean_std_per_timestep(data["timesteps"], data["win_rates"])
        _, sc_means, sc_stds = _mean_std_per_timestep(data["timesteps"], data["avg_score"])
        label = CONDITION_LABELS.get(name, name)
        print(
            f"{label:<20}  {wr_means[-1]:.3f} ± {wr_stds[-1]:.3f}  "
            f"{sc_means[-1]:.1f} ± {sc_stds[-1]:.1f}"
        )

    # --- Early convergence: steps to reach 0.55 win rate ---
    print("\n=== Steps to first reach WR ≥ 0.55 (per seed) ===")
    by_condition: dict[str, list] = defaultdict(list)
    for c in conditions:
        name = c["condition"]
        for entry in c.get("win_rate_history", []):
            if entry["win_rate_vs_random"] >= 0.55:
                by_condition[name].append(entry["timestep"])
                break
        else:
            by_condition[name].append(None)

    for name, steps in by_condition.items():
        label = CONDITION_LABELS.get(name, name)
        valid = [s for s in steps if s is not None]
        if valid:
            print(f"  {label:<20}  {int(np.mean(valid)):>10,} steps (mean)")
        else:
            print(f"  {label:<20}  never reached")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot TabletopOracle ablation curves")
    p.add_argument("--results", type=str, required=True, help="Path to results.json")
    p.add_argument(
        "--output", type=str, default="figures/ablation_curves.png",
        help="Output path for the figure",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_ablation(Path(args.results), Path(args.output))
