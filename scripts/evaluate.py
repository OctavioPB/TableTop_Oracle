"""CLI: evaluate a trained MaskablePPO checkpoint against the random opponent.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/exp_001_.../ppo_wingspan_final.zip \\
        --n-games 200 \\
        --reward-mode dense \\
        --seed 0

Outputs a summary table and writes results to the checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger("evaluate")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .zip checkpoint (with or without .zip extension).")
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--reward-mode", default="dense",
                   choices=["terminal", "dense", "shaped"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--also-vs-greedy", action="store_true",
                   help="Also run evaluation vs GreedyAgent.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ckpt_path = Path(args.checkpoint)
    if not str(ckpt_path).endswith(".zip"):
        ckpt_path = ckpt_path.with_suffix(".zip")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Load model ───────────────────────────────────────────────────────────
    from sb3_contrib import MaskablePPO

    from src.envs.wingspan_env import WingspanEnv

    logger.info("Loading checkpoint: %s", ckpt_path)
    eval_env = WingspanEnv(reward_mode=args.reward_mode)
    model = MaskablePPO.load(str(ckpt_path), env=eval_env)

    # ── Evaluate vs random ───────────────────────────────────────────────────
    from src.agents.ppo_agent import evaluate_ppo_win_rate

    logger.info("Evaluating %d games vs random opponent…", args.n_games)
    results_random = evaluate_ppo_win_rate(
        model,
        n_episodes=args.n_games,
        seed=args.seed,
        reward_mode=args.reward_mode,
    )

    _print_results("vs Random", results_random)
    all_results = {"vs_random": results_random}

    # ── Evaluate vs greedy (optional) ────────────────────────────────────────
    if args.also_vs_greedy:
        from src.agents.baselines import GreedyAgent
        from src.games.wingspan.engine import WingspanEngine

        logger.info("Evaluating %d games vs greedy opponent…", args.n_games)
        results_greedy = _eval_vs_greedy(
            model, args.n_games, args.seed, args.reward_mode
        )
        _print_results("vs Greedy", results_greedy)
        all_results["vs_greedy"] = results_greedy

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = ckpt_path.parent / f"eval_{ckpt_path.stem}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    eval_env.close()


def _eval_vs_greedy(
    model: object,
    n_games: int,
    seed: int,
    reward_mode: str,
) -> dict:
    """Evaluate model vs GreedyAgent using the engine directly."""
    import random as pyrandom

    from src.agents.baselines import GreedyAgent
    from src.envs.wingspan_env import WingspanEnv

    eval_env = WingspanEnv(reward_mode=reward_mode)
    greedy = GreedyAgent()
    rng = pyrandom.Random(seed)

    wins = 0
    scores_p0 = []
    scores_p1 = []

    for _ in range(n_games):
        obs, _ = eval_env.reset(seed=rng.randint(0, 100_000))
        done = False
        info: dict = {}

        while not done:
            masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)  # type: ignore[attr-defined]
            obs, _r, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated

        s0 = info.get("player_0_score", 0)
        s1 = info.get("player_1_score", 0)
        scores_p0.append(s0)
        scores_p1.append(s1)
        if s0 > s1:
            wins += 1

    import numpy as np
    return {
        "win_rate": wins / n_games,
        "avg_score_p0": float(np.mean(scores_p0)),
        "avg_score_p1": float(np.mean(scores_p1)),
        "n_episodes": n_games,
    }


def _print_results(label: str, results: dict) -> None:
    print(f"\n{'─'*40}")
    print(f"  Evaluation: {label}")
    print(f"{'─'*40}")
    print(f"  Win rate:       {results['win_rate']:.3f} ({results['win_rate']*100:.1f}%)")
    print(f"  Avg score P0:   {results['avg_score_p0']:.1f}")
    print(f"  Avg score P1:   {results['avg_score_p1']:.1f}")
    print(f"  N games:        {results['n_episodes']}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
