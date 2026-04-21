"""CLI: train MaskablePPO on a Wingspan environment.

Usage:
    python scripts/train_ppo.py \\
        --game wingspan \\
        --total-timesteps 1_000_000 \\
        --n-envs 4 \\
        --reward-mode dense \\
        --seed 42

All hyperparameters are saved to experiments/<exp_name>/config.json so
every run is fully reproducible. Never overwrites an existing experiment dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("train_ppo")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaskablePPO on Wingspan.")
    p.add_argument("--game", default="wingspan", choices=["wingspan", "seven_wonders_duel", "splendor"])
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel environments (DummyVecEnv).")
    p.add_argument("--reward-mode", default="dense",
                   choices=["terminal", "dense", "shaped"])
    p.add_argument("--seed", type=int, default=42)
    # Network / optimisation
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Rollout steps per env per update.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--features-dim", type=int, default=256)
    # Evaluation
    p.add_argument("--eval-freq", type=int, default=50_000,
                   help="Total timesteps between win-rate evaluations.")
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--save-freq", type=int, default=100_000,
                   help="Total timesteps between checkpoint saves.")
    # Experiment tracking
    p.add_argument("--experiment-name", default=None,
                   help="Experiment directory name (auto-generated if not set).")
    p.add_argument("--no-tensorboard", action="store_true")
    return p.parse_args()


def _next_exp_id(experiments_dir: Path) -> int:
    """Return the next sequential experiment ID."""
    ids: list[int] = []
    if experiments_dir.exists():
        for d in experiments_dir.iterdir():
            if d.is_dir() and d.name.startswith("exp_"):
                try:
                    ids.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
    return max(ids, default=0) + 1


def main() -> None:
    args = _parse_args()

    # ── Reproducibility seeds ────────────────────────────────────────────────
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    # ── Experiment directory ─────────────────────────────────────────────────
    experiments_root = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
    checkpoints_root = Path(os.environ.get("CHECKPOINTS_DIR", "./checkpoints"))

    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_id = _next_exp_id(experiments_root)
        exp_name = (
            f"exp_{exp_id:03d}_ppo_{args.game}_"
            f"{args.reward_mode}_seed{args.seed}"
        )

    exp_dir = experiments_root / exp_name
    ckpt_dir = checkpoints_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment dir: %s", exp_dir)

    # ── Config ───────────────────────────────────────────────────────────────
    config: dict = {
        "game":            args.game,
        "total_timesteps": args.total_timesteps,
        "n_envs":          args.n_envs,
        "reward_mode":     args.reward_mode,
        "seed":            args.seed,
        "learning_rate":   args.learning_rate,
        "n_steps":         args.n_steps,
        "batch_size":      args.batch_size,
        "n_epochs":        args.n_epochs,
        "features_dim":    args.features_dim,
        "eval_freq":       args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "save_freq":       args.save_freq,
        "net_arch":        [256, 256],
        "gamma":           0.99,
        "gae_lambda":      0.95,
        "clip_range":      0.2,
        "ent_coef":        0.01,
        "exp_name":        exp_name,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("Config saved to %s/config.json", exp_dir)

    # ── Environments ─────────────────────────────────────────────────────────
    from stable_baselines3.common.env_util import make_vec_env

    reward_mode = args.reward_mode

    if args.game == "seven_wonders_duel":
        from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv

        def _make_env():  # type: ignore[return]
            return SevenWondersDuelEnv(reward_mode=reward_mode)

        eval_env = SevenWondersDuelEnv(reward_mode=reward_mode)
    elif args.game == "splendor":
        from src.envs.splendor_env import SplendorEnv

        def _make_env():  # type: ignore[return]
            return SplendorEnv(reward_mode=reward_mode)

        eval_env = SplendorEnv(reward_mode=reward_mode)
    else:
        from src.envs.wingspan_env import WingspanEnv

        def _make_env():  # type: ignore[return]
            return WingspanEnv(reward_mode=reward_mode)

        eval_env = WingspanEnv(reward_mode=reward_mode)

    logger.info("Creating %d training envs…", args.n_envs)
    vec_env = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed)

    # ── Model ────────────────────────────────────────────────────────────────
    from src.agents.ppo_agent import build_maskable_ppo, make_callbacks

    tb_log = None if args.no_tensorboard else str(experiments_root / "tensorboard")

    model = build_maskable_ppo(
        vec_env,
        seed=args.seed,
        tensorboard_log=tb_log,
        game=args.game,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        features_dim=args.features_dim,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    win_rate_cb, checkpoint_cb = make_callbacks(
        eval_env=eval_env,
        checkpoints_dir=ckpt_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq_total=args.save_freq,
        n_envs=args.n_envs,
    )

    # ── Training ─────────────────────────────────────────────────────────────
    logger.info("Training for %d timesteps…", args.total_timesteps)
    t0 = time.time()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[win_rate_cb, checkpoint_cb],
        tb_log_name=exp_name,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs (%.0f steps/s)",
                elapsed, args.total_timesteps / max(elapsed, 1))

    # ── Save final model ─────────────────────────────────────────────────────
    final_path = ckpt_dir / f"ppo_{args.game}_final"
    model.save(str(final_path))
    logger.info("Final model saved to %s.zip", final_path)

    # ── Final evaluation ─────────────────────────────────────────────────────
    from src.agents.ppo_agent import evaluate_ppo_win_rate

    logger.info("Running final evaluation (%d episodes)…", args.n_eval_episodes * 5)
    eval_results = evaluate_ppo_win_rate(
        model,
        n_episodes=args.n_eval_episodes * 5,
        seed=args.seed + 1,
        reward_mode=reward_mode,
        game=args.game,
    )

    # ── Results ──────────────────────────────────────────────────────────────
    results: dict = {
        "config":            config,
        "win_rate_history":  win_rate_cb.win_rate_history,
        "final_eval":        eval_results,
        "training_seconds":  elapsed,
    }
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(
        "Results saved. Final win_rate_vs_random=%.3f | avg_score_p0=%.1f",
        eval_results["win_rate"],
        eval_results["avg_score_p0"],
    )

    # ── Training curve plot ───────────────────────────────────────────────────
    _plot_training_curves(win_rate_cb.win_rate_history, exp_dir)

    vec_env.close()
    eval_env.close()


def _plot_training_curves(
    history: list[dict],
    exp_dir: Path,
) -> None:
    """Save a training_curves.png to the experiment directory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not history:
            return

        steps = [r["timestep"] for r in history]
        win_rates = [r["win_rate_vs_random"] for r in history]
        scores_p0 = [r["avg_score_p0"] for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(steps, win_rates, marker="o", color="steelblue")
        ax1.axhline(0.5, ls="--", color="gray", alpha=0.5, label="50% baseline")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Win rate vs Random")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title("Win Rate vs Random Agent")

        ax2.plot(steps, scores_p0, marker="s", color="salmon")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Avg score (P0)")
        ax2.grid(alpha=0.3)
        ax2.set_title("Average Score (PPO agent)")

        fig.suptitle(f"MaskablePPO — {exp_dir.name}")
        fig.tight_layout()
        fig.savefig(exp_dir / "training_curves.png", dpi=120)
        plt.close(fig)
        logger.info("Training curves saved to %s/training_curves.png", exp_dir)
    except Exception as exc:
        logger.warning("Could not save training curves: %s", exc)


if __name__ == "__main__":
    main()
