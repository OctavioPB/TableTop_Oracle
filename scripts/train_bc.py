"""CLI: BC pre-training followed by PPO fine-tuning on Wingspan.

Implements three experimental conditions for ablation study:

  Condition A — PPO from scratch (baseline)
  Condition B — BC pre-train → PPO fine-tune
  Condition C — BC only (no RL)

Usage:
    python scripts/train_bc.py \\
        --n-demo-games 100 \\
        --bc-epochs 50 \\
        --ppo-steps 500_000 \\
        --seed 42

Outputs (all in experiments/<exp_name>/):
  config.json          — full hyperparameter record
  results.json         — win rates for all three conditions
  bc_metrics.json      — BC accuracy and loss curve
  training_curves.png  — win-rate training curves

Checkpoints are saved to checkpoints/<exp_name>/.
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
logger = logging.getLogger("train_bc")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BC pre-training + PPO fine-tuning for Wingspan."
    )
    # Demo generation
    p.add_argument("--n-demo-games", type=int, default=100,
                   help="Number of GreedyAgent games to generate as demonstrations.")
    p.add_argument("--only-wins", action="store_true",
                   help="Filter demonstrations to only keep GreedyAgent wins.")

    # BC training
    p.add_argument("--bc-epochs", type=int, default=50,
                   help="Epochs for Behavioural Cloning pre-training.")
    p.add_argument("--bc-batch-size", type=int, default=64)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--bc-val-split", type=float, default=0.1)

    # PPO fine-tuning (applied to conditions A and B)
    p.add_argument("--ppo-steps", type=int, default=500_000,
                   help="Total PPO timesteps for fine-tuning (conditions A and B).")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--reward-mode", default="dense",
                   choices=["terminal", "dense", "shaped"])
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--features-dim", type=int, default=256)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)

    # Experiment
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--skip-condition-a", action="store_true",
                   help="Skip condition A (PPO from scratch) to save time.")
    p.add_argument("--skip-condition-c", action="store_true",
                   help="Skip condition C (BC only) to save time.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Experiment directory helpers
# ---------------------------------------------------------------------------


def _next_exp_id(experiments_dir: Path) -> int:
    ids: list[int] = []
    if experiments_dir.exists():
        for d in experiments_dir.iterdir():
            if d.is_dir() and d.name.startswith("exp_"):
                try:
                    ids.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
    return max(ids, default=0) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Reproducibility
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

    # ── Experiment directories ───────────────────────────────────────────────
    experiments_root = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
    checkpoints_root = Path(os.environ.get("CHECKPOINTS_DIR", "./checkpoints"))

    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_id = _next_exp_id(experiments_root)
        exp_name = f"exp_{exp_id:03d}_bc_ppo_wingspan_{args.reward_mode}_seed{args.seed}"

    exp_dir = experiments_root / exp_name
    ckpt_dir = checkpoints_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment dir: %s", exp_dir)

    # ── Config ───────────────────────────────────────────────────────────────
    config: dict = {
        "n_demo_games": args.n_demo_games,
        "only_wins": args.only_wins,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_lr": args.bc_lr,
        "bc_val_split": args.bc_val_split,
        "ppo_steps": args.ppo_steps,
        "n_envs": args.n_envs,
        "reward_mode": args.reward_mode,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "features_dim": args.features_dim,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "seed": args.seed,
        "exp_name": exp_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("Config saved.")

    # ── Generate demonstrations ───────────────────────────────────────────────
    from src.imitation.demo_buffer import SyntheticDemoGenerator

    logger.info("Generating %d GreedyAgent demonstrations…", args.n_demo_games)
    t_demo = time.time()
    generator = SyntheticDemoGenerator(reward_mode=args.reward_mode)
    demo_buffer = generator.generate(
        n_games=args.n_demo_games,
        seed=args.seed,
        only_wins=args.only_wins,
    )
    logger.info(
        "Demos ready: %d transitions, %d games, %d wins  (%.1fs)",
        len(demo_buffer), demo_buffer.n_games, demo_buffer.win_count,
        time.time() - t_demo,
    )

    # Save demo buffer for reproducibility
    buffer_path = ckpt_dir / "demo_buffer.pkl.gz"
    demo_buffer.save(buffer_path)
    logger.info("Demo buffer saved to %s", buffer_path)

    # ── Build environments & models ──────────────────────────────────────────
    from stable_baselines3.common.env_util import make_vec_env

    from src.agents.ppo_agent import build_maskable_ppo, make_callbacks
    from src.envs.wingspan_env import WingspanEnv

    reward_mode = args.reward_mode

    def _make_env() -> WingspanEnv:
        return WingspanEnv(reward_mode=reward_mode)

    tb_log = None if args.no_tensorboard else str(experiments_root / "tensorboard")

    ppo_kwargs = {
        "seed": args.seed,
        "tensorboard_log": tb_log,
        "learning_rate": 3e-4,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "features_dim": args.features_dim,
    }

    # ── Condition B: BC pre-train → PPO fine-tune ────────────────────────────
    logger.info("=== Condition B: BC → PPO ===")
    t_bc = time.time()

    vec_env_b = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed)
    model_b = build_maskable_ppo(vec_env_b, **ppo_kwargs)

    from src.agents.bc_agent import BehavioralCloningTrainer

    bc_trainer = BehavioralCloningTrainer(
        model=model_b,
        device="cpu",
        learning_rate=args.bc_lr,
    )
    bc_metrics = bc_trainer.train(
        demo_buffer,
        n_epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        val_split=args.bc_val_split,
    )
    logger.info(
        "BC complete: train_acc=%.3f  val_acc=%.3f  (%.1fs)",
        bc_metrics.bc_accuracy, bc_metrics.val_accuracy, time.time() - t_bc,
    )

    # Save BC checkpoint before PPO fine-tuning
    bc_save_path = ckpt_dir / "bc_pretrained"
    model_b.save(str(bc_save_path))
    logger.info("BC checkpoint saved to %s.zip", bc_save_path)

    bc_metrics_dict = {
        "bc_accuracy": bc_metrics.bc_accuracy,
        "val_accuracy": bc_metrics.val_accuracy,
        "loss_per_epoch": bc_metrics.loss_per_epoch,
        "n_transitions": bc_metrics.n_transitions,
    }
    (exp_dir / "bc_metrics.json").write_text(json.dumps(bc_metrics_dict, indent=2))

    # PPO fine-tune on top of BC weights
    eval_env_b = WingspanEnv(reward_mode=reward_mode)
    win_cb_b, ckpt_cb_b = make_callbacks(
        eval_env=eval_env_b,
        checkpoints_dir=ckpt_dir / "condition_b",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq_total=args.ppo_steps,
        n_envs=args.n_envs,
    )

    logger.info("PPO fine-tuning (condition B) for %d steps…", args.ppo_steps)
    t_b = time.time()
    model_b.learn(
        total_timesteps=args.ppo_steps,
        callback=[win_cb_b, ckpt_cb_b],
        tb_log_name=exp_name + "_condB",
        reset_num_timesteps=True,
    )
    elapsed_b = time.time() - t_b
    logger.info("Condition B PPO done in %.1fs", elapsed_b)

    model_b.save(str(ckpt_dir / "condition_b_final"))
    vec_env_b.close()
    eval_env_b.close()

    # Evaluate B
    from src.agents.ppo_agent import evaluate_ppo_win_rate

    results_b = evaluate_ppo_win_rate(
        model_b, n_episodes=args.n_eval_episodes * 5, seed=args.seed + 10,
        reward_mode=reward_mode,
    )
    logger.info(
        "Condition B final: win_rate=%.3f  avg_score_p0=%.1f",
        results_b["win_rate"], results_b["avg_score_p0"],
    )

    # ── Condition A: PPO from scratch ────────────────────────────────────────
    results_a: dict = {}
    if not args.skip_condition_a:
        logger.info("=== Condition A: PPO from scratch ===")
        vec_env_a = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed + 1)
        model_a = build_maskable_ppo(vec_env_a, **{**ppo_kwargs, "seed": args.seed + 1})
        eval_env_a = WingspanEnv(reward_mode=reward_mode)
        win_cb_a, ckpt_cb_a = make_callbacks(
            eval_env=eval_env_a,
            checkpoints_dir=ckpt_dir / "condition_a",
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq_total=args.ppo_steps,
            n_envs=args.n_envs,
        )

        t_a = time.time()
        model_a.learn(
            total_timesteps=args.ppo_steps,
            callback=[win_cb_a, ckpt_cb_a],
            tb_log_name=exp_name + "_condA",
            reset_num_timesteps=True,
        )
        elapsed_a = time.time() - t_a
        logger.info("Condition A PPO done in %.1fs", elapsed_a)

        model_a.save(str(ckpt_dir / "condition_a_final"))
        vec_env_a.close()
        eval_env_a.close()

        results_a = evaluate_ppo_win_rate(
            model_a, n_episodes=args.n_eval_episodes * 5, seed=args.seed + 20,
            reward_mode=reward_mode,
        )
        logger.info(
            "Condition A final: win_rate=%.3f  avg_score_p0=%.1f",
            results_a["win_rate"], results_a["avg_score_p0"],
        )

    # ── Condition C: BC only ─────────────────────────────────────────────────
    results_c: dict = {}
    if not args.skip_condition_c:
        logger.info("=== Condition C: BC only (no RL) ===")
        # Reload BC checkpoint into a fresh model for evaluation
        from sb3_contrib import MaskablePPO
        model_c = MaskablePPO.load(str(bc_save_path) + ".zip")
        results_c = evaluate_ppo_win_rate(
            model_c, n_episodes=args.n_eval_episodes * 5, seed=args.seed + 30,
            reward_mode=reward_mode,
        )
        logger.info(
            "Condition C final: win_rate=%.3f  avg_score_p0=%.1f",
            results_c["win_rate"], results_c["avg_score_p0"],
        )

    # ── Save results ─────────────────────────────────────────────────────────
    results: dict = {
        "config": config,
        "bc_metrics": bc_metrics_dict,
        "condition_a_ppo_scratch": results_a,
        "condition_b_bc_plus_ppo": results_b,
        "condition_c_bc_only": results_c,
        "win_rate_history_b": win_cb_b.win_rate_history,
        "win_rate_history_a": win_cb_a.win_rate_history if not args.skip_condition_a else [],
    }
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s/results.json", exp_dir)

    # ── Print comparison ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  BC + PPO Ablation — {exp_name}")
    print(f"{'─'*50}")
    print(f"  BC accuracy (train):     {bc_metrics.bc_accuracy:.3f}")
    print(f"  BC accuracy (val):       {bc_metrics.val_accuracy:.3f}")
    if results_a:
        print(f"  Condition A win rate:    {results_a['win_rate']:.3f}  (PPO scratch)")
    print(f"  Condition B win rate:    {results_b['win_rate']:.3f}  (BC + PPO)")
    if results_c:
        print(f"  Condition C win rate:    {results_c['win_rate']:.3f}  (BC only)")
    print(f"{'─'*50}\n")

    # ── Training curve plot ───────────────────────────────────────────────────
    _plot_training_curves(
        history_a=win_cb_a.win_rate_history if not args.skip_condition_a else [],
        history_b=win_cb_b.win_rate_history,
        bc_accuracy=bc_metrics.bc_accuracy,
        exp_dir=exp_dir,
    )


def _plot_training_curves(
    history_a: list[dict],
    history_b: list[dict],
    bc_accuracy: float,
    exp_dir: Path,
) -> None:
    """Save training_curves.png comparing conditions A and B."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        if history_b:
            steps_b = [r["timestep"] for r in history_b]
            wr_b = [r["win_rate_vs_random"] for r in history_b]
            ax1.plot(steps_b, wr_b, marker="o", color="steelblue", label="B: BC+PPO")

        if history_a:
            steps_a = [r["timestep"] for r in history_a]
            wr_a = [r["win_rate_vs_random"] for r in history_a]
            ax1.plot(steps_a, wr_a, marker="s", color="salmon", label="A: PPO scratch")

        ax1.axhline(0.5, ls="--", color="gray", alpha=0.5, label="50% baseline")
        ax1.axhline(bc_accuracy, ls=":", color="green", alpha=0.7,
                    label=f"BC accuracy ({bc_accuracy:.2f})")
        ax1.set_xlabel("PPO Timesteps")
        ax1.set_ylabel("Win rate vs Random")
        ax1.set_ylim(0, 1)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_title("Win Rate Comparison")

        # Loss curve from bc_metrics stored in exp_dir/bc_metrics.json
        try:
            import json
            bc_data = json.loads((exp_dir / "bc_metrics.json").read_text())
            losses = bc_data.get("loss_per_epoch", [])
            if losses:
                ax2.plot(range(1, len(losses) + 1), losses, color="purple", marker=".")
                ax2.set_xlabel("BC Epoch")
                ax2.set_ylabel("Cross-entropy loss")
                ax2.grid(alpha=0.3)
                ax2.set_title("BC Training Loss")
        except Exception:
            pass

        fig.suptitle(f"Wingspan BC+PPO — {exp_dir.name}")
        fig.tight_layout()
        fig.savefig(exp_dir / "training_curves.png", dpi=120)
        plt.close(fig)
        logger.info("Training curves saved.")
    except Exception as exc:
        logger.warning("Could not save training curves: %s", exc)


if __name__ == "__main__":
    main()
