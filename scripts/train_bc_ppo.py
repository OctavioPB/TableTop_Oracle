"""BC pre-train -> PPO fine-tune for any supported game.

Usage:
    python scripts/train_bc_ppo.py --game seven_wonders_duel --seeds 42 123 7
    python scripts/train_bc_ppo.py --game wingspan --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import json
import logging
import random as py_random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_bc_ppo")


def _make_env(game: str, reward_mode: str):
    if game == "seven_wonders_duel":
        from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv
        return SevenWondersDuelEnv(reward_mode=reward_mode)
    if game == "splendor":
        from src.envs.splendor_env import SplendorEnv
        return SplendorEnv(reward_mode=reward_mode)
    from src.envs.wingspan_env import WingspanEnv
    return WingspanEnv(reward_mode=reward_mode)


def _run_bc_ppo(
    game: str,
    seed: int,
    total_timesteps: int,
    n_envs: int,
    reward_mode: str,
    n_demo_games: int,
    bc_epochs: int,
    exp_dir: Path,
) -> dict:
    import numpy as np
    import torch
    from stable_baselines3.common.env_util import make_vec_env

    from src.agents.bc_agent import BehavioralCloningTrainer, load_bc_weights_into_ppo
    from src.agents.ppo_agent import WinRateCallback, build_maskable_ppo
    from src.imitation.demo_buffer import SyntheticDemoGenerator

    torch.manual_seed(seed)
    np.random.seed(seed)
    py_random.seed(seed)

    exp_dir.mkdir(parents=True, exist_ok=True)

    # --- BC phase ---
    logger.info("BC phase: generating %d demo games (seed=%d)...", n_demo_games, seed)
    gen = SyntheticDemoGenerator(reward_mode=reward_mode, game=game)
    buffer = gen.generate(n_games=n_demo_games, seed=seed)

    env_single = _make_env(game, reward_mode)
    model_bc = build_maskable_ppo(env_single, seed=seed, tensorboard_log=None, game=game)

    trainer = BehavioralCloningTrainer(model_bc, device="cpu")
    bc_metrics = trainer.train(buffer, n_epochs=bc_epochs, batch_size=64)
    logger.info(
        "BC done: train_acc=%.3f val_acc=%.3f n_transitions=%d",
        bc_metrics.bc_accuracy, bc_metrics.val_accuracy, bc_metrics.n_transitions,
    )

    bc_checkpoint = exp_dir / "bc_pretrain.zip"
    model_bc.save(str(bc_checkpoint))

    # --- PPO phase ---
    env_vec = make_vec_env(
        lambda: _make_env(game, reward_mode), n_envs=n_envs, seed=seed,
    )
    model_ppo = build_maskable_ppo(env_vec, seed=seed, tensorboard_log=None, game=game)
    load_bc_weights_into_ppo(model_bc, model_ppo)

    eval_env = _make_env(game, reward_mode)
    cb = WinRateCallback(eval_env=eval_env, eval_freq=50_000, n_eval_episodes=20, verbose=1)

    logger.info("PPO phase: %d timesteps...", total_timesteps)
    model_ppo.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
    eval_env.close()

    checkpoint = exp_dir / f"bc_ppo_{game}_seed{seed}.zip"
    model_ppo.save(str(checkpoint))

    return {
        "game": game,
        "condition": "bc_ppo",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "bc_accuracy": bc_metrics.bc_accuracy,
        "bc_val_accuracy": bc_metrics.val_accuracy,
        "bc_n_transitions": bc_metrics.n_transitions,
        "win_rate_history": cb.win_rate_history,
        "checkpoint": str(checkpoint),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC pre-train + PPO fine-tune")
    p.add_argument("--game", default="wingspan",
                   choices=["wingspan", "seven_wonders_duel", "splendor"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--reward-mode", default="dense",
                   choices=["terminal", "dense", "shaped"])
    p.add_argument("--n-demo-games", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=50)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--experiments-dir", default="./experiments")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    exp_name = args.experiment_name or f"bc_ppo_{args.game}"
    exp_root = Path(args.experiments_dir) / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    (exp_root / "config.json").write_text(json.dumps(config, indent=2))

    all_results = []
    t0 = time.time()

    for seed in args.seeds:
        logger.info("=== seed %d ===", seed)
        result = _run_bc_ppo(
            game=args.game,
            seed=seed,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            reward_mode=args.reward_mode,
            n_demo_games=args.n_demo_games,
            bc_epochs=args.bc_epochs,
            exp_dir=exp_root / f"seed_{seed}",
        )
        all_results.append(result)

    summary = {
        "total_elapsed_seconds": time.time() - t0,
        "conditions": all_results,
    }
    (exp_root / "results.json").write_text(json.dumps(summary, indent=2))
    logger.info("Results saved to %s/results.json", exp_root)

    logger.info("\n=== Summary ===")
    for r in all_results:
        last_wr = r["win_rate_history"][-1]["win_rate_vs_random"] if r.get("win_rate_history") else "N/A"
        logger.info("  seed=%-4d  final_wr=%s  bc_val_acc=%.3f",
                    r["seed"], last_wr, r["bc_val_accuracy"])


if __name__ == "__main__":
    main()
