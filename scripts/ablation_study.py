"""Ablation study script — Sprint 6.

Runs 4 conditions × 3 seeds × 1M steps and writes results to an experiment
directory. Each condition isolates one component of the full system:

  Variant 1 (baseline): PPO from scratch, no BC, no RAG
  Variant 2 (rag):      PPO + RAG oracle consulted on edge cases (not yet impl)
  Variant 3 (bc):       BC → PPO warm-start, no RAG
  Variant 4 (full):     BC → PPO + RAG (complete system)

Because variants 2 and 4 depend on Rule Oracle infrastructure that requires a
populated ChromaDB, they are gated behind --include-rag. Without that flag,
only variants 1 and 3 run.

Usage:
    python scripts/ablation_study.py \
        --total-timesteps 1_000_000 \
        --seeds 42 123 7 \
        --n-envs 4 \
        --n-demo-games 200 \
        --bc-epochs 50 \
        --reward-mode dense \
        [--include-rag] \
        [--experiment-name ablation_s6]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ablation_study")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_experiment_dir(experiments_root: Path, name: str) -> Path:
    """Create a unique numbered experiment directory."""
    existing = sorted(experiments_root.glob("exp_*"))
    next_num = len(existing) + 1
    exp_dir = experiments_root / f"exp_{next_num:03d}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _evaluate_agent(agent_wrapper: object, engine: object, n_games: int = 200) -> dict:
    """Return a metrics dict for agent_wrapper using baseline opponents."""
    from src.agents.baselines import GreedyAgent, RandomAgent
    from src.eval.metrics import avg_score, win_rate

    random_opp = RandomAgent(seed=0)
    greedy_opp = GreedyAgent()

    wr_random = win_rate(agent_wrapper, random_opp, engine, n_games=n_games)
    wr_greedy = win_rate(agent_wrapper, greedy_opp, engine, n_games=n_games)
    mean_score, std_score = avg_score(agent_wrapper, engine, n_games=n_games)

    return {
        "win_rate_vs_random": wr_random,
        "win_rate_vs_greedy": wr_greedy,
        "avg_score_mean": mean_score,
        "avg_score_std": std_score,
    }


class _PPOAgentWrapper:
    """Adapts a MaskablePPO model to the BaseAgent interface for eval.

    select_action uses the deterministic policy (argmax), which is standard
    for evaluation (no exploration noise).
    """

    def __init__(self, model: object) -> None:
        self._model = model

    def select_action(self, state: object, legal_actions: list) -> object:
        from src.envs.wingspan_env import WingspanEnv

        env = WingspanEnv.__new__(WingspanEnv)
        env._state = state
        env._engine = self._model.env.envs[0]._engine  # type: ignore[attr-defined]

        obs = env._get_obs() if hasattr(env, "_get_obs") else {}
        action_array, _ = self._model.predict(obs, deterministic=True)
        action_idx = int(action_array)

        if hasattr(env, "_idx_to_action"):
            return env._idx_to_action(action_idx, state)
        return legal_actions[0]


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def _run_condition_baseline(
    seed: int,
    total_timesteps: int,
    n_envs: int,
    reward_mode: str,
    exp_dir: Path,
    no_tensorboard: bool,
) -> dict:
    """Variant 1: PPO from scratch."""
    import torch
    import numpy as np
    import random as py_random
    from stable_baselines3.common.env_util import make_vec_env
    from src.agents.ppo_agent import WinRateCallback, build_maskable_ppo
    from src.envs.wingspan_env import WingspanEnv

    torch.manual_seed(seed)
    np.random.seed(seed)
    py_random.seed(seed)

    tb_log = None if no_tensorboard else str(exp_dir / "tensorboard")
    env = make_vec_env(lambda: WingspanEnv(reward_mode=reward_mode), n_envs=n_envs)
    eval_env = WingspanEnv(reward_mode=reward_mode)
    model = build_maskable_ppo(env, seed=seed, tensorboard_log=tb_log)

    cb = WinRateCallback(eval_env=eval_env, eval_freq=50_000, n_eval_episodes=50)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    eval_env.close()

    checkpoint = exp_dir / "ppo_baseline.zip"
    model.save(str(checkpoint))

    return {
        "condition": "baseline",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "win_rate_history": cb.win_rate_history,
        "checkpoint": str(checkpoint),
    }


def _run_condition_bc_ppo(
    seed: int,
    total_timesteps: int,
    n_envs: int,
    reward_mode: str,
    n_demo_games: int,
    bc_epochs: int,
    exp_dir: Path,
    no_tensorboard: bool,
) -> dict:
    """Variant 3: BC pre-train → PPO fine-tune."""
    import torch
    import numpy as np
    import random as py_random
    from stable_baselines3.common.env_util import make_vec_env
    from src.agents.bc_agent import BehavioralCloningTrainer, load_bc_weights_into_ppo
    from src.agents.ppo_agent import WinRateCallback, build_maskable_ppo
    from src.envs.wingspan_env import WingspanEnv
    from src.imitation.demo_buffer import SyntheticDemoGenerator

    torch.manual_seed(seed)
    np.random.seed(seed)
    py_random.seed(seed)

    # --- BC phase ---
    logger.info("Condition bc_ppo seed=%d: generating %d demo games…", seed, n_demo_games)
    gen = SyntheticDemoGenerator(reward_mode=reward_mode)
    buffer = gen.generate(n_games=n_demo_games, seed=seed)

    env_single = WingspanEnv(reward_mode=reward_mode)
    model_bc = build_maskable_ppo(env_single, seed=seed, tensorboard_log=None)

    trainer = BehavioralCloningTrainer(model_bc, device="cpu")
    bc_metrics = trainer.train(buffer, n_epochs=bc_epochs, batch_size=64)

    bc_checkpoint = exp_dir / "bc_pretrain.zip"
    model_bc.save(str(bc_checkpoint))

    # --- PPO phase ---
    tb_log = None if no_tensorboard else str(exp_dir / "tensorboard")
    env_vec = make_vec_env(lambda: WingspanEnv(reward_mode=reward_mode), n_envs=n_envs)
    model_ppo = build_maskable_ppo(env_vec, seed=seed, tensorboard_log=tb_log)
    load_bc_weights_into_ppo(model_bc, model_ppo)

    eval_env_ppo = WingspanEnv(reward_mode=reward_mode)
    cb = WinRateCallback(eval_env=eval_env_ppo, eval_freq=50_000, n_eval_episodes=50)
    model_ppo.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    eval_env_ppo.close()

    checkpoint = exp_dir / "ppo_bc.zip"
    model_ppo.save(str(checkpoint))

    return {
        "condition": "bc_ppo",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "bc_accuracy": bc_metrics.bc_accuracy,
        "bc_val_accuracy": bc_metrics.val_accuracy,
        "bc_n_transitions": bc_metrics.n_transitions,
        "win_rate_history": cb.win_rate_history,
        "checkpoint": str(checkpoint),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TabletopOracle ablation study (S6)")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-demo-games", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=50)
    p.add_argument("--reward-mode", type=str, default="dense")
    p.add_argument("--include-rag", action="store_true",
                   help="Run variants 2 and 4 (requires populated ChromaDB)")
    p.add_argument("--eval-games", type=int, default=200,
                   help="Games per agent pair for final evaluation")
    p.add_argument("--experiment-name", type=str, default="ablation_s6")
    p.add_argument("--experiments-dir", type=str, default="./experiments")
    p.add_argument("--no-tensorboard", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    experiments_root = Path(args.experiments_dir)
    experiments_root.mkdir(parents=True, exist_ok=True)
    exp_dir = _next_experiment_dir(experiments_root, args.experiment_name)
    logger.info("Experiment directory: %s", exp_dir)

    config = vars(args)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    all_results: list[dict] = []
    start_time = time.time()

    for seed in args.seeds:
        logger.info("=== Seed %d ===", seed)
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        # Variant 1: baseline
        logger.info("Running variant 1 (baseline) seed=%d", seed)
        v1 = _run_condition_baseline(
            seed=seed,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            reward_mode=args.reward_mode,
            exp_dir=seed_dir,
            no_tensorboard=args.no_tensorboard,
        )
        all_results.append(v1)

        # Variant 3: BC → PPO
        logger.info("Running variant 3 (bc_ppo) seed=%d", seed)
        v3 = _run_condition_bc_ppo(
            seed=seed,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            reward_mode=args.reward_mode,
            n_demo_games=args.n_demo_games,
            bc_epochs=args.bc_epochs,
            exp_dir=seed_dir,
            no_tensorboard=args.no_tensorboard,
        )
        all_results.append(v3)

        if args.include_rag:
            logger.warning(
                "RAG variants (2 and 4) not yet implemented — skipping for seed %d.", seed
            )

    elapsed = time.time() - start_time

    results_summary = {
        "total_elapsed_seconds": elapsed,
        "conditions": all_results,
    }
    (exp_dir / "results.json").write_text(json.dumps(results_summary, indent=2))
    logger.info("Results saved to %s", exp_dir / "results.json")

    # --- Summary table ---
    logger.info("\n=== Ablation summary ===")
    for r in all_results:
        last_wr = r["win_rate_history"][-1]["win_rate_vs_random"] if r.get("win_rate_history") else "N/A"
        logger.info(
            "  condition=%-12s  seed=%d  final_win_rate=%s",
            r["condition"], r["seed"], last_wr,
        )

    logger.info("Experiment complete. Directory: %s", exp_dir)


if __name__ == "__main__":
    main()
