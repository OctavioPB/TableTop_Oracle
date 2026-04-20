"""MaskablePPO agent configuration, callbacks, and evaluation helpers.

Hyperparameter choices (documented for reproducibility / ablation):
  learning_rate  3e-4   — Adam default; works well for discrete action spaces
  n_steps        2048   — rollout length per env; covers ~150 game turns per env
  batch_size     64     — mini-batch size for PPO update
  n_epochs       10     — PPO update epochs per rollout
  gamma          0.99   — discount; long game (26 turns) needs high gamma
  gae_lambda     0.95   — GAE bias-variance tradeoff
  clip_range     0.2    — PPO clip (standard)
  ent_coef       0.01   — small entropy bonus to prevent early collapse
  net_arch      [256, 256] — actor and critic MLP layers after feature extractor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

logger = logging.getLogger(__name__)

# Default hyperparameters — change via build_maskable_ppo kwargs, not here
_DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "net_arch": [256, 256],
    "features_dim": 256,
}


def build_maskable_ppo(
    env: Any,
    seed: int = 42,
    tensorboard_log: str | None = "./experiments/tensorboard",
    game: str = "wingspan",
    **hyperparams: Any,
) -> Any:
    """Construct and return a MaskablePPO model with project defaults.

    Args:
        env: Vectorised gym env (from make_vec_env).
        seed: Random seed for reproducibility.
        tensorboard_log: Directory for TensorBoard logs; None to disable.
        game: Game name — selects the appropriate features extractor.
        **hyperparams: Override any default hyperparameter.

    Returns:
        Configured MaskablePPO ready for .learn().
    """
    from sb3_contrib import MaskablePPO

    from src.agents.encoders import SWDFeaturesExtractor, WingspanFeaturesExtractor

    hp = {**_DEFAULT_HYPERPARAMS, **hyperparams}
    features_dim: int = hp.pop("features_dim")
    net_arch: list[int] = hp.pop("net_arch")

    extractor_cls = SWDFeaturesExtractor if game == "seven_wonders_duel" else WingspanFeaturesExtractor

    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": net_arch,
    }

    model = MaskablePPO(
        "MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=0,
        **hp,
    )
    logger.info(
        "Built MaskablePPO: lr=%.2e n_steps=%d batch=%d epochs=%d net_arch=%s",
        hp["learning_rate"],
        hp["n_steps"],
        hp["batch_size"],
        hp["n_epochs"],
        net_arch,
    )
    return model


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------


class WinRateCallback(BaseCallback):
    """Evaluates win rate against a random opponent every `eval_freq` steps.

    Win rate is recorded in `win_rate_history` (list of dicts) and logged
    to TensorBoard under `eval/win_rate_vs_random`.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self.win_rate_history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        wins = 0
        scores_p0: list[int] = []
        scores_p1: list[int] = []

        for _ in range(self._n_eval_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            info: dict[str, Any] = {}

            while not done:
                masks = self._eval_env.action_masks()
                action, _ = self.model.predict(
                    obs,
                    action_masks=masks,
                    deterministic=True,
                )
                obs, _reward, terminated, truncated, info = self._eval_env.step(int(action))
                done = terminated or truncated

            s0 = info.get("player_0_score", 0)
            s1 = info.get("player_1_score", 0)
            scores_p0.append(s0)
            scores_p1.append(s1)
            winner = info.get("winner")
            if winner == 0 or (winner is None and s0 > s1):
                wins += 1

        win_rate = wins / self._n_eval_episodes
        avg_s0 = float(np.mean(scores_p0))
        avg_s1 = float(np.mean(scores_p1))

        self.logger.record("eval/win_rate_vs_random", win_rate)
        self.logger.record("eval/avg_score_p0", avg_s0)
        self.logger.record("eval/avg_score_p1", avg_s1)

        record = {
            "timestep": self.num_timesteps,
            "win_rate_vs_random": win_rate,
            "avg_score_p0": avg_s0,
            "avg_score_p1": avg_s1,
        }
        self.win_rate_history.append(record)

        if self.verbose >= 1:
            logger.info(
                "Eval @ %d steps | win_rate=%.3f | score P0=%.1f P1=%.1f",
                self.num_timesteps, win_rate, avg_s0, avg_s1,
            )
        return True


def make_callbacks(
    eval_env: Any,
    checkpoints_dir: Path,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 20,
    save_freq_total: int = 100_000,
    n_envs: int = 1,
) -> tuple[WinRateCallback, CheckpointCallback]:
    """Create the standard training callback pair.

    Args:
        eval_env: A single (non-vectorised) WingspanEnv for evaluation.
        checkpoints_dir: Directory for model checkpoints.
        eval_freq: Total steps between win-rate evaluations.
        n_eval_episodes: Games per evaluation.
        save_freq_total: Total steps between checkpoints.
        n_envs: Number of training envs (used to convert total→per-env freq).

    Returns:
        (WinRateCallback, CheckpointCallback)
    """
    win_rate_cb = WinRateCallback(
        eval_env=eval_env,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq_total // n_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix="ppo_wingspan",
        verbose=0,
    )
    return win_rate_cb, checkpoint_cb


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_ppo_win_rate(
    model: Any,
    n_episodes: int = 100,
    seed: int = 0,
    reward_mode: str = "dense",
    game: str = "wingspan",
) -> dict[str, Any]:
    """Evaluate a trained MaskablePPO against the random opponent.

    Creates a fresh env (random opponent built-in) for the given game and runs
    n_episodes games using the model's deterministic policy.

    Returns:
        dict with win_rate, avg_score_p0, avg_score_p1, n_episodes.
    """
    import random as pyrandom

    if game == "seven_wonders_duel":
        from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv
        eval_env = SevenWondersDuelEnv(reward_mode=reward_mode)
    else:
        from src.envs.wingspan_env import WingspanEnv  # type: ignore[attr-defined]
        eval_env = WingspanEnv(reward_mode=reward_mode)
    rng = pyrandom.Random(seed)

    wins = 0
    scores_p0: list[int] = []
    scores_p1: list[int] = []

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=rng.randint(0, 100_000))
        done = False
        info: dict[str, Any] = {}

        while not done:
            masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _r, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated

        s0 = info.get("player_0_score", 0)
        s1 = info.get("player_1_score", 0)
        scores_p0.append(s0)
        scores_p1.append(s1)
        winner = info.get("winner")
        if winner == 0 or (winner is None and s0 > s1):
            wins += 1

    return {
        "win_rate":    wins / n_episodes,
        "avg_score_p0": float(np.mean(scores_p0)),
        "avg_score_p1": float(np.mean(scores_p1)),
        "n_episodes":  n_episodes,
    }
