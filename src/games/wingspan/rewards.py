"""Wingspan reward shaping — three modes for ablation study.

Modes (hyperparameter for Sprint 4/6 ablation):
  terminal: +1 win / -1 loss — pure, slowest convergence
  dense:    delta score per action — faster convergence, potential bias
  shaped:   potential-based shaping (Ng et al. 1999) — theoretically sound
"""

from __future__ import annotations

from typing import Literal

from src.games.wingspan.actions import WingspanAction
from src.games.wingspan.state import WingspanState

RewardMode = Literal["terminal", "dense", "shaped"]


def compute_reward(
    state_before: WingspanState,
    action: WingspanAction,
    state_after: WingspanState,
    done: bool,
    reward_mode: RewardMode = "dense",
    player_id: int = 0,
    engine=None,
) -> float:
    """Compute the scalar reward for a (s, a, s') transition.

    Args:
        state_before: Game state before the action.
        action: The action that was applied.
        state_after: Game state after the action.
        done: Whether the episode ended.
        reward_mode: Reward type; one of 'terminal', 'dense', 'shaped'.
        player_id: The acting player.
        engine: WingspanEngine instance (needed for score computation).

    Returns:
        Scalar reward signal.
    """
    if reward_mode == "terminal":
        return _terminal_reward(state_after, done, player_id, engine)
    if reward_mode == "dense":
        return _dense_reward(state_before, state_after, done, player_id, engine)
    if reward_mode == "shaped":
        return _shaped_reward(state_before, state_after, done, player_id, engine)
    raise ValueError(f"Unknown reward_mode: {reward_mode}")


# ---------------------------------------------------------------------------
# Reward implementations
# ---------------------------------------------------------------------------


def _terminal_reward(
    state_after: WingspanState,
    done: bool,
    player_id: int,
    engine,
) -> float:
    if not done:
        return 0.0
    if engine is None:
        return 0.0
    winner = engine.get_winner(state_after)
    return 1.0 if winner == player_id else -1.0


def _dense_reward(
    state_before: WingspanState,
    state_after: WingspanState,
    done: bool,
    player_id: int,
    engine,
) -> float:
    """Reward = normalised score delta per action.

    Normalised by max possible score (~200 pts for a full 15-bird board)
    so reward stays in [-1, 1].
    """
    _MAX_SCORE = 150.0

    if engine is None:
        return 0.0

    before = engine._compute_final_score(state_before, player_id) / _MAX_SCORE
    after = engine._compute_final_score(state_after, player_id) / _MAX_SCORE
    delta = after - before

    if done:
        winner = engine.get_winner(state_after)
        delta += 0.5 if winner == player_id else -0.5

    return delta


def _shaped_reward(
    state_before: WingspanState,
    state_after: WingspanState,
    done: bool,
    player_id: int,
    engine,
) -> float:
    """Potential-based reward shaping: r' = r + γΦ(s') - Φ(s).

    Φ(s) = normalised score. γ = 0.99.
    Preserves the optimal policy of the terminal reward.
    """
    _GAMMA = 0.99
    _MAX_SCORE = 150.0

    base = _terminal_reward(state_after, done, player_id, engine)
    if engine is None:
        return base

    phi_before = engine._compute_final_score(state_before, player_id) / _MAX_SCORE
    phi_after = engine._compute_final_score(state_after, player_id) / _MAX_SCORE

    return base + _GAMMA * phi_after - phi_before
