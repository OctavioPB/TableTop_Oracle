"""Wingspan reward shaping — Sprint 2.

Three reward modes available (hyperparameter for ablation):
  terminal: +1 win / -1 loss — pure, slow to converge
  dense:    delta score per action — faster convergence, potential bias
  shaped:   potential-based shaping — theoretically sound per Ng et al. 1999
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
) -> float:
    """Compute the scalar reward for a transition.

    Args:
        state_before: Game state before the action.
        action: The action that was applied.
        state_after: Game state after the action.
        done: Whether the episode ended.
        reward_mode: One of "terminal", "dense", "shaped".

    Returns:
        Scalar reward signal.
    """
    raise NotImplementedError("S2.6 — implement in Sprint 2")
