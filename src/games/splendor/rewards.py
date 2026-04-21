"""Splendor reward functions."""

from __future__ import annotations

from enum import Enum

from src.games.base.game_state import ActionResult
from src.games.splendor.state import SplendorState


class RewardMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"


def compute_reward(
    prev_state: SplendorState,
    result: ActionResult,
    mode: RewardMode = RewardMode.DENSE,
) -> float:
    """Compute the reward for transitioning from prev_state via result.

    Dense: VP gained this step.
    Sparse: +1 on win, -1 on loss, 0 otherwise.
    """
    new_state: SplendorState = result.new_state  # type: ignore[assignment]

    if mode == RewardMode.SPARSE:
        if new_state.winner == 0:
            return 1.0
        if new_state.winner == 1:
            return -1.0
        return 0.0

    # Dense: VP delta for player 0
    prev_vp = prev_state.get_board(0).vp()
    new_vp = new_state.get_board(0).vp()
    return float(new_vp - prev_vp)
