"""Reward functions for 7 Wonders Duel — Sprint 7.

Two modes mirror the Wingspan design:
  sparse — +1 win / -1 loss at game end, 0 otherwise.
  dense  — incremental reward each step based on VPs, military, and science progress.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class RewardMode(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"


def compute_reward(
    state_before: Any,
    state_after: Any,
    player_id: int,
    done: bool,
    winner: int | None,
    mode: str = RewardMode.DENSE.value,
) -> float:
    """Compute per-step reward for player_id.

    Args:
        state_before: SWDState before the action.
        state_after: SWDState after the action.
        player_id: Agent player_id (0 or 1).
        done: Whether the game is over.
        winner: Winner player_id, or None on draw/timeout.
        mode: "sparse" or "dense".

    Returns:
        Float reward value.
    """
    if done:
        if winner == player_id:
            return 1.0
        if winner is None:
            return 0.0
        return -1.0

    if mode == RewardMode.SPARSE.value:
        return 0.0

    # Dense: incremental VP, military progress, science pairs
    reward = 0.0

    board_before = state_before.get_board(player_id)
    board_after = state_after.get_board(player_id)

    # VP delta from newly built cards/wonders
    vp_delta = (
        board_after.vp_from_cards + board_after.vp_from_wonders
        - board_before.vp_from_cards - board_before.vp_from_wonders
    )
    reward += vp_delta * 0.05

    # Military progress (closing gap or extending lead)
    if player_id == 0:
        military_delta = state_before.military_pawn - state_after.military_pawn
    else:
        military_delta = state_after.military_pawn - state_before.military_pawn
    reward += military_delta * 0.03

    # Science symbol count delta
    sci_before = sum(v for v in board_before.science_symbols.values())
    sci_after = sum(v for v in board_after.science_symbols.values())
    reward += (sci_after - sci_before) * 0.02

    return reward
