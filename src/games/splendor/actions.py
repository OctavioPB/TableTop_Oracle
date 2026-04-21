"""Splendor action space — flat Discrete(45) layout.

Index layout:
  0–9   take_3_gems     10 combinations of 3 distinct gem types from 5
  10–14 take_2_gems     one per gem type (valid only if bank has ≥4)
  15–26 reserve_board   tier*4+slot  (tier 0-2, slot 0-3) → offset 15
  27–29 reserve_deck    top of tier 1/2/3 → offsets 27/28/29
  30–41 buy_board       tier*4+slot → offset 30
  42–44 buy_reserved    reserved slot 0/1/2 → offsets 42/43/44
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

from src.games.base.game_state import Action
from src.games.splendor.cards import GEM_TYPES

# ---------------------------------------------------------------------------
# Pre-computed index maps
# ---------------------------------------------------------------------------

# 10 sorted 3-gem combinations from 5 types
_TAKE3_COMBOS: list[tuple[str, str, str]] = [
    tuple(sorted(c)) for c in combinations(GEM_TYPES, 3)  # type: ignore[misc]
]

_TAKE3_COMBO_TO_IDX: dict[tuple[str, ...], int] = {
    combo: i for i, combo in enumerate(_TAKE3_COMBOS)
}
_GEM_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GEM_TYPES)}

N_MAX_ACTIONS_SPLENDOR: int = 45

# Index ranges
TAKE3_START = 0
TAKE3_END = 9       # inclusive
TAKE2_START = 10
TAKE2_END = 14
RESERVE_BOARD_START = 15
RESERVE_BOARD_END = 26
RESERVE_DECK_START = 27
RESERVE_DECK_END = 29
BUY_BOARD_START = 30
BUY_BOARD_END = 41
BUY_RESERVED_START = 42
BUY_RESERVED_END = 44


def action_to_index(action: "SplendorAction") -> int:
    """Convert a SplendorAction to its flat integer index."""
    t = action.action_type
    if t == SplendorActionType.TAKE_3_GEMS:
        key = tuple(sorted(action.gems_taken))
        return TAKE3_START + _TAKE3_COMBO_TO_IDX[key]
    if t == SplendorActionType.TAKE_2_GEMS:
        gem = action.gems_taken[0]
        return TAKE2_START + _GEM_TO_IDX[gem]
    if t == SplendorActionType.RESERVE_BOARD:
        return RESERVE_BOARD_START + action.tier * 4 + action.slot
    if t == SplendorActionType.RESERVE_DECK:
        return RESERVE_DECK_START + (action.tier - 1)
    if t == SplendorActionType.BUY_BOARD:
        return BUY_BOARD_START + action.tier * 4 + action.slot
    if t == SplendorActionType.BUY_RESERVED:
        return BUY_RESERVED_START + action.reserve_slot
    raise ValueError(f"Unknown action type: {t}")


def index_to_action_params(idx: int) -> dict[str, Any]:
    """Return kwargs needed to reconstruct a SplendorAction from its index."""
    if TAKE3_START <= idx <= TAKE3_END:
        combo = _TAKE3_COMBOS[idx - TAKE3_START]
        return {"action_type": SplendorActionType.TAKE_3_GEMS, "gems_taken": list(combo)}
    if TAKE2_START <= idx <= TAKE2_END:
        gem = GEM_TYPES[idx - TAKE2_START]
        return {"action_type": SplendorActionType.TAKE_2_GEMS, "gems_taken": [gem, gem]}
    if RESERVE_BOARD_START <= idx <= RESERVE_BOARD_END:
        offset = idx - RESERVE_BOARD_START
        return {"action_type": SplendorActionType.RESERVE_BOARD,
                "tier": offset // 4, "slot": offset % 4}
    if RESERVE_DECK_START <= idx <= RESERVE_DECK_END:
        return {"action_type": SplendorActionType.RESERVE_DECK,
                "tier": (idx - RESERVE_DECK_START) + 1}
    if BUY_BOARD_START <= idx <= BUY_BOARD_END:
        offset = idx - BUY_BOARD_START
        return {"action_type": SplendorActionType.BUY_BOARD,
                "tier": offset // 4, "slot": offset % 4}
    if BUY_RESERVED_START <= idx <= BUY_RESERVED_END:
        return {"action_type": SplendorActionType.BUY_RESERVED,
                "reserve_slot": idx - BUY_RESERVED_START}
    raise ValueError(f"Invalid action index: {idx}")


# ---------------------------------------------------------------------------
# Action class
# ---------------------------------------------------------------------------


class SplendorActionType(str, Enum):
    TAKE_3_GEMS = "take_3_gems"
    TAKE_2_GEMS = "take_2_gems"
    RESERVE_BOARD = "reserve_board"
    RESERVE_DECK = "reserve_deck"
    BUY_BOARD = "buy_board"
    BUY_RESERVED = "buy_reserved"


@dataclass
class SplendorAction(Action):
    """A fully-specified Splendor action."""

    gems_taken: list[str] = field(default_factory=list)   # for TAKE_3 / TAKE_2
    tier: int = 0                                           # 0-indexed (board), 1-indexed (deck)
    slot: int = 0                                           # board slot 0–3
    reserve_slot: int = 0                                   # reserved hand slot 0–2
    payment: dict[str, int] = field(default_factory=dict)  # gem → amount paid (set by validator)
    card_id: str = ""                                       # card being bought/reserved

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, str):
            self.action_type = self.action_type.value

    def __str__(self) -> str:
        t = self.action_type
        if t == SplendorActionType.TAKE_3_GEMS:
            return f"TAKE_3({'+'.join(self.gems_taken)})"
        if t == SplendorActionType.TAKE_2_GEMS:
            gem = self.gems_taken[0] if self.gems_taken else "?"
            return f"TAKE_2({gem}×2)"
        if t == SplendorActionType.RESERVE_BOARD:
            return f"RESERVE_BOARD(tier={self.tier}, slot={self.slot}, card={self.card_id})"
        if t == SplendorActionType.RESERVE_DECK:
            return f"RESERVE_DECK(tier={self.tier})"
        if t == SplendorActionType.BUY_BOARD:
            return f"BUY_BOARD(tier={self.tier}, slot={self.slot}, card={self.card_id})"
        if t == SplendorActionType.BUY_RESERVED:
            return f"BUY_RESERVED(slot={self.reserve_slot}, card={self.card_id})"
        return f"SplendorAction({self.action_type})"
