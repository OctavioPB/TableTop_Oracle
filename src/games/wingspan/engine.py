"""Wingspan game engine — Sprint 2."""

from __future__ import annotations

import logging

from src.games.base.engine import GameEngine
from src.games.base.game_state import Action, ActionResult, GameState
from src.games.wingspan.state import WingspanState

logger = logging.getLogger(__name__)


class WingspanEngine(GameEngine):
    """Deterministic Wingspan simulator.

    Implements the 4 core actions (gain food, lay eggs, draw cards, play bird)
    plus bird power resolution for the base set.
    """

    def reset(self) -> WingspanState:
        """Return initial state for a 2-player Wingspan game."""
        raise NotImplementedError("S2.4 — implement in Sprint 2")

    def step(self, state: GameState, action: Action) -> ActionResult:
        """Apply action and return the resulting state."""
        raise NotImplementedError("S2.4 — implement in Sprint 2")

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Return all legal actions for the current player.

        Must never return [] when is_terminal(state) is False.
        """
        raise NotImplementedError("S2.4 — implement in Sprint 2")

    def is_terminal(self, state: GameState) -> bool:
        """Game ends after round 4 is complete."""
        raise NotImplementedError("S2.4 — implement in Sprint 2")

    def get_winner(self, state: GameState) -> int | None:
        """Return player_id of highest scorer, or None if game not over."""
        raise NotImplementedError("S2.4 — implement in Sprint 2")
