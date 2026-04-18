"""Wingspan legal move validator — deterministic Python, no LLM at runtime."""

from __future__ import annotations

import logging

from src.games.wingspan.actions import WingspanAction
from src.games.wingspan.state import WingspanState

logger = logging.getLogger(__name__)


class LegalMoveValidator:
    """Validates Wingspan actions against the current game state.

    All logic is pure Python. The Rule Oracle (LLM) is only consulted
    for edge cases not covered here — never per-step during RL training.
    """

    def get_legal_play_bird_actions(self, state: WingspanState) -> list[WingspanAction]:
        raise NotImplementedError("S2.5 — implement in Sprint 2")

    def get_legal_gain_food_actions(self, state: WingspanState) -> list[WingspanAction]:
        raise NotImplementedError("S2.5 — implement in Sprint 2")

    def get_legal_lay_eggs_actions(self, state: WingspanState) -> list[WingspanAction]:
        raise NotImplementedError("S2.5 — implement in Sprint 2")

    def get_legal_draw_cards_actions(self, state: WingspanState) -> list[WingspanAction]:
        raise NotImplementedError("S2.5 — implement in Sprint 2")

    def validate_action(
        self, state: WingspanState, action: WingspanAction
    ) -> tuple[bool, str]:
        """Return (is_legal, reason) for the proposed action."""
        raise NotImplementedError("S2.5 — implement in Sprint 2")
