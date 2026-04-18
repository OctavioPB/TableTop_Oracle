"""BoardGameArena log parser — Sprint 5."""

from __future__ import annotations

import logging
from pathlib import Path

from src.imitation.demo_buffer import Transition

logger = logging.getLogger(__name__)


class BGALogParser:
    """Parses BGA Wingspan game logs into engine Transitions."""

    def parse_game_log(self, raw_log: dict) -> list[Transition]:
        """Map BGA events to WingspanAction / WingspanState pairs.

        Args:
            raw_log: Raw BGA log dict.

        Returns:
            List of Transition objects for this game.
        """
        raise NotImplementedError("S5.1 — implement in Sprint 5")
