"""Tabletop Simulator log parser — Sprint 5."""

from __future__ import annotations

import logging

from src.imitation.demo_buffer import Transition

logger = logging.getLogger(__name__)


class TTSLogParser:
    """Parses Tabletop Simulator Wingspan history JSON into Transitions."""

    def parse_game_log(self, raw_log: dict) -> list[Transition]:
        raise NotImplementedError("S5.1 — implement in Sprint 5")
