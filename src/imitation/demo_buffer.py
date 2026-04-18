"""Demonstration buffer for Behavioural Cloning — Sprint 5."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single (s, a, s', r, done) tuple from a demonstration game."""

    state: Any
    action: Any
    next_state: Any
    reward: float
    done: bool


class DemonstrationBuffer:
    """Stores and samples expert transitions for Behavioural Cloning."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    def add_game(self, transitions: list[Transition]) -> None:
        self._transitions.extend(transitions)

    def sample(self, batch_size: int) -> tuple:
        raise NotImplementedError("S5.2 — implement in Sprint 5")

    def filter_by_winner(self) -> "DemonstrationBuffer":
        raise NotImplementedError("S5.2 — implement in Sprint 5")

    def __len__(self) -> int:
        return len(self._transitions)
