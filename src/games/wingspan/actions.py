"""Wingspan action space — Sprint 2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.games.base.game_state import Action


class WingspanActionType(str, Enum):
    GAIN_FOOD = "gain_food"
    LAY_EGGS = "lay_eggs"
    DRAW_CARDS = "draw_cards"
    PLAY_BIRD = "play_bird"
    ACTIVATE_POWER = "activate_power"


@dataclass
class WingspanAction(Action):
    """A Wingspan-specific action."""

    card_index: int | None = None
    target_habitat: str | None = None
    bird_slot: tuple[str, int] | None = None
    power_choices: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, str):
            self.action_type = self.action_type.value
