"""Wingspan action space."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.games.base.game_state import Action


class WingspanActionType(str, Enum):
    GAIN_FOOD = "gain_food"
    LAY_EGGS = "lay_eggs"
    DRAW_CARDS = "draw_cards"
    PLAY_BIRD = "play_bird"


@dataclass
class WingspanAction(Action):
    """A fully-specified Wingspan action.

    The engine's get_legal_actions() returns WingspanAction objects with all
    fields pre-filled (including food_payment and egg_payment). The RL agent
    selects by index; the gym env maps integer → WingspanAction.
    """

    # For GAIN_FOOD: which food-type die to take from the feeder
    food_choice: str = ""           # FoodType.value

    # For DRAW_CARDS: which source to draw the *first* card from;
    # additional entitled cards come from deck
    draw_source: str = "deck"       # "tray_0" | "tray_1" | "tray_2" | "deck"

    # For PLAY_BIRD
    card_name: str = ""             # name of bird to play
    target_habitat: str = ""        # Habitat.value

    # Pre-computed payment (set by LegalMoveValidator, not by agent)
    food_payment: dict[str, int] = field(default_factory=dict)
    egg_payment: int = 0            # number of eggs to remove from board

    # Optional power-resolution choices (for multi-step powers in future sprints)
    power_choices: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, str):
            self.action_type = self.action_type.value

    def __str__(self) -> str:
        if self.action_type == WingspanActionType.GAIN_FOOD:
            return f"GAIN_FOOD({self.food_choice})"
        if self.action_type == WingspanActionType.LAY_EGGS:
            return "LAY_EGGS"
        if self.action_type == WingspanActionType.DRAW_CARDS:
            return f"DRAW_CARDS(src={self.draw_source})"
        if self.action_type == WingspanActionType.PLAY_BIRD:
            return f"PLAY_BIRD({self.card_name} → {self.target_habitat})"
        return f"WingspanAction({self.action_type})"
