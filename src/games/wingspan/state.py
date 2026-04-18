"""Wingspan game state — Sprint 2."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from src.games.base.game_state import GameState, PlayerState


class Habitat(str, Enum):
    FOREST = "forest"
    GRASSLAND = "grassland"
    WETLAND = "wetland"


class FoodType(str, Enum):
    SEED = "seed"
    FRUIT = "fruit"
    INVERTEBRATE = "invertebrate"
    RODENT = "rodent"
    FISH = "fish"
    WILD = "wild"


class WingspanPlayerState(PlayerState):
    """Per-player state for Wingspan."""

    habitats: dict[str, list[Any]] = Field(default_factory=dict)
    food_supply: dict[str, int] = Field(default_factory=dict)
    eggs_per_habitat: dict[str, int] = Field(default_factory=dict)


class WingspanState(GameState):
    """Full game state for a Wingspan session — Sprint 2."""

    players: list[WingspanPlayerState] = Field(default_factory=list)  # type: ignore[assignment]
    bird_feeder: dict[str, int] = Field(default_factory=dict)
    bird_tray: list[Any] = Field(default_factory=list)
    draw_deck_count: int = 0
    round: int = 1
    turn_in_round: int = 8
    current_player: int = 0
    round_end_goals: list[Any] = Field(default_factory=list)
