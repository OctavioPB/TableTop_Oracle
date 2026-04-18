from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class PlayerState(BaseModel):
    """Resources and hand for a single player."""

    player_id: int
    hand: list[Any] = Field(default_factory=list)
    score: int = 0
    resources: dict[str, int] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class GameState(BaseModel, ABC):
    """Abstract base for all game states.

    Subclasses must add game-specific fields. Kept as Pydantic so state
    is always serialisable to JSON for logging and caching.
    """

    player_id: int
    turn: int
    phase: str
    players: list[PlayerState]
    shared_board: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def model_copy_deep(self) -> "GameState":
        return self.model_copy(deep=True)


@dataclass
class Action:
    """A player action: type name + arbitrary parameters."""

    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    player_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "params": self.params,
            "player_id": self.player_id,
        }


@dataclass
class ActionResult:
    """The result of applying an action to a game state."""

    success: bool
    new_state: GameState
    events: list[str] = field(default_factory=list)
    reward: float = 0.0
