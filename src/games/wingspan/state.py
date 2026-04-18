"""Wingspan game state — full Pydantic v2 models."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from src.games.base.game_state import GameState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TURNS_PER_ROUND: dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5}
N_HABITAT_SLOTS = 5
N_ROUNDS = 4
N_PLAYERS = 2  # MVP: 2-player only


# ---------------------------------------------------------------------------
# Per-slot state
# ---------------------------------------------------------------------------


class BirdSlotState:
    """State of one occupied slot. Plain Python class for performance."""

    __slots__ = ("bird_name", "eggs", "cached_food", "tucked_cards")

    def __init__(
        self,
        bird_name: str,
        eggs: int = 0,
        cached_food: dict[str, int] | None = None,
        tucked_cards: int = 0,
    ) -> None:
        self.bird_name = bird_name
        self.eggs = eggs
        self.cached_food: dict[str, int] = cached_food or {}
        self.tucked_cards = tucked_cards

    def copy(self) -> "BirdSlotState":
        return BirdSlotState(
            bird_name=self.bird_name,
            eggs=self.eggs,
            cached_food=dict(self.cached_food),
            tucked_cards=self.tucked_cards,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bird_name": self.bird_name,
            "eggs": self.eggs,
            "cached_food": self.cached_food,
            "tucked_cards": self.tucked_cards,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BirdSlotState":
        return cls(
            bird_name=d["bird_name"],
            eggs=d.get("eggs", 0),
            cached_food=d.get("cached_food", {}),
            tucked_cards=d.get("tucked_cards", 0),
        )

    def __repr__(self) -> str:
        return f"BirdSlotState({self.bird_name}, eggs={self.eggs})"


# ---------------------------------------------------------------------------
# Player board
# ---------------------------------------------------------------------------


class WingspanPlayerBoard:
    """One player's complete board. Plain class for mutable RL performance.

    Habitats store 5 slots each. None = empty slot, BirdSlotState = occupied.
    """

    __slots__ = (
        "player_id",
        "forest",
        "grassland",
        "wetland",
        "food_supply",
        "hand",
        "bonus_card",
        "action_cubes",
        "goal_positions",
    )

    def __init__(
        self,
        player_id: int,
        forest: list | None = None,
        grassland: list | None = None,
        wetland: list | None = None,
        food_supply: dict[str, int] | None = None,
        hand: list[str] | None = None,
        bonus_card: str = "",
        action_cubes: int = 8,
        goal_positions: list[int] | None = None,
    ) -> None:
        self.player_id = player_id
        self.forest: list[BirdSlotState | None] = forest if forest is not None else [None] * 5
        self.grassland: list[BirdSlotState | None] = grassland if grassland is not None else [None] * 5
        self.wetland: list[BirdSlotState | None] = wetland if wetland is not None else [None] * 5
        self.food_supply: dict[str, int] = food_supply or {}
        self.hand: list[str] = hand or []
        self.bonus_card: str = bonus_card
        self.action_cubes: int = action_cubes
        self.goal_positions: list[int] = goal_positions or [0, 0, 0, 0]

    # ------------------------------------------------------------------
    # Habitat helpers
    # ------------------------------------------------------------------

    def get_habitat(self, habitat: str) -> list[BirdSlotState | None]:
        if habitat == "forest":
            return self.forest
        if habitat == "grassland":
            return self.grassland
        if habitat == "wetland":
            return self.wetland
        raise ValueError(f"Unknown habitat: {habitat}")

    def birds_in_habitat(self, habitat: str) -> list[tuple[int, BirdSlotState]]:
        """Return (slot_idx, slot_state) for occupied slots in habitat."""
        return [(i, s) for i, s in enumerate(self.get_habitat(habitat)) if s is not None]

    def all_birds(self) -> list[tuple[str, int, BirdSlotState]]:
        """Return (habitat, slot_idx, slot_state) for all occupied slots."""
        result = []
        for h in ("forest", "grassland", "wetland"):
            for i, s in self.birds_in_habitat(h):
                result.append((h, i, s))
        return result

    def total_birds(self) -> int:
        return sum(s is not None for h in ("forest", "grassland", "wetland") for s in self.get_habitat(h))

    def total_eggs(self) -> int:
        return sum(s.eggs for _, _, s in self.all_birds())

    def total_cached_food(self) -> int:
        return sum(sum(s.cached_food.values()) for _, _, s in self.all_birds())

    def total_tucked_cards(self) -> int:
        return sum(s.tucked_cards for _, _, s in self.all_birds())

    def total_food(self) -> int:
        return sum(self.food_supply.values())

    def first_empty_slot(self, habitat: str) -> int | None:
        """Return index of leftmost empty slot in habitat, or None."""
        for i, s in enumerate(self.get_habitat(habitat)):
            if s is None:
                return i
        return None

    def next_slot_index(self, habitat: str) -> int:
        """Number of occupied slots in habitat (= index of next bird to play)."""
        return sum(1 for s in self.get_habitat(habitat) if s is not None)

    def egg_cost_for_habitat(self, habitat: str) -> int:
        """Eggs to discard when playing a bird into this habitat."""
        # Column:  1  2  3  4  5
        # Eggs:    0  1  1  2  2
        col = self.next_slot_index(habitat) + 1  # 1-indexed
        return [0, 0, 1, 1, 2, 2][min(col, 5)]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def copy(self) -> "WingspanPlayerBoard":
        return WingspanPlayerBoard(
            player_id=self.player_id,
            forest=[s.copy() if s else None for s in self.forest],
            grassland=[s.copy() if s else None for s in self.grassland],
            wetland=[s.copy() if s else None for s in self.wetland],
            food_supply=dict(self.food_supply),
            hand=list(self.hand),
            bonus_card=self.bonus_card,
            action_cubes=self.action_cubes,
            goal_positions=list(self.goal_positions),
        )

    def to_dict(self) -> dict[str, Any]:
        def _hab(slots: list) -> list:
            return [s.to_dict() if s else None for s in slots]

        return {
            "player_id": self.player_id,
            "forest": _hab(self.forest),
            "grassland": _hab(self.grassland),
            "wetland": _hab(self.wetland),
            "food_supply": self.food_supply,
            "hand": self.hand,
            "bonus_card": self.bonus_card,
            "action_cubes": self.action_cubes,
            "goal_positions": self.goal_positions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WingspanPlayerBoard":
        def _hab(lst: list) -> list:
            return [BirdSlotState.from_dict(s) if s else None for s in lst]

        return cls(
            player_id=d["player_id"],
            forest=_hab(d.get("forest", [None] * 5)),
            grassland=_hab(d.get("grassland", [None] * 5)),
            wetland=_hab(d.get("wetland", [None] * 5)),
            food_supply=dict(d.get("food_supply", {})),
            hand=list(d.get("hand", [])),
            bonus_card=d.get("bonus_card", ""),
            action_cubes=d.get("action_cubes", 8),
            goal_positions=list(d.get("goal_positions", [0, 0, 0, 0])),
        )


# ---------------------------------------------------------------------------
# Full game state
# ---------------------------------------------------------------------------


class WingspanState(GameState):
    """Complete Wingspan game state — Pydantic v2, JSON-serialisable.

    Inherits player_id (current player), turn, and phase from GameState.
    The `players` field from GameState is unused; `boards` holds all data.
    """

    # Override base class 'players' so Pydantic doesn't validate them
    players: list[Any] = Field(default_factory=list)

    # Wingspan-specific
    boards_data: list[dict[str, Any]] = Field(default_factory=list)
    bird_feeder: list[str] = Field(default_factory=list)  # FoodType values per die
    bird_tray: list[str] = Field(default_factory=list)  # 0-3 bird names
    draw_deck: list[str] = Field(default_factory=list)
    discard_pile: list[str] = Field(default_factory=list)
    round: int = 1
    round_end_goals: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Board access (boards are not stored as Pydantic to avoid deep nesting)
    # ------------------------------------------------------------------

    def get_boards(self) -> list[WingspanPlayerBoard]:
        return [WingspanPlayerBoard.from_dict(d) for d in self.boards_data]

    def get_board(self, player_id: int) -> WingspanPlayerBoard:
        return WingspanPlayerBoard.from_dict(self.boards_data[player_id])

    def with_boards(self, boards: list[WingspanPlayerBoard]) -> "WingspanState":
        """Return a new state with updated boards."""
        return self.model_copy(
            update={"boards_data": [b.to_dict() for b in boards]}, deep=False
        )

    def with_board(self, player_id: int, board: WingspanPlayerBoard) -> "WingspanState":
        new_data = list(self.boards_data)
        new_data[player_id] = board.to_dict()
        return self.model_copy(update={"boards_data": new_data}, deep=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def current_board(self) -> WingspanPlayerBoard:
        return self.get_board(self.player_id)

    def all_players_done(self) -> bool:
        """True if all players have 0 action cubes remaining."""
        return all(
            WingspanPlayerBoard.from_dict(d).action_cubes == 0
            for d in self.boards_data
        )

    def model_dump_json(self, **kw: Any) -> str:
        """JSON representation for prompts / logging."""
        import json

        d = {
            "player_id": self.player_id,
            "turn": self.turn,
            "round": self.round,
            "phase": self.phase,
            "bird_feeder": self.bird_feeder,
            "bird_tray": self.bird_tray,
            "draw_deck_count": len(self.draw_deck),
            "boards": self.boards_data,
        }
        return json.dumps(d, indent=kw.get("indent", 2))
