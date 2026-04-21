"""Splendor game state — Pydantic v2."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from src.games.base.game_state import GameState
from src.games.splendor.cards import GEM_TYPES, GOLD


# ---------------------------------------------------------------------------
# Player board
# ---------------------------------------------------------------------------


class SplendorPlayerBoard:
    """One player's board. Plain Python class for mutation performance."""

    __slots__ = ("player_id", "gems", "cards_owned", "reserved", "nobles_claimed")

    def __init__(
        self,
        player_id: int,
        gems: dict[str, int] | None = None,
        cards_owned: list[str] | None = None,
        reserved: list[str | None] | None = None,
        nobles_claimed: list[str] | None = None,
    ) -> None:
        self.player_id = player_id
        self.gems: dict[str, int] = gems or {g: 0 for g in GEM_TYPES + [GOLD]}
        self.cards_owned: list[str] = cards_owned or []
        # 3 reserve slots; None = empty
        self.reserved: list[str | None] = reserved if reserved is not None else [None, None, None]
        self.nobles_claimed: list[str] = nobles_claimed or []

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def bonus(self) -> dict[str, int]:
        """Permanent gem discount from owned cards."""
        from src.games.splendor.cards import CARDS_BY_ID
        result = {g: 0 for g in GEM_TYPES}
        for cid in self.cards_owned:
            card = CARDS_BY_ID[cid]
            result[card.bonus_color] += 1
        return result

    def vp(self) -> int:
        """Total VP from cards + nobles."""
        from src.games.splendor.cards import CARDS_BY_ID, NOBLES_BY_ID
        total = sum(CARDS_BY_ID[cid].vp for cid in self.cards_owned)
        total += sum(NOBLES_BY_ID[nid].vp for nid in self.nobles_claimed)
        return total

    def total_gems(self) -> int:
        return sum(self.gems.values())

    def n_reserved(self) -> int:
        return sum(1 for s in self.reserved if s is not None)

    def first_empty_reserve_slot(self) -> int | None:
        for i, s in enumerate(self.reserved):
            if s is None:
                return i
        return None

    def effective_cost(self, cost: dict[str, int]) -> dict[str, int]:
        """Net gem cost after applying card bonuses (gold fills any remainder)."""
        bonuses = self.bonus()
        net: dict[str, int] = {}
        for gem, amount in cost.items():
            remaining = max(0, amount - bonuses.get(gem, 0))
            if remaining > 0:
                net[gem] = remaining
        return net

    def can_afford(self, cost: dict[str, int]) -> bool:
        """True if the player can pay net cost using gems + gold."""
        net = self.effective_cost(cost)
        shortfall = 0
        for gem, amount in net.items():
            shortfall += max(0, amount - self.gems.get(gem, 0))
        return shortfall <= self.gems.get(GOLD, 0)

    def payment_for(self, cost: dict[str, int]) -> dict[str, int]:
        """Exact gem payment dict (including gold used) for a given cost."""
        net = self.effective_cost(cost)
        payment: dict[str, int] = {}
        gold_needed = 0
        for gem, amount in net.items():
            have = self.gems.get(gem, 0)
            if have >= amount:
                payment[gem] = amount
            else:
                payment[gem] = have
                gold_needed += amount - have
        if gold_needed > 0:
            payment[GOLD] = gold_needed
        return payment

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def copy(self) -> "SplendorPlayerBoard":
        return SplendorPlayerBoard(
            player_id=self.player_id,
            gems=dict(self.gems),
            cards_owned=list(self.cards_owned),
            reserved=list(self.reserved),
            nobles_claimed=list(self.nobles_claimed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "gems": dict(self.gems),
            "cards_owned": list(self.cards_owned),
            "reserved": list(self.reserved),
            "nobles_claimed": list(self.nobles_claimed),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplendorPlayerBoard":
        return cls(
            player_id=d["player_id"],
            gems=dict(d.get("gems", {})),
            cards_owned=list(d.get("cards_owned", [])),
            reserved=list(d.get("reserved", [None, None, None])),
            nobles_claimed=list(d.get("nobles_claimed", [])),
        )


# ---------------------------------------------------------------------------
# Full game state
# ---------------------------------------------------------------------------


class SplendorState(GameState):
    """Complete Splendor game state — Pydantic v2, JSON-serialisable.

    board[tier-1][slot] = card_id | None  (tier is 1-indexed, stored 0-indexed)
    decks[tier-1] = remaining card_ids not yet revealed
    nobles_available = visible noble_ids not yet claimed
    bank = gem color → count remaining
    """

    players: list[Any] = Field(default_factory=list)

    boards_data: list[dict[str, Any]] = Field(default_factory=list)
    board: list[list[str | None]] = Field(default_factory=list)   # 3 × 4
    decks: list[list[str]] = Field(default_factory=list)          # 3 decks
    nobles_available: list[str] = Field(default_factory=list)
    bank: dict[str, int] = Field(default_factory=dict)
    winner: int | None = None
    final_round: bool = False  # True once any player hits 15 VP

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Board access
    # ------------------------------------------------------------------

    def get_boards(self) -> list[SplendorPlayerBoard]:
        return [SplendorPlayerBoard.from_dict(d) for d in self.boards_data]

    def get_board(self, pid: int) -> SplendorPlayerBoard:
        return SplendorPlayerBoard.from_dict(self.boards_data[pid])

    def with_board(self, pid: int, board: SplendorPlayerBoard) -> "SplendorState":
        new_data = list(self.boards_data)
        new_data[pid] = board.to_dict()
        return self.model_copy(update={"boards_data": new_data})

    def with_boards(self, boards: list[SplendorPlayerBoard]) -> "SplendorState":
        return self.model_copy(update={"boards_data": [b.to_dict() for b in boards]})

    @property
    def current_board(self) -> SplendorPlayerBoard:
        return self.get_board(self.player_id)
