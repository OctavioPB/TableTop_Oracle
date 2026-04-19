"""Game state for 7 Wonders Duel — Sprint 7.

State is immutable in the public API (model_copy for transitions).
Pydantic v2 is used for all data models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.games.seven_wonders_duel.cards import ALL_RESOURCES, ALL_SCIENCE_SYMBOLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_AGES = 3
N_WONDERS_PER_PLAYER = 4      # each player builds up to 4 wonders
N_PROGRESS_TOKENS_AVAILABLE = 5   # shown face-up at game start
MILITARY_TRACK_MAX = 9            # 0..9, center = 4 (P0 side: 0..3, center: 4, P1 side: 5..9)
MILITARY_SUPREMACY_THRESHOLD = 0  # P1 loses if token reaches 0; P0 loses if reaches 9

# Military conflict pawn starts in the center (position 4 in 0-indexed 0..8 track)
MILITARY_CENTER = 4

# 7 Wonders Duel pyramid card counts per age row (bottom to top):
# Age 1: 2(face-up) 3(face-down) 4(face-up) ... varies by edition.
# Simplified: we track accessible cards via a flat list + a set of "covered" indices.
# The real pyramid rules are: a card is accessible if both cards below it are taken.
# We encode each age as a flat list of card names + a list of "face_down" positions.

AGE_PYRAMID_LAYOUTS: dict[int, list[list[int]]] = {
    # Each inner list = card indices in a row; row 0 = top (1 card)
    # A card at position P in row R is accessible only when both cards
    # in row R+1 covering it are removed.
    # Age 1 layout (23 cards):  row sizes [2, 3, 4, 5, 6, 3] — simplified 6-row layout
    1: [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22]],
    # Age 2 layout (23 cards): inverted pyramid [6, 5, 4, 3, 2, 3]
    2: [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17], [18, 19], [20, 21, 22]],
    # Age 3 layout (23 cards): sphinx/castle layout [3, 4, 5, 4, 3, 2, 2]
    3: [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18], [19, 20], [21, 22]],
}

# Indices of face-down cards in each age (hidden from opponent until uncovered)
AGE_FACE_DOWN: dict[int, set[int]] = {
    1: {2, 3, 4, 9, 10, 11, 12, 13},
    2: {6, 7, 8, 9, 10, 18, 19},
    3: {7, 8, 9, 10, 11},
}


# ---------------------------------------------------------------------------
# Player board
# ---------------------------------------------------------------------------


class SWDPlayerBoard(BaseModel):
    """State of one player in 7 Wonders Duel.

    Attributes:
        player_id: 0 or 1.
        coins: Current coin count (starts at 7).
        resources: Produced resources (wood/stone/clay/glass/papyrus counts).
        discounts: Set of resource names with discount (cost = 1 for that resource).
        shields: Military shield count.
        science_symbols: Dict of symbol_name → count.
        built_cards: List of card names already built (for scoring and VP effects).
        wonders: List of 4 wonder names assigned to this player.
        built_wonders: List of booleans, True if wonder slot i is built.
        progress_tokens: List of progress token names owned.
        vp_from_wonders: Running VP total from built wonders.
        vp_from_cards: Running VP total from civilian/guild/commercial cards.
    """

    model_config = {"frozen": False}

    player_id: int
    coins: int = 7
    resources: dict[str, int] = Field(default_factory=lambda: {r: 0 for r in ALL_RESOURCES})
    discounts: dict[str, bool] = Field(default_factory=dict)
    shields: int = 0
    science_symbols: dict[str, int] = Field(
        default_factory=lambda: {s: 0 for s in ALL_SCIENCE_SYMBOLS}
    )
    built_cards: list[str] = Field(default_factory=list)
    wonders: list[str] = Field(default_factory=list)
    built_wonders: list[bool] = Field(default_factory=lambda: [False, False, False, False])
    progress_tokens: list[str] = Field(default_factory=list)
    vp_from_wonders: int = 0
    vp_from_cards: int = 0

    def total_resources(self) -> dict[str, int]:
        return dict(self.resources)

    def n_built_wonders(self) -> int:
        return sum(1 for b in self.built_wonders if b)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SWDPlayerBoard:
        data = dict(d)
        data["resources"] = dict(data.get("resources", {}))
        data["science_symbols"] = dict(data.get("science_symbols", {}))
        data["built_cards"] = list(data.get("built_cards", []))
        data["wonders"] = list(data.get("wonders", []))
        data["built_wonders"] = list(data.get("built_wonders", [False, False, False, False]))
        data["progress_tokens"] = list(data.get("progress_tokens", []))
        data["discounts"] = dict(data.get("discounts", {}))
        return cls(**data)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


class SWDState(BaseModel):
    """Full game state for 7 Wonders Duel.

    Attributes:
        player_id: Whose turn it is (0 or 1).
        age: Current age (1, 2, or 3).
        turn: Global turn counter (increments each action).
        phase: "main" or "finished".
        age_deck: Ordered list of card names in the current age pyramid.
        taken_cards: Set of card names already removed from the pyramid.
        face_down_cards: Set of card names still face-down (not yet uncovered).
        boards_data: List of serialised SWDPlayerBoard dicts (index = player_id).
        progress_tokens_available: List of face-up progress token names.
        progress_tokens_taken: List of all taken progress token names (both players).
        military_pawn: Position of the conflict pawn (0=P0 side, 9=P1 side, 4=center).
        discard_pile: List of discarded card names (for Mausoleum wonder).
        winner: player_id of winner, or None if game is ongoing.
    """

    model_config = {"frozen": False}

    player_id: int = 0
    age: int = 1
    turn: int = 0
    phase: str = "main"
    age_deck: list[str] = Field(default_factory=list)
    taken_cards: set[str] = Field(default_factory=set)
    face_down_cards: set[str] = Field(default_factory=set)
    boards_data: list[dict[str, Any]] = Field(default_factory=list)
    progress_tokens_available: list[str] = Field(default_factory=list)
    progress_tokens_taken: list[str] = Field(default_factory=list)
    military_pawn: int = MILITARY_CENTER
    discard_pile: list[str] = Field(default_factory=list)
    winner: int | None = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_board(self, player_id: int) -> SWDPlayerBoard:
        return SWDPlayerBoard.from_dict(self.boards_data[player_id])

    def get_boards(self) -> list[SWDPlayerBoard]:
        return [SWDPlayerBoard.from_dict(d) for d in self.boards_data]

    def with_board(self, player_id: int, board: SWDPlayerBoard) -> SWDState:
        """Return a new state with board[player_id] replaced."""
        new_boards = list(self.boards_data)
        new_boards[player_id] = board.to_dict()
        return self.model_copy(update={"boards_data": new_boards})

    def accessible_cards(self, catalog_by_age: list[str]) -> list[str]:
        """Return list of card names currently accessible in the pyramid.

        A card is accessible if it is not yet taken AND both cards below it
        (if any) have been taken.
        """
        layout = AGE_PYRAMID_LAYOUTS[self.age]
        accessible: list[str] = []

        # Build a lookup: position_index → card_name
        deck = self.age_deck
        if not deck:
            return []

        # For each row, a card at column c is covered by cards at row+1, cols c and c+1
        # We work bottom-up: bottom row cards are always accessible if not taken.
        taken = self.taken_cards

        for row_idx, row in enumerate(layout):
            for col_idx, card_pos in enumerate(row):
                if card_pos >= len(deck):
                    continue
                card_name = deck[card_pos]
                if card_name in taken:
                    continue

                # Check if covered by cards in the next row
                if row_idx + 1 < len(layout):
                    next_row = layout[row_idx + 1]
                    # Cards in next_row that cover this position are at col_idx and col_idx+1
                    covering_positions = []
                    if col_idx < len(next_row):
                        covering_positions.append(next_row[col_idx])
                    if col_idx + 1 < len(next_row):
                        covering_positions.append(next_row[col_idx + 1])

                    covered_by_remaining = any(
                        cp < len(deck) and deck[cp] not in taken
                        for cp in covering_positions
                    )
                    if covered_by_remaining:
                        continue

                accessible.append(card_name)

        return accessible

    def is_face_down(self, card_name: str) -> bool:
        return card_name in self.face_down_cards

    def military_winner(self) -> int | None:
        """Return player_id who won by military supremacy, or None."""
        if self.military_pawn <= 0:
            return 0   # pawn at P1 capital → P0 wins
        if self.military_pawn >= MILITARY_TRACK_MAX - 1:
            return 1   # pawn at P0 capital → P1 wins
        return None

    def science_winner(self, catalog: dict[str, Any]) -> int | None:
        """Return player_id who won by science supremacy (6 distinct symbols), or None."""
        for pid in range(2):
            board = self.get_board(pid)
            n_distinct = sum(1 for s, c in board.science_symbols.items() if c > 0)
            if n_distinct >= 6:
                return pid
        return None
