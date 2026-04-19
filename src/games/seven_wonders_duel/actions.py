"""Action definitions for 7 Wonders Duel — Sprint 7.

Action space:
  BUILD_CARD(card_name)           — pay cost, gain card effect
  DISCARD_CARD(card_name)         — take 2+commercial_count coins, no card effect
  BUILD_WONDER(card_name, wonder_slot)  — use card as building material for wonder slot

Action index layout (used by gym env):
  Index = card_position * 6 + action_type_offset
  action_type_offsets:
    0 → BUILD_CARD
    1 → DISCARD_CARD
    2 → BUILD_WONDER slot 0
    3 → BUILD_WONDER slot 1
    4 → BUILD_WONDER slot 2
    5 → BUILD_WONDER slot 3

  card_position is the index of the card in the current age_deck list (0..N_CARDS-1).
  N_MAX_ACTIONS = 23 * 6 = 138 → padded to 150.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.games.base.game_state import Action


class SWDActionType(str, Enum):
    BUILD_CARD = "build_card"
    DISCARD_CARD = "discard_card"
    BUILD_WONDER = "build_wonder"


# Number of action type slots per card position
N_ACTION_TYPES_PER_CARD = 6   # build + discard + 4 wonder slots
N_MAX_CARDS_PER_AGE = 23
N_MAX_ACTIONS_7WD = 150        # 23 * 6 = 138, padded to 150


@dataclass
class SWDAction(Action):
    """A fully-specified 7 Wonders Duel action.

    Attributes:
        card_name: Name of the card being used (from age_deck accessible cards).
        wonder_slot: Index (0..3) of the wonder slot being filled; -1 for non-wonder actions.
    """

    card_name: str = ""
    wonder_slot: int = -1     # -1 = not a BUILD_WONDER action

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, str):
            self.action_type = self.action_type.value

    def __str__(self) -> str:
        if self.action_type == SWDActionType.BUILD_WONDER.value:
            return f"BUILD_WONDER({self.card_name} → slot {self.wonder_slot})"
        return f"{self.action_type.upper()}({self.card_name})"
