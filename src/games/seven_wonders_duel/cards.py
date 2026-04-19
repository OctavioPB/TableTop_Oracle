"""Card definitions and catalog loader for 7 Wonders Duel — Sprint 7.

Card types:
  raw_material  — produces stone/wood/clay
  manufactured  — produces glass/papyrus
  civilian      — generates victory points
  scientific    — provides science symbols and VPs
  commercial    — generates coins or trade discounts
  military      — provides shields
  guild         — end-game VP based on board state
  wonder        — special buildable structure (not in age decks)
  progress      — progress tokens awarded for science pairs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CardType(str, Enum):
    RAW_MATERIAL = "raw_material"
    MANUFACTURED = "manufactured"
    CIVILIAN = "civilian"
    SCIENTIFIC = "scientific"
    COMMERCIAL = "commercial"
    MILITARY = "military"
    GUILD = "guild"
    WONDER = "wonder"
    PROGRESS = "progress"


class Resource(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    CLAY = "clay"
    GLASS = "glass"
    PAPYRUS = "papyrus"


class ScienceSymbol(str, Enum):
    TABLET = "tablet"
    COMPASS = "compass"
    GEAR = "gear"
    QUILL = "quill"
    BALANCE = "balance"
    MORTAR = "mortar"


ALL_RESOURCES: list[str] = [r.value for r in Resource]
ALL_SCIENCE_SYMBOLS: list[str] = [s.value for s in ScienceSymbol]

# Pyramid layouts per age: list of (row, col) offsets, row 0 = bottom (most accessible)
# Age I: 5-4-3-2-1 layout (23 cards, 3 face-down in rows 1-2)
# Simplified: we track accessibility via the pyramid position model in engine.

N_AGE_1_CARDS = 23
N_AGE_2_CARDS = 23
N_AGE_3_CARDS = 23  # includes 3 guild cards
N_WONDER_CARDS = 12  # total pool; 4 chosen per player
N_PROGRESS_TOKENS = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Card:
    """A single 7 Wonders Duel card.

    Attributes:
        name: Unique card identifier.
        card_type: One of CardType values.
        age: Age (1, 2, 3) or 0 for wonders/progress tokens.
        cost_coins: Direct coin cost to build.
        cost_resources: Dict of resource_name → quantity required.
        effect: Dict describing the card's effect (resources, shields, VPs, etc.).
    """

    name: str
    card_type: str
    age: int = 0
    cost_coins: int = 0
    cost_resources: dict[str, int] = field(default_factory=dict)
    effect: dict[str, Any] = field(default_factory=dict)

    def gives_resource(self, resource: str) -> int:
        """Return how many of resource this card produces each turn."""
        return int(self.effect.get(resource, 0))

    def gives_shields(self) -> int:
        return int(self.effect.get("shields", 0))

    def gives_vp(self) -> int:
        return int(self.effect.get("vp", 0))

    def science_symbols(self) -> list[str]:
        """Return list of science symbols this card provides (may be empty)."""
        return [s for s in ALL_SCIENCE_SYMBOLS if self.effect.get(s, 0) > 0]

    def coin_effect(self) -> int:
        """Immediate coins gained when building this card."""
        return int(self.effect.get("coins", 0))

    def grants_extra_turn(self) -> bool:
        return bool(self.effect.get("extra_turn", 0))


@dataclass
class WonderCard(Card):
    """A wonder structure.  Treated like a card but lives in its own pool."""

    def __post_init__(self) -> None:
        self.card_type = CardType.WONDER.value
        self.age = 0


@dataclass
class ProgressToken:
    """A progress token awarded for collecting two identical science symbols."""

    name: str
    effect: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

_DEFAULT_CATALOG_PATH = Path(
    os.environ.get(
        "CARD_CATALOG_DIR",
        str(Path(__file__).parent.parent.parent.parent / "data" / "card_catalogs"),
    )
) / "seven_wonders_duel_cards.json"


def load_card_catalog(
    path: str | Path | None = None,
) -> tuple[list[Card], list[Card], list[Card], list[WonderCard], list[ProgressToken]]:
    """Load all 7WD cards from the JSON catalog.

    Returns:
        Tuple (age1_cards, age2_cards, age3_cards, wonder_cards, progress_tokens).
        Each age list is ordered as in the JSON (pyramid order assumed at runtime).
    """
    catalog_path = Path(path) if path else _DEFAULT_CATALOG_PATH
    with catalog_path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    def _parse_cards(raw_list: list[dict]) -> list[Card]:
        return [
            Card(
                name=d["name"],
                card_type=d["card_type"],
                age=d.get("age", 0),
                cost_coins=d.get("cost_coins", 0),
                cost_resources=dict(d.get("cost_resources", {})),
                effect=dict(d.get("effect", {})),
            )
            for d in raw_list
        ]

    age1 = _parse_cards(data.get("age_1", []))
    age2 = _parse_cards(data.get("age_2", []))
    age3 = _parse_cards(data.get("age_3", []))

    wonders = [
        WonderCard(
            name=d["name"],
            card_type=CardType.WONDER.value,
            age=0,
            cost_coins=d.get("cost_coins", 0),
            cost_resources=dict(d.get("cost_resources", {})),
            effect=dict(d.get("effect", {})),
        )
        for d in data.get("wonders", [])
    ]

    tokens = [
        ProgressToken(name=d["name"], effect=dict(d.get("effect", {})))
        for d in data.get("progress_tokens", [])
    ]

    return age1, age2, age3, wonders, tokens
