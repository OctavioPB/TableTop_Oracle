"""Wingspan bird card model and catalog loader."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


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
    WILD = "wild"  # "any food" — not a real die face


class NestType(str, Enum):
    CUP = "cup"
    PLATFORM = "platform"
    CAVITY = "cavity"
    GROUND = "ground"
    STAR = "star"
    WILD = "wild"


class PowerTiming(str, Enum):
    WHEN_PLAYED = "when_played"
    WHEN_ACTIVATED = "when_activated"  # brown — activates when habitat action taken
    ONCE_BETWEEN_TURNS = "once_between_turns"  # pink
    ALL_PLAYERS = "all_players"  # white (flocking)
    END_OF_ROUND = "end_of_round"  # teal
    NO_POWER = "no_power"


class PowerType(str, Enum):
    ACCUMULATE = "accumulate"  # cache food, tuck cards
    PRODUCTIVE = "productive"  # lay eggs, gain food, draw cards
    PREDATOR = "predator"
    FLOCKING = "flocking"  # white all-player power
    OTHER = "other"
    NONE = "none"


class PowerID(str, Enum):
    """Structured power identifier for engine dispatch.

    Every supported power maps to a deterministic handler.
    COMPLEX = not yet implemented, no effect in MVP.
    """

    NONE = "none"
    # When-activated (brown) — habitat action trigger
    GAIN_FOOD_FEEDER = "gain_food_feeder"      # gain 1 food from feeder
    GAIN_FOOD_SUPPLY = "gain_food_supply"       # gain 1 food of specific type from supply
    LAY_EGG_SELF = "lay_egg_self"              # lay 1 egg on this bird
    LAY_EGG_ANY = "lay_egg_any"                # lay 1 egg on any bird
    DRAW_CARD = "draw_card"                    # draw 1 card from deck
    TUCK_CARD_DECK = "tuck_card_deck"          # tuck 1 card from top of deck
    CACHE_FOOD = "cache_food"                  # cache 1 food from supply on this bird
    PREDATOR = "predator"                      # hunt: tuck if wingspan ≤ power_param
    # When-played (brown flash)
    WHEN_PLAYED_DRAW = "when_played_draw"      # draw N cards (power_param = N)
    WHEN_PLAYED_GAIN_FOOD = "when_played_gain_food"  # gain N food (power_param = N)
    WHEN_PLAYED_LAY_EGG = "when_played_lay_egg"      # lay N eggs (power_param = N)
    # Once-between-turns (pink)
    ONCE_BTW_LAY_EGG = "once_btw_lay_egg"     # lay 1 egg when another player lays
    ONCE_BTW_GAIN_FOOD = "once_btw_gain_food"  # gain 1 food when another player gains food
    ONCE_BTW_DRAW = "once_btw_draw"            # draw 1 when another player draws
    # White (all players)
    FLOCKING = "flocking"                      # all players: tuck 1 and gain 1 egg
    # Teal (end of round)
    END_ROUND_LAY_EGG = "end_round_lay_egg"   # lay 1 egg at end of round
    # Stub
    COMPLEX = "complex"                        # multi-step power, no effect in MVP


class RoundGoalType(str, Enum):
    MOST_BIRDS_FOREST = "most_birds_forest"
    MOST_BIRDS_GRASSLAND = "most_birds_grassland"
    MOST_BIRDS_WETLAND = "most_birds_wetland"
    MOST_EGGS_CUP = "most_eggs_cup"
    MOST_EGGS_PLATFORM = "most_eggs_platform"
    MOST_EGGS_CAVITY = "most_eggs_cavity"
    MOST_EGGS_GROUND = "most_eggs_ground"
    MOST_BIRDS_TOTAL = "most_birds_total"


# 8 available goals; 4 are randomly selected per game
ALL_ROUND_GOALS: list[str] = [g.value for g in RoundGoalType]

# Points awarded per position in 2-player game
GOAL_POINTS_2P: dict[int, int] = {1: 5, 2: 1}


# ---------------------------------------------------------------------------
# Bird card model
# ---------------------------------------------------------------------------


class BirdCard(BaseModel):
    """One bird card from the Wingspan base set."""

    name: str
    habitats: list[str]          # Habitat.value list
    food_cost: dict[str, int]    # FoodType.value → count
    nest_type: str               # NestType.value
    egg_limit: int
    points: int
    wingspan_cm: int
    power_timing: str            # PowerTiming.value
    power_id: str                # PowerID.value
    power_type: str              # PowerType.value
    power_text: str = ""
    power_param: int = 0         # e.g., wingspan threshold for predator, N for gain N
    card_id: str = ""

    model_config = {"frozen": True}

    @property
    def total_food_cost(self) -> int:
        return sum(self.food_cost.values())

    def can_play_in(self, habitat: str) -> bool:
        return habitat in self.habitats


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------


def _parse_food_cost(raw: str) -> dict[str, int]:
    """Parse 'seed:1,invertebrate:2' into {'seed': 1, 'invertebrate': 2}."""
    if not raw or str(raw).strip() in ("", "nan", "0", "none"):
        return {}
    result: dict[str, int] = {}
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            food, count = part.split(":", 1)
            result[food.strip()] = int(count.strip())
        else:
            # Just a number → wild cost
            result[FoodType.WILD.value] = int(part)
    return result


def _parse_habitats(raw: str) -> list[str]:
    """Parse 'forest,grassland' into ['forest', 'grassland']."""
    return [h.strip() for h in str(raw).split(",") if h.strip()]


def load_bird_catalog(csv_path: str | Path) -> dict[str, BirdCard]:
    """Load the bird catalog from CSV and return a name → BirdCard mapping.

    Args:
        csv_path: Path to wingspan_birds.csv.

    Returns:
        Dict mapping bird name → BirdCard.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Bird catalog not found: {path}")

    df = pd.read_csv(path, dtype=str).fillna("")
    catalog: dict[str, BirdCard] = {}

    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            continue

        card = BirdCard(
            name=name,
            habitats=_parse_habitats(row.get("habitat", "")),
            food_cost=_parse_food_cost(row.get("food_cost", "")),
            nest_type=str(row.get("nest_type", "cup")).strip(),
            egg_limit=int(row.get("egg_limit", 2)),
            points=int(row.get("points", 0)),
            wingspan_cm=int(row.get("wingspan_cm", 50)),
            power_timing=str(row.get("power_timing", PowerTiming.NO_POWER.value)).strip(),
            power_id=str(row.get("power_id", PowerID.NONE.value)).strip(),
            power_type=str(row.get("power_type", PowerType.NONE.value)).strip(),
            power_text=str(row.get("power_text", "")).strip(),
            power_param=int(row.get("power_param", 0)) if row.get("power_param", "") else 0,
            card_id=str(row.get("card_id", name)).strip() or name,
        )
        catalog[name] = card

    logger.info("Loaded %d birds from %s", len(catalog), path.name)
    return catalog
