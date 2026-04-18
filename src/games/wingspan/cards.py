"""Wingspan bird card dataclass and catalog loader — Sprint 2."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

PowerTiming = Literal["when_played", "once_between_turns", "when_activated", "end_of_round"]
PowerType = Literal["accumulate", "productive", "flocking", "predator", "other", "none"]
NestType = Literal["cup", "platform", "cavity", "ground", "wild", "star"]


@dataclass
class BirdCard:
    """Represents a single bird card in Wingspan."""

    name: str
    habitat: list[str]
    cost_food: dict[str, int]
    nest_type: NestType
    egg_limit: int
    points: int
    power_timing: PowerTiming
    power_type: PowerType
    power_text: str
    card_id: str = ""
    wingspan_cm: int = 0
    extra_metadata: dict = field(default_factory=dict)


def load_bird_catalog(csv_path: str | Path) -> list[BirdCard]:
    """Load the bird catalog from a CSV file.

    Args:
        csv_path: Path to wingspan_birds.csv.

    Returns:
        List of BirdCard objects.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    raise NotImplementedError("S2.1 — implement in Sprint 2")
