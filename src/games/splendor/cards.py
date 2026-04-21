"""Splendor card catalog — representative 45-card + 5-noble set.

Covers the full range of mechanics: all 5 gem colors, all 3 tiers, VPs 0-5,
costs from 2 to 9 gems. Enough variety for RL to learn meaningful strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEM_TYPES: list[str] = ["white", "blue", "green", "red", "black"]
GOLD = "gold"
ALL_COLORS = GEM_TYPES + [GOLD]

VICTORY_POINTS_TARGET = 15  # first player to reach this (then finish the round)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplendorCard:
    """A development card."""

    card_id: str
    tier: int                           # 1, 2, or 3
    bonus_color: str                    # permanent gem discount color
    vp: int                             # victory points (0–5)
    cost: dict[str, int] = field(default_factory=dict)  # gem color → amount

    def __hash__(self) -> int:
        return hash(self.card_id)


@dataclass(frozen=True)
class Noble:
    """A noble tile — auto-claimed when card bonus requirements are met."""

    noble_id: str
    vp: int                              # always 3 in standard Splendor
    requirements: dict[str, int] = field(default_factory=dict)  # bonus color → count

    def __hash__(self) -> int:
        return hash(self.noble_id)


# ---------------------------------------------------------------------------
# Card catalog
# ---------------------------------------------------------------------------

# Tier 1 — low cost, mostly 0 VP, cheap engine building
_TIER1: list[SplendorCard] = [
    # White bonus cards
    SplendorCard("t1_w_0a", 1, "white", 0, {"blue": 3}),
    SplendorCard("t1_w_0b", 1, "white", 0, {"green": 2, "red": 1}),
    SplendorCard("t1_w_0c", 1, "white", 0, {"black": 2, "blue": 1}),
    SplendorCard("t1_w_0d", 1, "white", 0, {"red": 2, "green": 1, "blue": 1}),
    SplendorCard("t1_w_1a", 1, "white", 1, {"blue": 4}),
    # Blue bonus cards
    SplendorCard("t1_b_0a", 1, "blue", 0, {"white": 3}),
    SplendorCard("t1_b_0b", 1, "blue", 0, {"red": 2, "white": 1}),
    SplendorCard("t1_b_0c", 1, "blue", 0, {"black": 2, "green": 1}),
    SplendorCard("t1_b_0d", 1, "blue", 0, {"green": 2, "red": 1, "black": 1}),
    SplendorCard("t1_b_1a", 1, "blue", 1, {"red": 4}),
    # Green bonus cards
    SplendorCard("t1_g_0a", 1, "green", 0, {"blue": 2, "white": 1}),
    SplendorCard("t1_g_0b", 1, "green", 0, {"black": 3}),
    SplendorCard("t1_g_0c", 1, "green", 0, {"blue": 2, "black": 2}),
    SplendorCard("t1_g_0d", 1, "green", 0, {"white": 1, "red": 1, "black": 2}),
    SplendorCard("t1_g_1a", 1, "green", 1, {"black": 4}),
    # Red bonus cards
    SplendorCard("t1_r_0a", 1, "red", 0, {"green": 3}),
    SplendorCard("t1_r_0b", 1, "red", 0, {"white": 2, "blue": 1}),
    SplendorCard("t1_r_0c", 1, "red", 0, {"white": 2, "green": 2}),
    SplendorCard("t1_r_0d", 1, "red", 0, {"blue": 1, "green": 1, "white": 2}),
    SplendorCard("t1_r_1a", 1, "red", 1, {"white": 4}),
    # Black bonus cards
    SplendorCard("t1_k_0a", 1, "black", 0, {"red": 3}),
    SplendorCard("t1_k_0b", 1, "black", 0, {"blue": 2, "green": 1}),
    SplendorCard("t1_k_0c", 1, "black", 0, {"red": 2, "white": 2}),
    SplendorCard("t1_k_0d", 1, "black", 0, {"green": 1, "blue": 1, "red": 2}),
    SplendorCard("t1_k_1a", 1, "black", 1, {"green": 4}),
]

# Tier 2 — medium cost, 1–3 VP, engine payoff
_TIER2: list[SplendorCard] = [
    # White bonus
    SplendorCard("t2_w_1a", 2, "white", 1, {"green": 3, "blue": 2, "black": 2}),
    SplendorCard("t2_w_2a", 2, "white", 2, {"blue": 1, "red": 4, "black": 2}),
    SplendorCard("t2_w_3a", 2, "white", 3, {"black": 6}),
    # Blue bonus
    SplendorCard("t2_b_1a", 2, "blue", 1, {"white": 3, "red": 2, "black": 2}),
    SplendorCard("t2_b_2a", 2, "blue", 2, {"green": 1, "white": 4, "red": 2}),
    SplendorCard("t2_b_3a", 2, "blue", 3, {"white": 6}),
    # Green bonus
    SplendorCard("t2_g_1a", 2, "green", 1, {"blue": 3, "white": 2, "red": 2}),
    SplendorCard("t2_g_2a", 2, "green", 2, {"black": 1, "blue": 4, "white": 2}),
    SplendorCard("t2_g_3a", 2, "green", 3, {"blue": 6}),
    # Red bonus
    SplendorCard("t2_r_1a", 2, "red", 1, {"black": 3, "green": 2, "white": 2}),
    SplendorCard("t2_r_2a", 2, "red", 2, {"white": 1, "black": 4, "green": 2}),
    SplendorCard("t2_r_3a", 2, "red", 3, {"green": 6}),
    # Black bonus
    SplendorCard("t2_k_1a", 2, "black", 1, {"red": 3, "black": 2, "blue": 2}),
    SplendorCard("t2_k_2a", 2, "black", 2, {"red": 1, "green": 4, "blue": 2}),
    SplendorCard("t2_k_3a", 2, "black", 3, {"red": 6}),
]

# Tier 3 — high cost, 3–5 VP, late-game dominance
_TIER3: list[SplendorCard] = [
    SplendorCard("t3_w_3a", 3, "white", 3, {"black": 3, "red": 3, "green": 3}),
    SplendorCard("t3_w_4a", 3, "white", 4, {"black": 7}),
    SplendorCard("t3_w_5a", 3, "white", 5, {"black": 7, "red": 3}),
    SplendorCard("t3_b_3a", 3, "blue", 3, {"white": 3, "red": 3, "black": 3}),
    SplendorCard("t3_b_4a", 3, "blue", 4, {"white": 7}),
    SplendorCard("t3_b_5a", 3, "blue", 5, {"white": 7, "green": 3}),
    SplendorCard("t3_g_3a", 3, "green", 3, {"blue": 3, "white": 3, "red": 3}),
    SplendorCard("t3_g_4a", 3, "green", 4, {"blue": 7}),
    SplendorCard("t3_g_5a", 3, "green", 5, {"blue": 7, "black": 3}),
    SplendorCard("t3_r_3a", 3, "red", 3, {"green": 3, "blue": 3, "white": 3}),
    SplendorCard("t3_r_4a", 3, "red", 4, {"green": 7}),
    SplendorCard("t3_r_5a", 3, "red", 5, {"green": 7, "white": 3}),
    SplendorCard("t3_k_3a", 3, "black", 3, {"red": 3, "green": 3, "blue": 3}),
    SplendorCard("t3_k_4a", 3, "black", 4, {"red": 7}),
    SplendorCard("t3_k_5a", 3, "black", 5, {"red": 7, "blue": 3}),
]

# Nobles — require card bonuses (not gems), award 3 VP each
_NOBLES: list[Noble] = [
    Noble("n_wb",  3, {"white": 4, "blue": 4}),
    Noble("n_wg",  3, {"white": 4, "green": 4}),
    Noble("n_wr",  3, {"white": 3, "red": 3, "black": 3}),
    Noble("n_bg",  3, {"blue": 4, "green": 4}),
    Noble("n_br",  3, {"blue": 3, "red": 3, "green": 3}),
    Noble("n_gr",  3, {"green": 4, "red": 4}),
    Noble("n_rk",  3, {"red": 4, "black": 4}),
    Noble("n_wk",  3, {"white": 4, "black": 4}),
    Noble("n_bk",  3, {"blue": 3, "black": 3, "white": 3}),
    Noble("n_gk",  3, {"green": 3, "black": 3, "red": 3}),
]

# ---------------------------------------------------------------------------
# Index lookups (built once at module load)
# ---------------------------------------------------------------------------

ALL_CARDS: list[SplendorCard] = _TIER1 + _TIER2 + _TIER3
CARDS_BY_ID: dict[str, SplendorCard] = {c.card_id: c for c in ALL_CARDS}
NOBLES_BY_ID: dict[str, Noble] = {n.noble_id: n for n in _NOBLES}

CARDS_BY_TIER: dict[int, list[SplendorCard]] = {
    1: _TIER1,
    2: _TIER2,
    3: _TIER3,
}

N_CARDS_BY_TIER: dict[int, int] = {t: len(cards) for t, cards in CARDS_BY_TIER.items()}

# ---------------------------------------------------------------------------
# Gem budget per player count (standard rules)
# ---------------------------------------------------------------------------

GEM_SUPPLY_2P: dict[str, int] = {
    "white": 4, "blue": 4, "green": 4, "red": 4, "black": 4, "gold": 5,
}
N_NOBLES_2P = 3  # n_players + 1
N_BOARD_SLOTS = 4  # visible cards per tier
