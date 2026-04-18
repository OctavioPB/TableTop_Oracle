"""Bird power execution for the Wingspan engine.

All powers are deterministic given a state. Powers requiring player choices
(D3 in PLAN.md) are either flattened to heuristic choices or stubbed as
COMPLEX with no effect in this MVP.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from src.games.wingspan.cards import BirdCard, FoodType, PowerID

if TYPE_CHECKING:
    from src.games.wingspan.state import BirdSlotState, WingspanPlayerBoard, WingspanState

logger = logging.getLogger(__name__)

# Default food type for GAIN_FOOD_SUPPLY when catalog has no specific type
_DEFAULT_SUPPLY_FOOD = FoodType.SEED.value


def execute_when_activated_power(
    state: "WingspanState",
    board: "WingspanPlayerBoard",
    slot: "BirdSlotState",
    bird: BirdCard,
    rng: random.Random,
) -> tuple["WingspanState", "WingspanPlayerBoard", list[str]]:
    """Resolve a brown 'when_activated' power.

    Returns updated (state, board, events). State is modified for global
    changes (draw_deck, bird_tray); board for player-local changes.
    """
    pid = bird.power_id
    events: list[str] = []

    if pid == PowerID.NONE.value or pid == PowerID.COMPLEX.value:
        return state, board, events

    if pid == PowerID.GAIN_FOOD_FEEDER.value:
        state, board, ev = _gain_food_from_feeder(state, board, rng)
        events.extend(ev)

    elif pid == PowerID.GAIN_FOOD_SUPPLY.value:
        food = bird.power_text.lower()
        food_type = _extract_food_type(food) or _DEFAULT_SUPPLY_FOOD
        board.food_supply[food_type] = board.food_supply.get(food_type, 0) + 1
        events.append(f"Gained 1 {food_type} from supply")

    elif pid == PowerID.LAY_EGG_SELF.value:
        from src.games.wingspan.cards import load_bird_catalog
        bird_catalog = _get_catalog(state)
        card = bird_catalog.get(slot.bird_name)
        if card and slot.eggs < card.egg_limit:
            slot.eggs += 1
            events.append(f"Laid 1 egg on {slot.bird_name}")

    elif pid == PowerID.LAY_EGG_ANY.value:
        # Place egg on bird with most room (heuristic: closest to egg limit)
        board, ev = _lay_egg_best_slot(board, _get_catalog(state), 1)
        events.extend(ev)

    elif pid == PowerID.DRAW_CARD.value:
        state, board, ev = _draw_from_deck(state, board, 1)
        events.extend(ev)

    elif pid == PowerID.TUCK_CARD_DECK.value:
        if state.draw_deck:
            card_name = state.draw_deck[0]
            new_deck = list(state.draw_deck[1:])
            state = state.model_copy(update={"draw_deck": new_deck})
            slot.tucked_cards += 1
            events.append(f"Tucked {card_name} under {slot.bird_name}")

    elif pid == PowerID.CACHE_FOOD.value:
        # Cache food from player supply (any food available)
        food_type = _take_any_food(board)
        if food_type:
            slot.cached_food[food_type] = slot.cached_food.get(food_type, 0) + 1
            events.append(f"Cached 1 {food_type} on {slot.bird_name}")

    elif pid == PowerID.PREDATOR.value:
        state, board, slot, ev = _resolve_predator(state, board, slot, bird, _get_catalog(state))
        events.extend(ev)

    return state, board, events


def execute_when_played_power(
    state: "WingspanState",
    board: "WingspanPlayerBoard",
    bird: BirdCard,
    rng: random.Random,
) -> tuple["WingspanState", "WingspanPlayerBoard", list[str]]:
    """Resolve a 'when_played' power immediately after a bird is placed."""
    pid = bird.power_id
    events: list[str] = []

    if pid == PowerID.WHEN_PLAYED_DRAW.value:
        n = max(1, bird.power_param)
        state, board, ev = _draw_from_deck(state, board, n)
        events.extend(ev)

    elif pid == PowerID.WHEN_PLAYED_GAIN_FOOD.value:
        n = max(1, bird.power_param)
        state, board, ev = _gain_food_from_feeder(state, board, rng, n=n)
        events.extend(ev)

    elif pid == PowerID.WHEN_PLAYED_LAY_EGG.value:
        n = max(1, bird.power_param)
        board, ev = _lay_egg_best_slot(board, _get_catalog(state), n)
        events.extend(ev)

    return state, board, events


def execute_end_of_round_power(
    state: "WingspanState",
    board: "WingspanPlayerBoard",
    slot: "BirdSlotState",
    bird: BirdCard,
) -> tuple["WingspanState", "WingspanPlayerBoard", list[str]]:
    """Resolve a teal 'end_of_round' power."""
    pid = bird.power_id
    events: list[str] = []

    if pid == PowerID.END_ROUND_LAY_EGG.value:
        board, ev = _lay_egg_best_slot(board, _get_catalog(state), 1)
        events.extend(ev)

    return state, board, events


def execute_once_between_turns(
    state: "WingspanState",
    passive_board: "WingspanPlayerBoard",
    slot: "BirdSlotState",
    bird: BirdCard,
    trigger_action: str,
    rng: random.Random,
) -> tuple["WingspanState", "WingspanPlayerBoard", list[str]]:
    """Resolve a pink 'once_between_turns' power triggered by another player.

    Args:
        trigger_action: The action type the active player just took.
    """
    pid = bird.power_id
    events: list[str] = []

    if pid == PowerID.ONCE_BTW_LAY_EGG.value and trigger_action in ("lay_eggs", "play_bird"):
        passive_board, ev = _lay_egg_best_slot(passive_board, _get_catalog(state), 1)
        events.extend(ev)

    elif pid == PowerID.ONCE_BTW_GAIN_FOOD.value and trigger_action == "gain_food":
        state, passive_board, ev = _gain_food_from_feeder(state, passive_board, rng)
        events.extend(ev)

    elif pid == PowerID.ONCE_BTW_DRAW.value and trigger_action == "draw_cards":
        state, passive_board, ev = _draw_from_deck(state, passive_board, 1)
        events.extend(ev)

    return state, passive_board, events


# ---------------------------------------------------------------------------
# Shared helper primitives
# ---------------------------------------------------------------------------


def _get_catalog(state: "WingspanState") -> dict[str, BirdCard]:
    """Lazy-load the bird catalog from the engine's global registry."""
    from src.games.wingspan.engine import _GLOBAL_CATALOG
    return _GLOBAL_CATALOG


def _gain_food_from_feeder(
    state: "WingspanState",
    board: "WingspanPlayerBoard",
    rng: random.Random,
    n: int = 1,
) -> tuple["WingspanState", "WingspanPlayerBoard", list[str]]:
    """Take up to n food from the feeder; auto-reroll if empty or all-same."""
    from src.games.wingspan.engine import _roll_feeder

    events: list[str] = []
    gained = 0
    feeder = list(state.bird_feeder)

    for _ in range(n):
        if not feeder:
            feeder = _roll_feeder(rng)
            events.append("Bird feeder rerolled (was empty)")

        _maybe_reroll(feeder, rng, events)
        if feeder:
            food = feeder.pop(0)
            board.food_supply[food] = board.food_supply.get(food, 0) + 1
            gained += 1
            events.append(f"Gained 1 {food} from feeder (power)")

    state = state.model_copy(update={"bird_feeder": feeder})
    return state, board, events


def _draw_from_deck(
    state: "WingspanState",
    board: "WingspanPlayerBoard",
    n: int,
) -> tuple["WingspanState", "WingspanPlayerBoard", list[str]]:
    events: list[str] = []
    deck = list(state.draw_deck)
    drawn = 0
    for _ in range(n):
        if deck:
            card = deck.pop(0)
            board.hand.append(card)
            drawn += 1
    if drawn:
        events.append(f"Drew {drawn} card(s) from deck (power)")
    state = state.model_copy(update={"draw_deck": deck})
    return state, board, events


def _lay_egg_best_slot(
    board: "WingspanPlayerBoard",
    catalog: dict[str, BirdCard],
    n: int,
) -> tuple["WingspanPlayerBoard", list[str]]:
    """Lay n eggs, distributing across birds with the most room first."""
    events: list[str] = []
    placed = 0
    for _ in range(n):
        best_slot = _find_best_egg_slot(board, catalog)
        if best_slot is None:
            break
        hab, idx = best_slot
        slot = board.get_habitat(hab)[idx]
        if slot:
            slot.eggs += 1
            placed += 1
    if placed:
        events.append(f"Laid {placed} egg(s) via power")
    return board, events


def _find_best_egg_slot(
    board: "WingspanPlayerBoard",
    catalog: dict[str, BirdCard],
) -> tuple[str, int] | None:
    """Find the bird slot with the most room for another egg."""
    best: tuple[str, int] | None = None
    best_room = 0
    for hab in ("forest", "grassland", "wetland"):
        for i, slot in enumerate(board.get_habitat(hab)):
            if slot is None:
                continue
            card = catalog.get(slot.bird_name)
            if not card:
                continue
            room = card.egg_limit - slot.eggs
            if room > best_room:
                best_room = room
                best = (hab, i)
    return best


def _resolve_predator(
    state: "WingspanState",
    board: "WingspanPlayerBoard",
    slot: "BirdSlotState",
    bird: BirdCard,
    catalog: dict[str, BirdCard],
) -> tuple["WingspanState", "WingspanPlayerBoard", "BirdSlotState", list[str]]:
    events: list[str] = []
    threshold = bird.power_param or 75  # default wingspan threshold
    deck = list(state.draw_deck)
    if not deck:
        return state, board, slot, events

    prey_name = deck.pop(0)
    prey_card = catalog.get(prey_name)

    if prey_card and prey_card.wingspan_cm <= threshold:
        slot.tucked_cards += 1
        events.append(f"Predator hunt succeeded: tucked {prey_name}")
    else:
        discard = list(state.discard_pile) + [prey_name]
        state = state.model_copy(update={"discard_pile": discard})
        events.append(f"Predator hunt failed: {prey_name} discarded")

    state = state.model_copy(update={"draw_deck": deck})
    return state, board, slot, events


def _take_any_food(board: "WingspanPlayerBoard") -> str | None:
    """Remove one food from player supply; return type taken, or None if empty."""
    for food_type, count in list(board.food_supply.items()):
        if count > 0:
            board.food_supply[food_type] -= 1
            if board.food_supply[food_type] == 0:
                del board.food_supply[food_type]
            return food_type
    return None


def _extract_food_type(text: str) -> str | None:
    from src.games.wingspan.cards import FoodType
    for ft in FoodType:
        if ft.value in text:
            return ft.value
    return None


def _maybe_reroll(
    feeder: list[str],
    rng: random.Random,
    events: list[str],
) -> None:
    """Reroll all feeder dice in-place if they all show the same face."""
    from src.games.wingspan.engine import _roll_feeder

    if len(feeder) >= 2 and len(set(feeder)) == 1:
        new = _roll_feeder(rng, len(feeder))
        feeder.clear()
        feeder.extend(new)
        events.append("Bird feeder rerolled (all same face)")
