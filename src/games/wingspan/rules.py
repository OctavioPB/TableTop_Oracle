"""Wingspan legal move validator — deterministic Python, zero LLM at runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.games.wingspan.actions import WingspanAction, WingspanActionType
from src.games.wingspan.cards import BirdCard, FoodType

if TYPE_CHECKING:
    from src.games.wingspan.state import WingspanPlayerBoard, WingspanState

logger = logging.getLogger(__name__)


class LegalMoveValidator:
    """Computes all legal actions for a given (state, player) pair.

    No LLM calls here — this is the hot path during RL training.
    """

    def __init__(self, catalog: dict[str, BirdCard]) -> None:
        self._catalog = catalog

    # ------------------------------------------------------------------
    # Top-level API
    # ------------------------------------------------------------------

    def get_legal_actions(
        self,
        state: "WingspanState",
        player_id: int,
    ) -> list[WingspanAction]:
        """Return all legal actions for player_id in state.

        Guarantees non-empty list for non-terminal states.
        """
        board = state.get_board(player_id)
        actions: list[WingspanAction] = []

        actions.extend(self.get_legal_gain_food_actions(state, board))
        actions.extend(self.get_legal_lay_eggs_actions(state, board))
        actions.extend(self.get_legal_draw_cards_actions(state, board))
        actions.extend(self.get_legal_play_bird_actions(state, board))

        # Safety guarantee: if somehow all else fails, gain food is forced available
        if not actions:
            logger.warning("No legal actions found for player %d — forcing GAIN_FOOD", player_id)
            feeder = state.bird_feeder or [FoodType.SEED.value]
            actions.append(
                WingspanAction(
                    action_type=WingspanActionType.GAIN_FOOD.value,
                    player_id=player_id,
                    food_choice=feeder[0],
                )
            )

        return actions

    # ------------------------------------------------------------------
    # Per-action legal generators
    # ------------------------------------------------------------------

    def get_legal_gain_food_actions(
        self,
        state: "WingspanState",
        board: "WingspanPlayerBoard",
    ) -> list[WingspanAction]:
        """One action per unique food type available in the feeder.

        If the feeder is empty, it will be auto-rerolled by the engine;
        we still report GAIN_FOOD as legal using a wildcard choice.
        """
        feeder = state.bird_feeder
        if not feeder:
            # Feeder empty → engine will reroll; report as legal with any food
            return [
                WingspanAction(
                    action_type=WingspanActionType.GAIN_FOOD.value,
                    player_id=board.player_id,
                    food_choice=FoodType.SEED.value,
                )
            ]

        seen: set[str] = set()
        actions: list[WingspanAction] = []
        for food in feeder:
            if food not in seen:
                seen.add(food)
                actions.append(
                    WingspanAction(
                        action_type=WingspanActionType.GAIN_FOOD.value,
                        player_id=board.player_id,
                        food_choice=food,
                    )
                )
        return actions

    def get_legal_lay_eggs_actions(
        self,
        state: "WingspanState",
        board: "WingspanPlayerBoard",
    ) -> list[WingspanAction]:
        """LAY_EGGS is legal iff the player has at least one bird with room for an egg."""
        for hab in ("forest", "grassland", "wetland"):
            for _, slot in board.birds_in_habitat(hab):
                card = self._catalog.get(slot.bird_name)
                if card and slot.eggs < card.egg_limit:
                    return [
                        WingspanAction(
                            action_type=WingspanActionType.LAY_EGGS.value,
                            player_id=board.player_id,
                        )
                    ]
        return []

    def get_legal_draw_cards_actions(
        self,
        state: "WingspanState",
        board: "WingspanPlayerBoard",
    ) -> list[WingspanAction]:
        """One action per available source (tray_0/1/2 + deck if non-empty)."""
        actions: list[WingspanAction] = []

        for i, card_name in enumerate(state.bird_tray):
            actions.append(
                WingspanAction(
                    action_type=WingspanActionType.DRAW_CARDS.value,
                    player_id=board.player_id,
                    draw_source=f"tray_{i}",
                )
            )

        if state.draw_deck:
            actions.append(
                WingspanAction(
                    action_type=WingspanActionType.DRAW_CARDS.value,
                    player_id=board.player_id,
                    draw_source="deck",
                )
            )

        return actions

    def get_legal_play_bird_actions(
        self,
        state: "WingspanState",
        board: "WingspanPlayerBoard",
    ) -> list[WingspanAction]:
        """One action per (card in hand × valid habitat) pair that can be afforded."""
        actions: list[WingspanAction] = []

        for card_name in board.hand:
            card = self._catalog.get(card_name)
            if card is None:
                continue

            for hab in card.habitats:
                # Check habitat has a free slot
                if board.first_empty_slot(hab) is None:
                    continue

                # Check food cost is payable
                food_pay = self._compute_food_payment(board.food_supply, card.food_cost)
                if food_pay is None:
                    continue

                # Check egg cost
                egg_cost = board.egg_cost_for_habitat(hab)
                if board.total_eggs() < egg_cost:
                    continue

                actions.append(
                    WingspanAction(
                        action_type=WingspanActionType.PLAY_BIRD.value,
                        player_id=board.player_id,
                        card_name=card_name,
                        target_habitat=hab,
                        food_payment=food_pay,
                        egg_payment=egg_cost,
                    )
                )

        return actions

    def validate_action(
        self,
        state: "WingspanState",
        action: WingspanAction,
    ) -> tuple[bool, str]:
        """Return (is_legal, reason) for the proposed action.

        Used for safety checks; the hot path uses get_legal_actions().
        """
        board = state.get_board(action.player_id)
        legal = self.get_legal_actions(state, action.player_id)
        for la in legal:
            if (
                la.action_type == action.action_type
                and la.food_choice == action.food_choice
                and la.draw_source == action.draw_source
                and la.card_name == action.card_name
                and la.target_habitat == action.target_habitat
            ):
                return True, "Legal"
        return False, f"Action {action} not in legal action set"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_food_payment(
        self,
        supply: dict[str, int],
        cost: dict[str, int],
    ) -> dict[str, int] | None:
        """Compute optimal payment for a food cost from the given supply.

        Returns a payment dict or None if the cost cannot be satisfied.
        Wild food costs are paid last with any remaining food.
        """
        remaining = dict(supply)
        payment: dict[str, int] = {}

        # Pay specific food types first
        for food, amount in cost.items():
            if food == FoodType.WILD.value:
                continue
            available = remaining.get(food, 0)
            if available < amount:
                return None
            payment[food] = payment.get(food, 0) + amount
            remaining[food] = available - amount

        # Pay wild costs with any remaining food
        wild_needed = cost.get(FoodType.WILD.value, 0)
        for food, count in list(remaining.items()):
            if wild_needed <= 0:
                break
            paid = min(count, wild_needed)
            if paid > 0:
                payment[food] = payment.get(food, 0) + paid
                remaining[food] -= paid
                wild_needed -= paid

        if wild_needed > 0:
            return None

        return payment
