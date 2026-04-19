"""Legal move validator for 7 Wonders Duel — Sprint 7.

A card is playable (BUILD_CARD) if:
  - It is accessible in the current pyramid.
  - The player can afford it: player_coins + produced_resources ≥ cost
    (resources can be bought from the bank at 2+opponent_count coins each,
    but the simplified MVP uses fixed-cost trading for fast training).

A card is always discardable (DISCARD_CARD) if it is accessible.

A wonder slot is buildable (BUILD_WONDER) if:
  - The wonder slot is not yet built.
  - The player can afford the wonder's resource cost.
  - There are still unbuilt wonders (max 7 wonders total per game).
"""

from __future__ import annotations

import logging
from typing import Any

from src.games.seven_wonders_duel.actions import SWDAction, SWDActionType
from src.games.seven_wonders_duel.cards import ALL_RESOURCES, Card
from src.games.seven_wonders_duel.state import SWDState

logger = logging.getLogger(__name__)

_MAX_TOTAL_WONDERS = 7   # official rule: at most 7 wonders total across both players


class SWDLegalMoveValidator:
    """Computes the set of legal actions for the current player.

    Args:
        card_catalog: Dict mapping card_name → Card object (all ages combined).
        wonder_catalog: Dict mapping wonder_name → WonderCard object.
    """

    def __init__(
        self,
        card_catalog: dict[str, Card],
        wonder_catalog: dict[str, Any],
    ) -> None:
        self._cards = card_catalog
        self._wonders = wonder_catalog

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_legal_actions(self, state: SWDState, player_id: int) -> list[SWDAction]:
        """Return all legal SWDAction objects for player_id in state.

        Never returns an empty list in a non-terminal state (the player can
        always discard any accessible card).
        """
        board = state.get_board(player_id)
        accessible = state.accessible_cards(state.age_deck)

        if not accessible:
            return []

        actions: list[SWDAction] = []

        for card_name in accessible:
            # Always legal to discard
            actions.append(SWDAction(
                action_type=SWDActionType.DISCARD_CARD.value,
                player_id=player_id,
                card_name=card_name,
            ))

            # Legal to build if affordable
            card = self._cards.get(card_name)
            if card and self._can_afford(board, card, state, player_id):
                actions.append(SWDAction(
                    action_type=SWDActionType.BUILD_CARD.value,
                    player_id=player_id,
                    card_name=card_name,
                ))

            # Legal to use card to build each unbuilt wonder slot
            total_built = sum(
                sum(1 for b in state.get_board(pid).built_wonders if b)
                for pid in range(2)
            )
            if total_built < _MAX_TOTAL_WONDERS:
                for slot_idx, wonder_name in enumerate(board.wonders):
                    if board.built_wonders[slot_idx]:
                        continue
                    wonder = self._wonders.get(wonder_name)
                    if wonder and self._can_afford_wonder(board, wonder, state, player_id):
                        actions.append(SWDAction(
                            action_type=SWDActionType.BUILD_WONDER.value,
                            player_id=player_id,
                            card_name=card_name,
                            wonder_slot=slot_idx,
                        ))

        return actions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _can_afford(
        self,
        board: Any,
        card: Card,
        state: SWDState,
        player_id: int,
    ) -> bool:
        """Return True if player_id can afford to build card."""
        # Direct coin cost
        if board.coins < card.cost_coins:
            return False

        # Resource cost — player resources + trading
        shortfall = self._resource_shortfall(board, card.cost_resources, state, player_id)
        return shortfall == 0

    def _can_afford_wonder(
        self,
        board: Any,
        wonder: Any,
        state: SWDState,
        player_id: int,
    ) -> bool:
        """Return True if player_id can afford to build wonder."""
        shortfall = self._resource_shortfall(board, wonder.cost_resources, state, player_id)
        return shortfall == 0

    def _resource_shortfall(
        self,
        board: Any,
        cost: dict[str, int],
        state: SWDState,
        player_id: int,
    ) -> int:
        """Return total coin cost needed to trade for missing resources.

        Trading rule (simplified): each missing resource unit costs
        2 + (opponent's count of that resource) coins.  Returns total coin
        cost for all missing units; if the player cannot afford trading, returns
        a very large number.
        """
        opponent_board = state.get_board(1 - player_id)
        total_coin_cost = 0

        for resource, qty_needed in cost.items():
            if resource not in ALL_RESOURCES:
                continue
            have = board.resources.get(resource, 0)
            # Check for "any_raw" or "any_manufactured" wildcards
            # (simplified: wildcards not counted here — handled in apply)
            missing = max(0, qty_needed - have)
            if missing == 0:
                continue

            if board.discounts.get(resource, False):
                unit_cost = 1
            else:
                opp_count = opponent_board.resources.get(resource, 0)
                unit_cost = 2 + opp_count

            total_coin_cost += missing * unit_cost

        # Check if player has enough coins for trading
        if board.coins < total_coin_cost:
            return total_coin_cost  # can't afford — return non-zero shortfall

        return 0
