"""Splendor game engine — deterministic 2-player simulator.

LLM is never called here. This is the hot path for RL training.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from src.games.base.engine import GameEngine
from src.games.base.game_state import Action, ActionResult
from src.games.splendor.actions import (
    SplendorAction,
    SplendorActionType,
    action_to_index,
)
from src.games.splendor.cards import (
    CARDS_BY_ID,
    CARDS_BY_TIER,
    GEM_SUPPLY_2P,
    GEM_TYPES,
    GOLD,
    N_BOARD_SLOTS,
    N_NOBLES_2P,
    NOBLES_BY_ID,
    VICTORY_POINTS_TARGET,
    Noble,
    SplendorCard,
)
from src.games.splendor.state import SplendorPlayerBoard, SplendorState

logger = logging.getLogger(__name__)

N_PLAYERS = 2


class SplendorEngine(GameEngine):
    """Implements the complete Splendor rule set for 2 players."""

    def reset(self, seed: int | None = None) -> SplendorState:
        rng = random.Random(seed)

        # Shuffle decks per tier
        decks: list[list[str]] = []
        for tier in (1, 2, 3):
            tier_cards = [c.card_id for c in CARDS_BY_TIER[tier]]
            rng.shuffle(tier_cards)
            decks.append(tier_cards)

        # Deal 4 face-up cards per tier
        board: list[list[str | None]] = []
        for tier_idx in range(3):
            row: list[str | None] = []
            for _ in range(N_BOARD_SLOTS):
                if decks[tier_idx]:
                    row.append(decks[tier_idx].pop(0))
                else:
                    row.append(None)
            board.append(row)

        # Select nobles
        all_noble_ids = list(NOBLES_BY_ID.keys())
        rng.shuffle(all_noble_ids)
        nobles_available = all_noble_ids[:N_NOBLES_2P]

        # Init player boards
        boards_data = [
            SplendorPlayerBoard(player_id=p).to_dict() for p in range(N_PLAYERS)
        ]

        return SplendorState(
            player_id=0,
            turn=0,
            phase="play",
            players=[],
            boards_data=boards_data,
            board=board,
            decks=decks,
            nobles_available=nobles_available,
            bank=dict(GEM_SUPPLY_2P),
        )

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def get_legal_actions(self, state: SplendorState) -> list[SplendorAction]:  # type: ignore[override]
        board = state.get_board(state.player_id)
        actions: list[SplendorAction] = []

        # --- TAKE GEMS ---
        available = [g for g in GEM_TYPES if state.bank.get(g, 0) > 0]
        current_gems = board.total_gems()

        # Take 3 different (only if total gems won't exceed 10 and ≥3 gem types in bank)
        if len(available) >= 3:
            from itertools import combinations
            for combo in combinations(available, 3):
                if current_gems + 3 <= 10:
                    actions.append(SplendorAction(
                        action_type=SplendorActionType.TAKE_3_GEMS,
                        gems_taken=list(combo),
                        player_id=state.player_id,
                    ))

        # Take 2 of same (bank must have ≥4 of that type, player ≤8 gems)
        for gem in GEM_TYPES:
            if state.bank.get(gem, 0) >= 4 and current_gems + 2 <= 10:
                actions.append(SplendorAction(
                    action_type=SplendorActionType.TAKE_2_GEMS,
                    gems_taken=[gem, gem],
                    player_id=state.player_id,
                ))

        # --- RESERVE CARD ---
        if board.n_reserved() < 3:
            for tier_idx in range(3):
                for slot_idx in range(N_BOARD_SLOTS):
                    card_id = state.board[tier_idx][slot_idx]
                    if card_id is not None:
                        actions.append(SplendorAction(
                            action_type=SplendorActionType.RESERVE_BOARD,
                            tier=tier_idx,
                            slot=slot_idx,
                            card_id=card_id,
                            player_id=state.player_id,
                        ))
                # Reserve top of deck
                if state.decks[tier_idx]:
                    actions.append(SplendorAction(
                        action_type=SplendorActionType.RESERVE_DECK,
                        tier=tier_idx + 1,
                        player_id=state.player_id,
                    ))

        # --- BUY CARD ---
        for tier_idx in range(3):
            for slot_idx in range(N_BOARD_SLOTS):
                card_id = state.board[tier_idx][slot_idx]
                if card_id is not None:
                    card = CARDS_BY_ID[card_id]
                    if board.can_afford(card.cost):
                        payment = board.payment_for(card.cost)
                        actions.append(SplendorAction(
                            action_type=SplendorActionType.BUY_BOARD,
                            tier=tier_idx,
                            slot=slot_idx,
                            card_id=card_id,
                            payment=payment,
                            player_id=state.player_id,
                        ))

        for rslot, card_id in enumerate(board.reserved):
            if card_id is not None:
                card = CARDS_BY_ID[card_id]
                if board.can_afford(card.cost):
                    payment = board.payment_for(card.cost)
                    actions.append(SplendorAction(
                        action_type=SplendorActionType.BUY_RESERVED,
                        reserve_slot=rslot,
                        card_id=card_id,
                        payment=payment,
                        player_id=state.player_id,
                    ))

        # Fallback: if somehow empty (e.g., all gems 0 and can't buy), allow pass
        # In practice Splendor always has at least one legal action early game
        if not actions:
            logger.warning("No legal actions found for player %d — returning pass", state.player_id)
            actions.append(SplendorAction(
                action_type=SplendorActionType.TAKE_3_GEMS,
                gems_taken=[],
                player_id=state.player_id,
            ))

        return actions

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: SplendorState, action: Action) -> ActionResult:  # type: ignore[override]
        act: SplendorAction = action  # type: ignore[assignment]
        pid = state.player_id
        board = state.get_board(pid)
        bank = dict(state.bank)
        board_grid = [list(row) for row in state.board]
        decks = [list(d) for d in state.decks]
        events: list[str] = []
        vp_gained = 0.0

        t = act.action_type

        if t == SplendorActionType.TAKE_3_GEMS:
            for gem in act.gems_taken:
                board.gems[gem] = board.gems.get(gem, 0) + 1
                bank[gem] = bank.get(gem, 0) - 1
            events.append(f"P{pid} took {act.gems_taken}")

        elif t == SplendorActionType.TAKE_2_GEMS:
            gem = act.gems_taken[0]
            board.gems[gem] = board.gems.get(gem, 0) + 2
            bank[gem] = bank.get(gem, 0) - 2
            events.append(f"P{pid} took 2×{gem}")

        elif t == SplendorActionType.RESERVE_BOARD:
            slot_card = board_grid[act.tier][act.slot]
            if slot_card is not None:
                reserve_slot = board.first_empty_reserve_slot()
                if reserve_slot is not None:
                    board.reserved[reserve_slot] = slot_card
                    board_grid[act.tier][act.slot] = None
                    # Refill from deck
                    if decks[act.tier]:
                        board_grid[act.tier][act.slot] = decks[act.tier].pop(0)
                    # Take 1 gold if available and player has room (≤9 gems → ≤10 after)
                    if bank.get(GOLD, 0) > 0 and board.total_gems() < 10:
                        board.gems[GOLD] = board.gems.get(GOLD, 0) + 1
                        bank[GOLD] -= 1
                        events.append(f"P{pid} took 1 gold")
                    events.append(f"P{pid} reserved {slot_card}")

        elif t == SplendorActionType.RESERVE_DECK:
            tier_idx = act.tier - 1
            if decks[tier_idx]:
                card_id = decks[tier_idx].pop(0)
                reserve_slot = board.first_empty_reserve_slot()
                if reserve_slot is not None:
                    board.reserved[reserve_slot] = card_id
                if bank.get(GOLD, 0) > 0 and board.total_gems() < 10:
                    board.gems[GOLD] = board.gems.get(GOLD, 0) + 1
                    bank[GOLD] -= 1
                events.append(f"P{pid} reserved top of tier {act.tier}")

        elif t == SplendorActionType.BUY_BOARD:
            card = CARDS_BY_ID[act.card_id]
            payment = board.payment_for(card.cost)
            for gem, amount in payment.items():
                board.gems[gem] = max(0, board.gems.get(gem, 0) - amount)
                bank[gem] = bank.get(gem, 0) + amount
            board.cards_owned.append(act.card_id)
            board_grid[act.tier][act.slot] = None
            if decks[act.tier]:
                board_grid[act.tier][act.slot] = decks[act.tier].pop(0)
            vp_gained = float(card.vp)
            events.append(f"P{pid} bought {act.card_id} (+{card.vp} VP)")

        elif t == SplendorActionType.BUY_RESERVED:
            card = CARDS_BY_ID[act.card_id]
            payment = board.payment_for(card.cost)
            for gem, amount in payment.items():
                board.gems[gem] = max(0, board.gems.get(gem, 0) - amount)
                bank[gem] = bank.get(gem, 0) + amount
            board.cards_owned.append(act.card_id)
            board.reserved[act.reserve_slot] = None
            vp_gained = float(card.vp)
            events.append(f"P{pid} bought reserved {act.card_id} (+{card.vp} VP)")

        # Check nobles (automatic at end of turn)
        nobles_available = list(state.nobles_available)
        bonus = board.bonus()
        for noble_id in list(nobles_available):
            noble = NOBLES_BY_ID[noble_id]
            if all(bonus.get(color, 0) >= req for color, req in noble.requirements.items()):
                board.nobles_claimed.append(noble_id)
                nobles_available.remove(noble_id)
                vp_gained += float(noble.vp)
                events.append(f"P{pid} claimed noble {noble_id} (+{noble.vp} VP)")

        # Build next state
        boards = state.get_boards()
        boards[pid] = board

        current_vp = board.vp()
        final_round = state.final_round or current_vp >= VICTORY_POINTS_TARGET

        # Advance turn: switch to opponent
        next_pid = 1 - pid
        new_turn = state.turn + 1

        # Check win condition: final_round ends when we complete the round
        # (player 0 gets the last action to tie-break)
        winner = None
        if final_round and next_pid == 0:
            # Both players have now taken equal turns; determine winner
            p0_vp = boards[0].vp()
            p1_vp = boards[1].vp()
            if p0_vp >= VICTORY_POINTS_TARGET or p1_vp >= VICTORY_POINTS_TARGET:
                if p0_vp > p1_vp:
                    winner = 0
                elif p1_vp > p0_vp:
                    winner = 1
                else:
                    # Tie-break: fewer cards owned
                    p0_cards = len(boards[0].cards_owned)
                    p1_cards = len(boards[1].cards_owned)
                    winner = 0 if p0_cards <= p1_cards else 1

        new_state = state.model_copy(update={
            "player_id": next_pid,
            "turn": new_turn,
            "boards_data": [b.to_dict() for b in boards],
            "board": board_grid,
            "decks": decks,
            "nobles_available": nobles_available,
            "bank": bank,
            "final_round": final_round,
            "winner": winner,
        })

        return ActionResult(
            success=True,
            new_state=new_state,
            events=events,
            reward=vp_gained,
        )

    # ------------------------------------------------------------------
    # Terminal / winner
    # ------------------------------------------------------------------

    def is_terminal(self, state: SplendorState) -> bool:  # type: ignore[override]
        return state.winner is not None

    def get_winner(self, state: SplendorState) -> int | None:  # type: ignore[override]
        return state.winner
