"""7 Wonders Duel game engine — Sprint 7.

Deterministic 2-player simulator following the same GameEngine ABC as Wingspan.
State is immutable in the public API; step() returns a new SWDState via model_copy.

Simplifications relative to the physical game (documented as technical debt):
  TD1 — Trading: per-resource dynamic pricing is implemented; complex chain discounts
        (e.g. caravansery / forum wildcards) are treated as one free token of any
        resource in the respective category.
  TD2 — Guild cards: scored at end of Age 3 based on board state; not all guilds
        have full VP formulas (placeholder: each guild = 3 VP).
  TD3 — Progress token interactions (e.g. Theology extra-turn stacking) use the
        simple interpretation.
  TD4 — Wonders with special effects (Mausoleum, etc.) apply a simplified version.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

from src.games.base.engine import GameEngine
from src.games.base.game_state import ActionResult
from src.games.seven_wonders_duel.actions import SWDAction, SWDActionType
from src.games.seven_wonders_duel.cards import (
    ALL_SCIENCE_SYMBOLS,
    Card,
    WonderCard,
    load_card_catalog,
)
from src.games.seven_wonders_duel.rules import SWDLegalMoveValidator
from src.games.seven_wonders_duel.state import (
    MILITARY_CENTER,
    MILITARY_TRACK_MAX,
    N_PROGRESS_TOKENS_AVAILABLE,
    AGE_FACE_DOWN,
    SWDPlayerBoard,
    SWDState,
)

logger = logging.getLogger(__name__)

_DEFAULT_CATALOG_PATH = Path(
    os.environ.get(
        "CARD_CATALOG_DIR",
        str(Path(__file__).parent.parent.parent.parent / "data" / "card_catalogs"),
    )
) / "seven_wonders_duel_cards.json"


class SWDEngine(GameEngine):
    """2-player 7 Wonders Duel simulator.

    Args:
        catalog_path: Path to seven_wonders_duel_cards.json.
        seed: RNG seed; None for non-deterministic.
    """

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        path = Path(catalog_path) if catalog_path else _DEFAULT_CATALOG_PATH
        age1, age2, age3, wonders, tokens = load_card_catalog(path)

        # Store as lists (age order) and merged dicts
        self._age_cards: dict[int, list[Card]] = {1: age1, 2: age2, 3: age3}
        self._card_catalog: dict[str, Card] = {}
        for cards_in_age in [age1, age2, age3]:
            for c in cards_in_age:
                self._card_catalog[c.name] = c

        self._wonder_catalog: dict[str, WonderCard] = {w.name: w for w in wonders}
        self._all_wonders: list[str] = [w.name for w in wonders]
        self._progress_tokens: list[Any] = tokens
        self._all_token_names: list[str] = [t.name for t in tokens]

        self._seed = seed
        self._rng = random.Random(seed)
        self._validator = SWDLegalMoveValidator(self._card_catalog, self._wonder_catalog)

    # ------------------------------------------------------------------
    # GameEngine interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> SWDState:
        """Return the initial state for a new 2-player 7WD game."""
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self._seed)

        # Deal 4 wonders to each player from the pool of 12
        shuffled_wonders = list(self._all_wonders)
        self._rng.shuffle(shuffled_wonders)
        p0_wonders = shuffled_wonders[:4]
        p1_wonders = shuffled_wonders[4:8]

        # Prepare progress tokens (5 face-up, rest removed)
        shuffled_tokens = list(self._all_token_names)
        self._rng.shuffle(shuffled_tokens)
        tokens_available = shuffled_tokens[:N_PROGRESS_TOKENS_AVAILABLE]

        # Prepare Age 1 deck
        age1_deck, face_down_1 = self._shuffle_age_deck(1)

        boards = [
            SWDPlayerBoard(player_id=0, coins=7, wonders=p0_wonders),
            SWDPlayerBoard(player_id=1, coins=7, wonders=p1_wonders),
        ]

        return SWDState(
            player_id=0,
            age=1,
            turn=0,
            phase="main",
            age_deck=age1_deck,
            taken_cards=set(),
            face_down_cards=face_down_1,
            boards_data=[b.to_dict() for b in boards],
            progress_tokens_available=tokens_available,
            progress_tokens_taken=[],
            military_pawn=MILITARY_CENTER,
            discard_pile=[],
            winner=None,
        )

    def step(self, state: SWDState, action: SWDAction) -> ActionResult:
        """Apply action to state and return ActionResult with new_state."""
        if not isinstance(action, SWDAction):
            raise TypeError(f"Expected SWDAction, got {type(action)}")

        events: list[str] = []
        reward = 0.0

        atype = action.action_type

        if atype == SWDActionType.BUILD_CARD.value:
            state, reward, events = self._apply_build_card(state, action, events)
        elif atype == SWDActionType.DISCARD_CARD.value:
            state, events = self._apply_discard_card(state, action, events)
        elif atype == SWDActionType.BUILD_WONDER.value:
            state, reward, events = self._apply_build_wonder(state, action, events)
        else:
            raise ValueError(f"Unknown action type: {atype}")

        # Remove card from pyramid and uncover face-down cards
        state = self._remove_card_from_pyramid(state, action.card_name)

        # Check instant win conditions
        mil_winner = state.military_winner()
        if mil_winner is not None:
            state = state.model_copy(update={"winner": mil_winner, "phase": "finished"})
            reward = 1.0 if mil_winner == state.player_id else -1.0

        sci_winner = state.science_winner(self._card_catalog)
        if sci_winner is not None and state.phase != "finished":
            state = state.model_copy(update={"winner": sci_winner, "phase": "finished"})
            reward = 1.0 if sci_winner == state.player_id else -1.0

        # Advance age if pyramid exhausted
        if state.phase != "finished" and not state.accessible_cards(state.age_deck):
            if state.age < 3:
                state = self._advance_age(state)
            else:
                state = state.model_copy(update={"phase": "finished"})
                winner = self._compute_vp_winner(state)
                state = state.model_copy(update={"winner": winner})
                reward = 1.0 if winner == state.player_id else -1.0

        # Alternate turns (no extra-turn handling in MVP)
        if state.phase != "finished":
            next_player = 1 - state.player_id
            state = state.model_copy(update={
                "player_id": next_player,
                "turn": state.turn + 1,
            })

        return ActionResult(
            success=True,
            new_state=state,
            events=events,
            reward=reward,
        )

    def get_legal_actions(self, state: SWDState) -> list[SWDAction]:
        return self._validator.get_legal_actions(state, state.player_id)

    def is_terminal(self, state: SWDState) -> bool:
        return state.phase == "finished"

    def get_winner(self, state: SWDState) -> int | None:
        if not self.is_terminal(state):
            return None
        return state.winner

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _apply_build_card(
        self,
        state: SWDState,
        action: SWDAction,
        events: list[str],
    ) -> tuple[SWDState, float, list[str]]:
        pid = state.player_id
        board = state.get_board(pid)
        opp_board = state.get_board(1 - pid)
        card = self._card_catalog.get(action.card_name)
        reward = 0.0

        if card is None:
            events.append(f"Unknown card: {action.card_name}")
            return state, reward, events

        # Pay cost
        board, state = self._pay_cost(board, opp_board, card.cost_resources, card.cost_coins, state, pid)

        # Apply card effect
        board, state, reward, events = self._apply_card_effect(board, card, state, pid, events)

        # Record built card
        board = SWDPlayerBoard.from_dict({**board.to_dict(), "built_cards": board.built_cards + [card.name]})
        state = state.with_board(pid, board)

        events.append(f"P{pid} built {card.name}")
        return state, reward, events

    def _apply_discard_card(
        self,
        state: SWDState,
        action: SWDAction,
        events: list[str],
    ) -> tuple[SWDState, list[str]]:
        pid = state.player_id
        board = state.get_board(pid)

        # Coins = 2 + number of commercial (yellow) cards player has built
        commercial_count = sum(
            1 for cn in board.built_cards
            if self._card_catalog.get(cn) and self._card_catalog[cn].card_type == "commercial"
        )
        coins_gained = 2 + commercial_count
        new_coins = board.coins + coins_gained

        board = SWDPlayerBoard.from_dict({**board.to_dict(), "coins": new_coins})
        state = state.with_board(pid, board)

        # Add to discard pile
        new_discard = list(state.discard_pile) + [action.card_name]
        state = state.model_copy(update={"discard_pile": new_discard})

        events.append(f"P{pid} discarded {action.card_name} for {coins_gained} coins")
        return state, events

    def _apply_build_wonder(
        self,
        state: SWDState,
        action: SWDAction,
        events: list[str],
    ) -> tuple[SWDState, float, list[str]]:
        pid = state.player_id
        board = state.get_board(pid)
        opp_board = state.get_board(1 - pid)
        slot = action.wonder_slot
        wonder_name = board.wonders[slot]
        wonder = self._wonder_catalog.get(wonder_name)
        reward = 0.0

        if wonder is None:
            return state, reward, events

        # Pay wonder cost
        board, state = self._pay_cost(board, opp_board, wonder.cost_resources, wonder.cost_coins, state, pid)

        # Mark wonder slot as built
        new_built = list(board.built_wonders)
        new_built[slot] = True

        # Apply wonder effect
        vp_gain = int(wonder.effect.get("vp", 0))
        coins_gain = int(wonder.effect.get("coins", 0))
        shields_gain = int(wonder.effect.get("shields", 0))

        board = SWDPlayerBoard.from_dict({
            **board.to_dict(),
            "built_wonders": new_built,
            "vp_from_wonders": board.vp_from_wonders + vp_gain,
            "coins": board.coins + coins_gain,
            "shields": board.shields + shields_gain,
        })

        # Update military pawn if shields gained
        if shields_gain > 0:
            state, events = self._advance_military(state, pid, shields_gain, events)

        state = state.with_board(pid, board)
        reward = vp_gain * 0.05
        events.append(f"P{pid} built wonder {wonder_name} (slot {slot})")
        return state, reward, events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pay_cost(
        self,
        board: SWDPlayerBoard,
        opp_board: SWDPlayerBoard,
        cost_resources: dict[str, int],
        cost_coins: int,
        state: SWDState,
        pid: int,
    ) -> tuple[SWDPlayerBoard, SWDState]:
        """Deduct coins for resource trading and direct coin costs."""
        total_coins_spent = cost_coins

        for resource, qty_needed in cost_resources.items():
            have = board.resources.get(resource, 0)
            missing = max(0, qty_needed - have)
            if missing > 0:
                if board.discounts.get(resource, False):
                    unit_cost = 1
                else:
                    unit_cost = 2 + opp_board.resources.get(resource, 0)
                total_coins_spent += missing * unit_cost

        new_board_data = {**board.to_dict(), "coins": board.coins - total_coins_spent}
        return SWDPlayerBoard.from_dict(new_board_data), state

    def _apply_card_effect(
        self,
        board: SWDPlayerBoard,
        card: Card,
        state: SWDState,
        pid: int,
        events: list[str],
    ) -> tuple[SWDPlayerBoard, SWDState, float, list[str]]:
        """Apply the card's effect to the board and state."""
        reward = 0.0
        data = board.to_dict()

        # Resource production
        for res in ["wood", "stone", "clay", "glass", "papyrus"]:
            if card.effect.get(res, 0):
                data["resources"] = dict(data.get("resources", {}))
                data["resources"][res] = data["resources"].get(res, 0) + card.effect[res]

        # Immediate coins
        if card.effect.get("coins", 0):
            data["coins"] = data.get("coins", 0) + card.effect["coins"]

        # Military shields
        if card.effect.get("shields", 0):
            new_board = SWDPlayerBoard.from_dict(data)
            state, events = self._advance_military(state, pid, card.effect["shields"], events)
            data = new_board.to_dict()
            data["shields"] = data.get("shields", 0) + card.effect["shields"]

        # VP
        if card.effect.get("vp", 0):
            data["vp_from_cards"] = data.get("vp_from_cards", 0) + card.effect["vp"]
            reward += card.effect["vp"] * 0.05

        # Science symbols
        for sym in ALL_SCIENCE_SYMBOLS:
            if card.effect.get(sym, 0):
                sci = dict(data.get("science_symbols", {}))
                sci[sym] = sci.get(sym, 0) + card.effect[sym]
                data["science_symbols"] = sci
                # Check for progress token (pair of same symbol)
                if sci[sym] == 2:
                    state, events = self._award_progress_token(state, pid, events)

        board = SWDPlayerBoard.from_dict(data)
        return board, state, reward, events

    def _advance_military(
        self,
        state: SWDState,
        pid: int,
        shields: int,
        events: list[str],
    ) -> tuple[SWDState, list[str]]:
        """Move the conflict pawn toward the opponent's capital."""
        if pid == 0:
            new_pawn = max(0, state.military_pawn - shields)
        else:
            new_pawn = min(MILITARY_TRACK_MAX - 1, state.military_pawn + shields)

        state = state.model_copy(update={"military_pawn": new_pawn})
        events.append(f"Military pawn → {new_pawn}")
        return state, events

    def _award_progress_token(
        self,
        state: SWDState,
        pid: int,
        events: list[str],
    ) -> tuple[SWDState, list[str]]:
        """Award a random available progress token to player pid."""
        available = list(state.progress_tokens_available)
        if not available:
            return state, events

        token_name = self._rng.choice(available)
        new_available = [t for t in available if t != token_name]
        new_taken = list(state.progress_tokens_taken) + [token_name]

        board = state.get_board(pid)
        board_data = board.to_dict()
        board_data["progress_tokens"] = board_data.get("progress_tokens", []) + [token_name]

        state = state.model_copy(update={
            "progress_tokens_available": new_available,
            "progress_tokens_taken": new_taken,
        })
        state = state.with_board(pid, SWDPlayerBoard.from_dict(board_data))
        events.append(f"P{pid} gained progress token: {token_name}")
        return state, events

    def _remove_card_from_pyramid(self, state: SWDState, card_name: str) -> SWDState:
        """Mark card as taken and uncover any newly exposed face-down cards."""
        new_taken = set(state.taken_cards) | {card_name}

        # Uncover face-down cards: a face-down card is uncovered when it becomes accessible
        new_face_down = set(state.face_down_cards)
        # We do this lazily: just remove from face_down when accessible() finds them
        # (the accessor handles face-down transparency)
        new_face_down.discard(card_name)

        return state.model_copy(update={"taken_cards": new_taken, "face_down_cards": new_face_down})

    def _advance_age(self, state: SWDState) -> SWDState:
        """Move to the next age with a fresh shuffled deck."""
        new_age = state.age + 1
        new_deck, new_face_down = self._shuffle_age_deck(new_age)
        return state.model_copy(update={
            "age": new_age,
            "age_deck": new_deck,
            "taken_cards": set(),
            "face_down_cards": new_face_down,
        })

    def _shuffle_age_deck(self, age: int) -> tuple[list[str], set[str]]:
        """Return (shuffled_card_name_list, face_down_set) for the given age."""
        cards = list(self._age_cards.get(age, []))
        if not cards:
            return [], set()

        names = [c.name for c in cards]
        self._rng.shuffle(names)

        face_down_positions = AGE_FACE_DOWN.get(age, set())
        face_down_names = {
            names[i] for i in face_down_positions if i < len(names)
        }
        return names, face_down_names

    def _compute_final_score(self, state: SWDState, pid: int) -> int:
        """Compute final VP for player pid.

        Components:
          - VP from built cards (civilian, guild, commercial effects)
          - VP from built wonders
          - 1 VP per 3 coins
          - Military bonus: +2/5/10 VP for pawn at opponent's position zones
          - Guild VP (simplified: each guild card = 3 VP)
          - Progress token VP
        """
        board = state.get_board(pid)
        total = board.vp_from_cards + board.vp_from_wonders

        # Coins → VP
        total += board.coins // 3

        # Military bonus
        if pid == 0:
            distance = 4 - state.military_pawn  # positive = P0 winning
        else:
            distance = state.military_pawn - 4

        if distance >= 6:
            total += 10
        elif distance >= 3:
            total += 5
        elif distance >= 1:
            total += 2

        # Guild VP (simplified)
        for cn in board.built_cards:
            card = self._card_catalog.get(cn)
            if card and card.card_type == "guild":
                total += 3

        # Progress token VP
        for token_name in board.progress_tokens:
            token = next((t for t in self._progress_tokens if t.name == token_name), None)
            if token:
                total += int(token.effect.get("vp", 0))

        return max(0, total)

    def _compute_vp_winner(self, state: SWDState) -> int:
        """Return player_id with higher final VP; tiebreak: fewest built civiliancards."""
        scores = [self._compute_final_score(state, pid) for pid in range(2)]
        if scores[0] > scores[1]:
            return 0
        if scores[1] > scores[0]:
            return 1
        # Tiebreak: civilian card count (fewer = better? official rule: most blue cards)
        civs = [
            sum(1 for cn in state.get_board(pid).built_cards
                if self._card_catalog.get(cn) and self._card_catalog[cn].card_type == "civilian")
            for pid in range(2)
        ]
        return 0 if civs[0] >= civs[1] else 1
