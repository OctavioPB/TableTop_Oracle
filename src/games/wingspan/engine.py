"""Wingspan game engine — deterministic simulator for 2-player games.

Design decisions:
- State is immutable in the public API (step() returns new state via model_copy)
- LLM is NEVER called here — this is the hot path for RL training
- Bird powers resolve deterministically; multi-step powers are heuristically flattened
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

from src.games.base.engine import GameEngine
from src.games.base.game_state import Action, ActionResult
from src.games.wingspan.actions import WingspanAction, WingspanActionType
from src.games.wingspan.cards import (
    ALL_ROUND_GOALS,
    GOAL_POINTS_2P,
    BirdCard,
    FoodType,
    PowerID,
    PowerTiming,
    RoundGoalType,
    load_bird_catalog,
)
from src.games.wingspan.state import (
    N_HABITAT_SLOTS,
    N_ROUNDS,
    TURNS_PER_ROUND,
    BirdSlotState,
    WingspanPlayerBoard,
    WingspanState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level catalog registry (shared between engine and powers module)
# ---------------------------------------------------------------------------

_GLOBAL_CATALOG: dict[str, BirdCard] = {}

_DEFAULT_CATALOG_PATH = Path(
    os.environ.get(
        "CARD_CATALOG_DIR",
        str(Path(__file__).parent.parent.parent.parent / "data" / "card_catalogs"),
    )
) / "wingspan_birds.csv"

# Feeder die faces (weighted like real dice: each food type appears once or twice)
_FEEDER_DIE_FACES: list[str] = [
    FoodType.SEED.value,
    FoodType.SEED.value,
    FoodType.INVERTEBRATE.value,
    FoodType.INVERTEBRATE.value,
    FoodType.FRUIT.value,
    FoodType.RODENT.value,
    FoodType.FISH.value,
]


def _roll_feeder(rng: random.Random, n: int = 5) -> list[str]:
    """Roll n feeder dice. Returns list of FoodType values."""
    return [rng.choice(_FEEDER_DIE_FACES) for _ in range(n)]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WingspanEngine(GameEngine):
    """2-player Wingspan simulator.

    The catalog is loaded once at construction. All state transitions
    return a new WingspanState via model_copy (no mutation of inputs).
    """

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        global _GLOBAL_CATALOG

        path = Path(catalog_path) if catalog_path else _DEFAULT_CATALOG_PATH
        if path.exists():
            self._catalog = load_bird_catalog(path)
            _GLOBAL_CATALOG = self._catalog
        else:
            logger.warning(
                "Bird catalog not found at %s — engine will use empty catalog", path
            )
            self._catalog = {}
            _GLOBAL_CATALOG = {}

        self._seed = seed
        self._rng = random.Random(seed)

        from src.games.wingspan.rules import LegalMoveValidator
        self._validator = LegalMoveValidator(self._catalog)

    # ------------------------------------------------------------------
    # GameEngine interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> WingspanState:
        """Return the initial state for a new 2-player Wingspan game."""
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self._seed)

        all_birds = list(self._catalog.keys())
        if len(all_birds) < 20:
            raise RuntimeError(
                f"Bird catalog has only {len(all_birds)} birds. "
                f"Need at least 20. Ensure the CSV is loaded correctly."
            )

        self._rng.shuffle(all_birds)

        # Deal: 5 cards to player 0, 5 to player 1, 3 to tray
        hand_size = 5
        p0_hand = all_birds[:hand_size]
        p1_hand = all_birds[hand_size : hand_size * 2]
        bird_tray = all_birds[hand_size * 2 : hand_size * 2 + 3]
        deck = all_birds[hand_size * 2 + 3 :]

        # Starting food: 5 random food tokens per player
        def random_food(n: int) -> dict[str, int]:
            food: dict[str, int] = {}
            real_types = [ft.value for ft in FoodType if ft != FoodType.WILD]
            for _ in range(n):
                f = self._rng.choice(real_types)
                food[f] = food.get(f, 0) + 1
            return food

        boards = [
            WingspanPlayerBoard(
                player_id=i,
                hand=list(hand),
                food_supply=random_food(5),
                action_cubes=TURNS_PER_ROUND[1],
            )
            for i, hand in [(0, p0_hand), (1, p1_hand)]
        ]

        # Select 4 round-end goals randomly
        goals = self._rng.sample(ALL_ROUND_GOALS, 4)

        return WingspanState(
            player_id=0,
            turn=0,
            phase="main",
            boards_data=[b.to_dict() for b in boards],
            bird_feeder=_roll_feeder(self._rng),
            bird_tray=list(bird_tray),
            draw_deck=list(deck),
            round=1,
            round_end_goals=goals,
        )

    def step(self, state: WingspanState, action: WingspanAction) -> ActionResult:
        """Apply action to state and return the result."""
        if not isinstance(action, WingspanAction):
            raise TypeError(f"Expected WingspanAction, got {type(action)}")

        board = state.get_board(state.player_id)
        boards = state.get_boards()

        events: list[str] = []
        reward: float = 0.0

        atype = action.action_type

        if atype == WingspanActionType.GAIN_FOOD.value:
            state, board, events = self._apply_gain_food(state, board, action, events)

        elif atype == WingspanActionType.LAY_EGGS.value:
            state, board, events = self._apply_lay_eggs(state, board, events)

        elif atype == WingspanActionType.DRAW_CARDS.value:
            state, board, events = self._apply_draw_cards(state, board, action, events)

        elif atype == WingspanActionType.PLAY_BIRD.value:
            state, board, events, reward = self._apply_play_bird(
                state, board, action, events
            )
        else:
            raise ValueError(f"Unknown action type: {atype}")

        # Trigger pink 'once_between_turns' powers for other players
        state, boards = self._trigger_pink_powers(state, board, boards, atype)

        # Spend one action cube
        board.action_cubes -= 1
        boards[state.player_id] = board
        state = state.with_boards(boards)

        # Advance turn
        new_turn = state.turn + 1
        state = state.model_copy(update={"turn": new_turn})

        # Check if round ends (all players have 0 cubes)
        if state.all_players_done():
            state, events_round = self._end_round(state)
            events.extend(events_round)

        done = self.is_terminal(state)

        if done:
            winner = self.get_winner(state)
            state = state.model_copy(update={"phase": "finished"})
            reward += 1.0 if winner == state.player_id else -1.0

        return ActionResult(
            success=True,
            new_state=state,
            events=events,
            reward=reward,
        )

    def get_legal_actions(self, state: WingspanState) -> list[WingspanAction]:
        return self._validator.get_legal_actions(state, state.player_id)

    def is_terminal(self, state: WingspanState) -> bool:
        return state.phase == "finished" or (
            state.round > N_ROUNDS and state.all_players_done()
        )

    def get_winner(self, state: WingspanState) -> int | None:
        if not self.is_terminal(state):
            return None
        scores = [self._compute_final_score(state, pid) for pid in range(len(state.boards_data))]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        # Tiebreak: highest total food (Wingspan standard tiebreak)
        if len(winners) > 1:
            best_food = max(
                sum(state.get_board(w).food_supply.values()) for w in winners
            )
            winners = [
                w for w in winners
                if sum(state.get_board(w).food_supply.values()) == best_food
            ]
        return winners[0]

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _apply_gain_food(
        self,
        state: WingspanState,
        board: WingspanPlayerBoard,
        action: WingspanAction,
        events: list[str],
    ) -> tuple[WingspanState, WingspanPlayerBoard, list[str]]:
        feeder = list(state.bird_feeder)
        food_choice = action.food_choice

        # Auto-reroll if feeder is empty
        if not feeder:
            feeder = _roll_feeder(self._rng)
            events.append("Bird feeder rerolled (was empty)")

        # Reroll if all remaining show same face
        if len(feeder) >= 2 and len(set(feeder)) == 1:
            feeder = _roll_feeder(self._rng, len(feeder))
            events.append("Bird feeder rerolled (all same face)")

        # Take one die of the chosen type (or first available)
        if food_choice in feeder:
            feeder.remove(food_choice)
        else:
            food_choice = feeder.pop(0)

        board.food_supply[food_choice] = board.food_supply.get(food_choice, 0) + 1
        events.append(f"Gained 1 {food_choice} from feeder")
        state = state.model_copy(update={"bird_feeder": feeder})

        # Bonus food from forest birds (right to left, including the food just gained)
        n_bonus = len(board.birds_in_habitat("forest"))
        for _ in range(n_bonus):
            if feeder:
                bonus = feeder.pop(0)
                board.food_supply[bonus] = board.food_supply.get(bonus, 0) + 1
                events.append(f"Forest bonus: gained {bonus}")
                state = state.model_copy(update={"bird_feeder": feeder})

        # Activate brown powers in forest (right to left)
        state, board, events = self._activate_habitat_powers(
            state, board, "forest", events
        )

        return state, board, events

    def _apply_lay_eggs(
        self,
        state: WingspanState,
        board: WingspanPlayerBoard,
        events: list[str],
    ) -> tuple[WingspanState, WingspanPlayerBoard, list[str]]:
        n_birds_in_grassland = len(board.birds_in_habitat("grassland"))
        n_eggs = 2 + n_birds_in_grassland

        from src.games.wingspan.powers import _lay_egg_best_slot
        board, ev = _lay_egg_best_slot(board, self._catalog, n_eggs)
        events.extend(ev)
        events.append(f"Lay eggs action: placed up to {n_eggs} eggs")

        # Activate brown powers in grassland (right to left)
        state, board, events = self._activate_habitat_powers(
            state, board, "grassland", events
        )

        return state, board, events

    def _apply_draw_cards(
        self,
        state: WingspanState,
        board: WingspanPlayerBoard,
        action: WingspanAction,
        events: list[str],
    ) -> tuple[WingspanState, WingspanPlayerBoard, list[str]]:
        n_birds_in_wetland = len(board.birds_in_habitat("wetland"))
        total_draws = 1 + n_birds_in_wetland

        source = action.draw_source
        tray = list(state.bird_tray)
        deck = list(state.draw_deck)
        drawn = 0

        # First card: from specified source
        if source.startswith("tray_"):
            idx = int(source.split("_")[1])
            if idx < len(tray):
                card = tray.pop(idx)
                board.hand.append(card)
                # Refill tray from deck
                if deck:
                    tray.insert(idx, deck.pop(0))
                events.append(f"Drew {card} from tray slot {idx}")
                drawn += 1
        elif source == "deck" and deck:
            card = deck.pop(0)
            board.hand.append(card)
            events.append(f"Drew card from deck")
            drawn += 1

        # Remaining draws come from deck
        for _ in range(total_draws - drawn):
            if deck:
                card = deck.pop(0)
                board.hand.append(card)
                drawn += 1

        state = state.model_copy(update={"bird_tray": tray, "draw_deck": deck})

        # Activate brown powers in wetland (right to left)
        state, board, events = self._activate_habitat_powers(
            state, board, "wetland", events
        )

        return state, board, events

    def _apply_play_bird(
        self,
        state: WingspanState,
        board: WingspanPlayerBoard,
        action: WingspanAction,
        events: list[str],
    ) -> tuple[WingspanState, WingspanPlayerBoard, list[str], float]:
        card_name = action.card_name
        habitat = action.target_habitat
        reward = 0.0

        # Remove card from hand
        if card_name not in board.hand:
            raise ValueError(f"Card {card_name} not in hand")
        board.hand.remove(card_name)

        # Pay food
        for food, amount in action.food_payment.items():
            board.food_supply[food] = board.food_supply.get(food, 0) - amount
            if board.food_supply[food] <= 0:
                board.food_supply.pop(food, None)

        # Pay eggs (remove from birds with most eggs first — greedy)
        eggs_to_remove = action.egg_payment
        for hab in ("forest", "grassland", "wetland"):
            for i, slot in sorted(
                board.birds_in_habitat(hab), key=lambda x: -x[1].eggs
            ):
                if eggs_to_remove <= 0:
                    break
                removed = min(slot.eggs, eggs_to_remove)
                slot.eggs -= removed
                eggs_to_remove -= removed

        # Place bird in habitat
        slot_idx = board.first_empty_slot(habitat)
        if slot_idx is None:
            raise ValueError(f"No empty slot in {habitat}")
        board.get_habitat(habitat)[slot_idx] = BirdSlotState(bird_name=card_name)

        card = self._catalog.get(card_name)
        if card:
            reward += card.points * 0.01  # small dense reward for bird value

        events.append(f"Played {card_name} to {habitat} slot {slot_idx}")

        # Resolve when-played power
        if card and card.power_timing == PowerTiming.WHEN_PLAYED.value:
            from src.games.wingspan.powers import execute_when_played_power
            state, board, ev = execute_when_played_power(state, board, card, self._rng)
            events.extend(ev)

        return state, board, events, reward

    # ------------------------------------------------------------------
    # Habitat power activation
    # ------------------------------------------------------------------

    def _activate_habitat_powers(
        self,
        state: WingspanState,
        board: WingspanPlayerBoard,
        habitat: str,
        events: list[str],
    ) -> tuple[WingspanState, WingspanPlayerBoard, list[str]]:
        """Activate all brown when_activated powers in habitat, right to left."""
        from src.games.wingspan.powers import execute_when_activated_power

        slots = board.birds_in_habitat(habitat)
        # Right to left = descending slot index
        for idx, slot in sorted(slots, key=lambda x: -x[0]):
            card = self._catalog.get(slot.bird_name)
            if not card:
                continue
            if card.power_timing != PowerTiming.WHEN_ACTIVATED.value:
                continue
            state, board, ev = execute_when_activated_power(
                state, board, slot, card, self._rng
            )
            events.extend(ev)
        return state, board, events

    # ------------------------------------------------------------------
    # Pink power triggers
    # ------------------------------------------------------------------

    def _trigger_pink_powers(
        self,
        state: WingspanState,
        active_board: WingspanPlayerBoard,
        boards: list[WingspanPlayerBoard],
        trigger_action: str,
    ) -> tuple[WingspanState, list[WingspanPlayerBoard]]:
        from src.games.wingspan.powers import execute_once_between_turns

        for pid, passive_board in enumerate(boards):
            if pid == state.player_id:
                continue
            for hab in ("forest", "grassland", "wetland"):
                for _, slot in passive_board.birds_in_habitat(hab):
                    card = self._catalog.get(slot.bird_name)
                    if not card:
                        continue
                    if card.power_timing != "once_between_turns":
                        continue
                    state, passive_board, _ = execute_once_between_turns(
                        state, passive_board, slot, card, trigger_action, self._rng
                    )
            boards[pid] = passive_board

        return state, boards

    # ------------------------------------------------------------------
    # Round end
    # ------------------------------------------------------------------

    def _end_round(
        self, state: WingspanState
    ) -> tuple[WingspanState, list[str]]:
        events: list[str] = []
        boards = state.get_boards()
        current_round = state.round

        # Activate teal end-of-round powers
        for pid, board in enumerate(boards):
            for hab in ("forest", "grassland", "wetland"):
                for _, slot in board.birds_in_habitat(hab):
                    card = self._catalog.get(slot.bird_name)
                    if not card:
                        continue
                    if card.power_timing != "end_of_round":
                        continue
                    from src.games.wingspan.powers import execute_end_of_round_power
                    state, board, ev = execute_end_of_round_power(
                        state, board, slot, card
                    )
                    events.extend(ev)
            boards[pid] = board

        # Score round-end goal
        if current_round <= len(state.round_end_goals):
            goal = state.round_end_goals[current_round - 1]
            boards, ev = self._score_round_goal(boards, goal, current_round - 1)
            events.extend(ev)
        state = state.with_boards(boards)

        # Check if game is over
        if current_round >= N_ROUNDS:
            state = state.model_copy(update={"round": current_round + 1, "phase": "finished"})
            events.append("Game over — final round complete")
            return state, events

        # Start next round
        next_round = current_round + 1
        cubes = {1: 8, 2: 7, 3: 6, 4: 5}[next_round]
        boards = state.get_boards()
        for b in boards:
            b.action_cubes = cubes
        state = state.with_boards(boards)

        # Next player first (player_id stays player 0 — turn order never changes)
        state = state.model_copy(
            update={
                "round": next_round,
                "player_id": 0,
                "phase": "main",
                "bird_feeder": _roll_feeder(self._rng),
            }
        )

        # Refresh bird tray for next round
        tray = list(state.bird_tray)
        deck = list(state.draw_deck)
        discard = list(state.discard_pile)
        discard.extend(tray)
        tray = []
        for _ in range(3):
            if deck:
                tray.append(deck.pop(0))
        state = state.model_copy(
            update={"bird_tray": tray, "draw_deck": deck, "discard_pile": discard}
        )

        events.append(f"Round {next_round} begins — action cubes reset to {cubes}")
        return state, events

    def _score_round_goal(
        self,
        boards: list[WingspanPlayerBoard],
        goal: str,
        goal_idx: int,
    ) -> tuple[list[WingspanPlayerBoard], list[str]]:
        events: list[str] = []
        scores = [self._evaluate_goal(b, goal) for b in boards]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        positions: dict[int, int] = {}
        prev_score = None
        rank = 1
        for pid in ranked:
            if scores[pid] == prev_score:
                positions[pid] = positions[ranked[ranked.index(pid) - 1]]
            else:
                positions[pid] = rank
                rank += 1
            prev_score = scores[pid]

        for pid, pos in positions.items():
            points = GOAL_POINTS_2P.get(pos, 0)
            boards[pid].goal_positions[goal_idx] = points
            events.append(
                f"Round goal '{goal}': P{pid} scored {scores[pid]} → {points} pts (pos {pos})"
            )
        return boards, events

    @staticmethod
    def _evaluate_goal(board: WingspanPlayerBoard, goal: str) -> int:
        if goal == RoundGoalType.MOST_BIRDS_FOREST.value:
            return len(board.birds_in_habitat("forest"))
        if goal == RoundGoalType.MOST_BIRDS_GRASSLAND.value:
            return len(board.birds_in_habitat("grassland"))
        if goal == RoundGoalType.MOST_BIRDS_WETLAND.value:
            return len(board.birds_in_habitat("wetland"))
        if goal == RoundGoalType.MOST_BIRDS_TOTAL.value:
            return board.total_birds()
        if goal == RoundGoalType.MOST_EGGS_CUP.value:
            return board.total_eggs()  # simplified: all eggs
        if goal == RoundGoalType.MOST_EGGS_PLATFORM.value:
            return board.total_eggs()
        if goal == RoundGoalType.MOST_EGGS_CAVITY.value:
            return board.total_eggs()
        if goal == RoundGoalType.MOST_EGGS_GROUND.value:
            return board.total_eggs()
        return 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_final_score(self, state: WingspanState, player_id: int) -> int:
        board = state.get_board(player_id)
        score = 0

        # Bird points
        for hab in ("forest", "grassland", "wetland"):
            for _, slot in board.birds_in_habitat(hab):
                card = self._catalog.get(slot.bird_name)
                if card:
                    score += card.points
                # Eggs on birds (1 pt each)
                score += slot.eggs
                # Cached food (1 pt each)
                score += sum(slot.cached_food.values())
                # Tucked cards (1 pt each)
                score += slot.tucked_cards

        # End-of-round goal points
        score += sum(board.goal_positions)

        return score

    def compute_scores(self, state: WingspanState) -> dict[int, int]:
        """Return {player_id: score} for all players (for evaluation)."""
        return {
            pid: self._compute_final_score(state, pid)
            for pid in range(len(state.boards_data))
        }
