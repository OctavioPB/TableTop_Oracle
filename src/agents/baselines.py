"""Baseline agents for evaluation benchmarking — Sprint 4.

Used as reference opponents during RL training and for ablation comparisons.
Both agents operate on WingspanAction objects returned by the engine.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any

from src.games.base.game_state import Action, GameState
from src.games.wingspan.actions import WingspanAction, WingspanActionType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Interface for all agents (baselines and trained)."""

    @abstractmethod
    def select_action(
        self,
        state: GameState,
        legal_actions: list[Action],
    ) -> Action:
        """Choose one action from the legal action set."""
        ...


class RandomAgent(BaseAgent):
    """Selects uniformly at random from legal actions.

    The minimum sensible baseline — any learning agent must beat this.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        legal_actions: list[Action],
    ) -> Action:
        return self._rng.choice(legal_actions)


class GreedyAgent(BaseAgent):
    """Simple heuristic that maximises immediate expected value.

    Priority order:
      1. Play the highest-point bird that can be afforded.
      2. Lay eggs (if any bird has room).
      3. Draw cards (always expands options).
      4. Gain food (resource building).

    Rationale: playing birds dominates in Wingspan because each bird
    contributes points every subsequent scoring opportunity. Eggs and food
    are only valuable instrumentally.
    """

    def __init__(
        self,
        catalog: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._catalog: dict[str, Any] = catalog or {}
        self._rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        legal_actions: list[Action],
    ) -> Action:
        if not self._catalog:
            # Lazy-load from the engine's global registry
            try:
                from src.games.wingspan.engine import _GLOBAL_CATALOG
                self._catalog = _GLOBAL_CATALOG
            except ImportError:
                pass

        # 1. Play highest-point bird
        play_actions = [
            a for a in legal_actions
            if a.action_type == WingspanActionType.PLAY_BIRD.value
        ]
        if play_actions:
            def _points(action: WingspanAction) -> int:
                card = self._catalog.get(action.card_name)
                return card.points if card else 0

            return max(play_actions, key=_points)

        # 2. Lay eggs
        for a in legal_actions:
            if a.action_type == WingspanActionType.LAY_EGGS.value:
                return a

        # 3. Draw cards
        for a in legal_actions:
            if a.action_type == WingspanActionType.DRAW_CARDS.value:
                return a

        # 4. Gain food (default)
        for a in legal_actions:
            if a.action_type == WingspanActionType.GAIN_FOOD.value:
                return a

        # Fallback: first legal
        return legal_actions[0]


def evaluate_agents(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    engine: Any,
    n_games: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Run a head-to-head evaluation between two agents.

    Agent A always plays as player 0; agent B as player 1.
    Uses the engine directly (no gym wrapper) for speed.

    Returns:
        dict with keys: win_rate_a, win_rate_b, draw_rate,
        avg_score_a, avg_score_b.
    """
    rng = random.Random(seed)
    wins_a = 0
    wins_b = 0
    draws = 0
    scores_a: list[int] = []
    scores_b: list[int] = []

    for game_idx in range(n_games):
        state = engine.reset(seed=rng.randint(0, 10_000))

        max_steps = 500
        steps = 0

        while not engine.is_terminal(state) and steps < max_steps:
            pid = state.player_id
            legal = engine.get_legal_actions(state)

            if pid == 0:
                action = agent_a.select_action(state, legal)
            else:
                action = agent_b.select_action(state, legal)

            result = engine.step(state, action)
            state = result.new_state

            # Alternate players
            if not engine.is_terminal(state):
                next_pid = 1 - pid
                state = state.model_copy(update={"player_id": next_pid})

            steps += 1

        game_scores = engine.compute_scores(state)
        sa, sb = game_scores.get(0, 0), game_scores.get(1, 0)
        scores_a.append(sa)
        scores_b.append(sb)

        winner = engine.get_winner(state)
        if winner == 0:
            wins_a += 1
        elif winner == 1:
            wins_b += 1
        else:
            draws += 1

    total = n_games
    return {
        "win_rate_a":  wins_a / total,
        "win_rate_b":  wins_b / total,
        "draw_rate":   draws / total,
        "avg_score_a": sum(scores_a) / total,
        "avg_score_b": sum(scores_b) / total,
        "n_games":     total,
    }
