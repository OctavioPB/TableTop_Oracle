"""Baseline agents for evaluation benchmarking — Sprint 4."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from src.games.base.engine import GameEngine
from src.games.base.game_state import Action, GameState


class BaseAgent(ABC):
    """Interface for all agents (baselines and trained)."""

    @abstractmethod
    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Choose an action from the legal action set."""
        ...


class RandomAgent(BaseAgent):
    """Selects uniformly at random from legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        return self._rng.choice(legal_actions)


class GreedyAgent(BaseAgent):
    """Simple heuristic: play highest-point bird available, else lay eggs, else gain food."""

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        raise NotImplementedError("S4.2 — implement in Sprint 4")
