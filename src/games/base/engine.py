from abc import ABC, abstractmethod

from src.games.base.game_state import Action, ActionResult, GameState


class GameEngine(ABC):
    """Abstract base for all game engines.

    Each concrete engine encodes the rules of one board game as deterministic
    Python. The LLM is used offline (during development) to help write this
    code; it is NOT called during inference or RL training.
    """

    @abstractmethod
    def reset(self) -> GameState:
        """Return the initial game state for a new episode."""
        ...

    @abstractmethod
    def step(self, state: GameState, action: Action) -> ActionResult:
        """Apply action to state and return the result.

        Args:
            state: Current game state (not mutated; a new state is returned).
            action: The action to apply.

        Returns:
            ActionResult with success flag, new state, events, and reward.
        """
        ...

    @abstractmethod
    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Return all legal actions for the current player in state.

        Must never return an empty list when is_terminal(state) is False.
        """
        ...

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Return True if the game is over."""
        ...

    @abstractmethod
    def get_winner(self, state: GameState) -> int | None:
        """Return the winning player_id, or None if the game is not over."""
        ...
