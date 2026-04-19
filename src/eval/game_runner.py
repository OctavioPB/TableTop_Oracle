"""Internal game-running helper shared by metrics and tournament modules.

Runs a single Wingspan game between two BaseAgent instances using the engine
directly (no gym env overhead) and returns the winner and final scores.

Never import this module from outside src/eval/ — it is an implementation
detail of the evaluation framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Maximum turns before we declare a draw (safety valve; real games end in ≤104)
_MAX_TURNS = 200


@dataclass
class GameResult:
    """Outcome of a single completed game.

    Attributes:
        winner: player_id of the winner, or None on draw/timeout.
        scores: Final score per player indexed by player_id.
        n_turns: Number of turns taken before termination.
    """

    winner: int | None
    scores: list[int]
    n_turns: int


def run_game(
    agent_a: Any,
    agent_b: Any,
    engine: Any,
    seed: int | None = None,
) -> GameResult:
    """Play one complete game between agent_a (P0) and agent_b (P1).

    Args:
        agent_a: BaseAgent controlling player 0.
        agent_b: BaseAgent controlling player 1.
        engine: WingspanEngine instance (reused across calls for speed).
        seed: RNG seed passed to engine.reset() for reproducibility.

    Returns:
        GameResult with winner, scores, and turn count.
    """
    state = engine.reset(seed=seed)
    agents = [agent_a, agent_b]
    n_turns = 0

    while not engine.is_terminal(state) and n_turns < _MAX_TURNS:
        current_player = state.player_id
        legal_actions = engine.get_legal_actions(state)

        if not legal_actions:
            logger.warning(
                "No legal actions for player %d at turn %d — terminating early.",
                current_player,
                n_turns,
            )
            break

        action = agents[current_player].select_action(state, legal_actions)
        result = engine.step(state, action)
        state = result.new_state
        n_turns += 1

    winner = engine.get_winner(state)
    scores = _extract_scores(engine, state)

    return GameResult(winner=winner, scores=scores, n_turns=n_turns)


def _extract_scores(engine: Any, state: Any) -> list[int]:
    """Return final scores for all players.

    Uses the engine's internal _compute_final_score when the game is over,
    falling back to 0 for any player whose board cannot be accessed.
    """
    n_players = len(state.boards_data)
    scores: list[int] = []
    for pid in range(n_players):
        try:
            score = engine._compute_final_score(state, pid)
            scores.append(int(score))
        except Exception:  # noqa: BLE001 — defensive; score extraction must not crash eval
            scores.append(0)
    return scores
