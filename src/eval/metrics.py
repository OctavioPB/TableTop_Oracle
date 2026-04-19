"""Evaluation metrics for TabletopOracle agents — Sprint 6.

All metrics operate on the engine level (no gym env) so they work for any
BaseAgent subclass including baselines and PPO-wrapped agents.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.eval.game_runner import GameResult, run_game

logger = logging.getLogger(__name__)


def win_rate(
    agent_a: Any,
    agent_b: Any,
    engine: Any,
    n_games: int = 500,
    base_seed: int = 0,
) -> float:
    """Estimate agent_a win rate against agent_b over n_games.

    Player roles alternate every game to cancel out first-mover advantage:
    even-indexed games → agent_a is P0; odd-indexed → agent_a is P1.

    Args:
        agent_a: Challenger agent (BaseAgent).
        agent_b: Opponent agent (BaseAgent).
        engine: WingspanEngine instance.
        n_games: Total number of games to play.
        base_seed: First RNG seed; each game uses base_seed + game_index.

    Returns:
        Win rate of agent_a in [0.0, 1.0].
    """
    wins = 0

    for i in range(n_games):
        seed = base_seed + i
        if i % 2 == 0:
            result: GameResult = run_game(agent_a, agent_b, engine, seed=seed)
            if result.winner == 0:
                wins += 1
        else:
            result = run_game(agent_b, agent_a, engine, seed=seed)
            if result.winner == 1:
                wins += 1

    rate = wins / n_games
    logger.info(
        "win_rate: %d/%d games won (%.3f) over %d games",
        wins, n_games, rate, n_games,
    )
    return rate


def avg_score(
    agent: Any,
    engine: Any,
    n_games: int = 200,
    base_seed: int = 0,
) -> tuple[float, float]:
    """Return (mean, std) of agent score as P0 over n_games.

    Args:
        agent: BaseAgent to measure.
        engine: WingspanEngine instance.
        n_games: Number of games to play.
        base_seed: First RNG seed.

    Returns:
        Tuple (mean_score, std_score) over the n_games runs.
    """
    from src.agents.baselines import RandomAgent

    opponent = RandomAgent(seed=999)
    scores: list[float] = []

    for i in range(n_games):
        result = run_game(agent, opponent, engine, seed=base_seed + i)
        if result.scores:
            scores.append(float(result.scores[0]))

    if not scores:
        return 0.0, 0.0

    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std())


def score_distribution(
    agent: Any,
    engine: Any,
    n_games: int = 200,
    base_seed: int = 0,
) -> list[int]:
    """Return the list of agent (P0) scores from n_games for histogram analysis.

    Args:
        agent: BaseAgent to measure.
        engine: WingspanEngine instance.
        n_games: Number of games to play.
        base_seed: First RNG seed.

    Returns:
        List of integer scores, one per game.
    """
    from src.agents.baselines import RandomAgent

    opponent = RandomAgent(seed=999)
    scores: list[int] = []

    for i in range(n_games):
        result = run_game(agent, opponent, engine, seed=base_seed + i)
        if result.scores:
            scores.append(result.scores[0])

    return scores


def rule_violation_rate(
    agent: Any,
    engine: Any,
    n_games: int = 100,
    base_seed: int = 0,
) -> float:
    """Fraction of attempted actions that were illegal over n_games.

    For MaskablePPO agents this should always be 0.0 — the metric is most
    relevant for LLM-only or rule-free baselines.  The implementation tracks
    violations by attempting the agent's chosen action and checking whether the
    engine raises an error or returns an illegal-move result.

    Because WingspanEngine always has action masking available via
    get_legal_actions(), we compute violations as:
        |{actions not in legal_set}| / total_action_attempts

    Args:
        agent: BaseAgent under evaluation.
        engine: WingspanEngine instance.
        n_games: Number of games to play.
        base_seed: First RNG seed.

    Returns:
        Violation rate in [0.0, 1.0]. Well-behaved agents score 0.0.
    """
    from src.agents.baselines import RandomAgent

    opponent = RandomAgent(seed=999)
    total_actions = 0
    violations = 0

    for i in range(n_games):
        state = engine.reset(seed=base_seed + i)
        n_turns = 0

        while not engine.is_terminal(state) and n_turns < 200:
            if state.player_id == 0:
                legal = engine.get_legal_actions(state)
                legal_set = set(id(a) for a in legal)

                chosen = agent.select_action(state, legal)
                total_actions += 1

                # A violation means the agent chose something NOT in legal set
                # (only detectable if the agent ignores the provided list).
                # We verify by checking if chosen is among legal_actions.
                is_legal = any(
                    chosen.action_type == a.action_type
                    and chosen.food_choice == a.food_choice
                    and chosen.card_name == a.card_name
                    and chosen.target_habitat == a.target_habitat
                    and chosen.draw_source == a.draw_source
                    for a in legal
                )
                if not is_legal:
                    violations += 1
                    # Substitute a random legal action to continue the game
                    chosen = legal[0]

                result = engine.step(state, chosen)
                state = result.new_state
            else:
                # Opponent turn
                legal = engine.get_legal_actions(state)
                action = opponent.select_action(state, legal)
                result = engine.step(state, action)
                state = result.new_state

            n_turns += 1

    if total_actions == 0:
        return 0.0
    rate = violations / total_actions
    logger.info(
        "rule_violation_rate: %d violations / %d actions = %.4f",
        violations, total_actions, rate,
    )
    return rate


def steps_to_target_winrate(
    win_rate_history: list[dict],
    target: float = 0.55,
) -> int | None:
    """Find the first training timestep where win_rate_vs_random reached target.

    Operates on the list of dicts produced by WinRateCallback.win_rate_history:
        [{"timestep": int, "win_rate_vs_random": float, ...}, ...]

    Args:
        win_rate_history: List of evaluation checkpoints from WinRateCallback.
        target: Win rate threshold (default 0.55 = clearly beating random).

    Returns:
        The training timestep at which win_rate_vs_random first met or exceeded
        target, or None if the target was never reached.
    """
    for entry in win_rate_history:
        if entry.get("win_rate_vs_random", 0.0) >= target:
            return int(entry["timestep"])
    return None
