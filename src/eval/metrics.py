"""Evaluation metrics for TabletopOracle agents — Sprint 6."""

from __future__ import annotations


def win_rate(agent_a, agent_b, engine, n_games: int = 500) -> float:
    """Estimate agent_a win rate against agent_b over n_games.

    Args:
        agent_a: Challenger agent.
        agent_b: Opponent agent.
        engine: Game engine.
        n_games: Number of games to play.

    Returns:
        Win rate of agent_a in [0.0, 1.0].
    """
    raise NotImplementedError("S6.1 — implement in Sprint 6")


def avg_score(agent, engine, n_games: int = 200) -> tuple[float, float]:
    """Return (mean, std) of agent score over n_games."""
    raise NotImplementedError("S6.1 — implement in Sprint 6")
