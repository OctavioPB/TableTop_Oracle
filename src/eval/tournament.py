"""Round-robin tournament with approximate Elo scoring — Sprint 6.

Elo update rule (standard):
    E_a  = 1 / (1 + 10^((R_b - R_a) / 400))
    R_a' = R_a + K * (S_a - E_a)

where S_a = 1 (win), 0.5 (draw), 0 (loss) and K = 32 (default).
All agents start at Elo 1000.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from src.eval.game_runner import run_game

logger = logging.getLogger(__name__)

_ELO_K = 32.0
_ELO_INITIAL = 1000.0


@dataclass
class EloTable:
    """Elo rating table for a set of named agents.

    Attributes:
        ratings: Mapping from agent name to current Elo rating.
        match_results: Raw win/draw/loss record per agent pair.
    """

    ratings: dict[str, float] = field(default_factory=dict)
    match_results: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def update(
        self,
        winner_name: str,
        loser_name: str,
        draw: bool = False,
        k: float = _ELO_K,
    ) -> None:
        """Update Elo ratings after one game outcome.

        Args:
            winner_name: Name of the winning agent (or first agent if draw).
            loser_name: Name of the losing agent (or second agent if draw).
            draw: If True, treat the result as a draw (S = 0.5 for both).
            k: Elo K-factor.
        """
        r_a = self.ratings.get(winner_name, _ELO_INITIAL)
        r_b = self.ratings.get(loser_name, _ELO_INITIAL)

        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        e_b = 1.0 - e_a

        s_a, s_b = (0.5, 0.5) if draw else (1.0, 0.0)

        self.ratings[winner_name] = r_a + k * (s_a - e_a)
        self.ratings[loser_name] = r_b + k * (s_b - e_b)

        # Record raw result
        for name_a, name_b, result in [
            (winner_name, loser_name, "win" if not draw else "draw"),
            (loser_name, winner_name, "loss" if not draw else "draw"),
        ]:
            if name_a not in self.match_results:
                self.match_results[name_a] = {}
            if name_b not in self.match_results[name_a]:
                self.match_results[name_a][name_b] = {"win": 0, "draw": 0, "loss": 0}
            self.match_results[name_a][name_b][result] += 1

    def standings(self) -> list[tuple[str, float]]:
        """Return agents sorted by Elo rating, highest first.

        Returns:
            List of (agent_name, elo_rating) tuples in descending order.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def __repr__(self) -> str:
        lines = ["EloTable standings:"]
        for rank, (name, rating) in enumerate(self.standings(), start=1):
            lines.append(f"  {rank}. {name:30s}  {rating:.1f}")
        return "\n".join(lines)


class Tournament:
    """Runs a round-robin tournament and computes approximate Elo ratings.

    Each pair of agents plays n_games_per_pair games, with roles alternated to
    cancel first-mover advantage. Elo is updated after every game.

    Args:
        engine: WingspanEngine instance shared across all games.
        base_seed: Starting RNG seed; each game gets a unique seed derived
            from this value, the pair index, and the game index.
    """

    def __init__(self, engine: Any, base_seed: int = 0) -> None:
        self._engine = engine
        self._base_seed = base_seed

    def run(
        self,
        agents: dict[str, Any],
        n_games_per_pair: int = 200,
    ) -> EloTable:
        """Run a full round-robin tournament.

        Args:
            agents: Dict mapping name → BaseAgent. All combinations are played.
            n_games_per_pair: Number of games per ordered pair (A vs B and B vs A
                each get n_games_per_pair // 2 games with alternating seats).

        Returns:
            EloTable with updated ratings after all games.
        """
        table = EloTable(ratings={name: _ELO_INITIAL for name in agents})
        names = list(agents.keys())

        pairs = list(itertools.combinations(names, 2))
        total_games = len(pairs) * n_games_per_pair
        logger.info(
            "Tournament: %d agents, %d pairs, %d games total",
            len(names), len(pairs), total_games,
        )

        for pair_idx, (name_a, name_b) in enumerate(pairs):
            agent_a = agents[name_a]
            agent_b = agents[name_b]
            pair_wins_a = 0
            pair_wins_b = 0
            pair_draws = 0

            for game_idx in range(n_games_per_pair):
                seed = self._base_seed + pair_idx * 10_000 + game_idx

                if game_idx % 2 == 0:
                    result = run_game(agent_a, agent_b, self._engine, seed=seed)
                    winner_id = result.winner
                    if winner_id == 0:
                        table.update(name_a, name_b)
                        pair_wins_a += 1
                    elif winner_id == 1:
                        table.update(name_b, name_a)
                        pair_wins_b += 1
                    else:
                        table.update(name_a, name_b, draw=True)
                        pair_draws += 1
                else:
                    result = run_game(agent_b, agent_a, self._engine, seed=seed)
                    winner_id = result.winner
                    if winner_id == 0:
                        table.update(name_b, name_a)
                        pair_wins_b += 1
                    elif winner_id == 1:
                        table.update(name_a, name_b)
                        pair_wins_a += 1
                    else:
                        table.update(name_a, name_b, draw=True)
                        pair_draws += 1

            logger.info(
                "Pair %s vs %s — W/D/L: %d/%d/%d (Elo: %.1f / %.1f)",
                name_a, name_b,
                pair_wins_a, pair_draws, pair_wins_b,
                table.ratings[name_a], table.ratings[name_b],
            )

        return table
