"""Round-robin tournament with Elo scoring — Sprint 6."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloTable:
    ratings: dict[str, float] = field(default_factory=dict)


class Tournament:
    """Runs a round-robin tournament and computes approximate Elo ratings."""

    def run(
        self, agents: dict[str, object], n_games_per_pair: int = 200
    ) -> EloTable:
        raise NotImplementedError("S6.2 — implement in Sprint 6")
