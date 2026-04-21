"""Demonstration buffer and synthetic data generator — Sprint 5.

DemonstrationBuffer stores (obs, action, next_obs, reward, done) transitions
collected from expert demonstrations (GreedyAgent or BGA logs) for use in
Behavioural Cloning pre-training.

SyntheticDemoGenerator rolls out GreedyAgent through WingspanEnv to produce
demonstrations without requiring external log data.
"""

from __future__ import annotations

import gzip
import logging
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """One environment step from an expert demonstration.

    Args:
        obs: Gym observation dict at the start of this step (player 0's view).
        action: Integer action index chosen by the expert.
        next_obs: Gym observation dict after the step completes.
        reward: Scalar reward received on this step.
        done: True if the episode ended after this step.
        info: Info dict returned by WingspanEnv.step().
    """

    obs: dict[str, np.ndarray]
    action: int
    next_obs: dict[str, np.ndarray]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DemonstrationBuffer
# ---------------------------------------------------------------------------


class DemonstrationBuffer:
    """Stores expert transitions grouped by game for Behavioural Cloning.

    Each game is added via add_game() and tagged with the winning player.
    Supports sampling random (obs, action) batches for supervised BC training
    and filtering to only keep winning-side demonstrations.
    """

    def __init__(self) -> None:
        self._transitions: list[Transition] = []
        # Each entry: (start_idx, end_idx, winner)
        self._game_ranges: list[tuple[int, int, int | None]] = []
        self._n_games: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_transition(self, t: Transition) -> None:
        """Add a single transition outside the context of a full game."""
        self._transitions.append(t)

    def add_game(
        self,
        transitions: list[Transition],
        winner: int | None = None,
    ) -> None:
        """Append all transitions from one completed game.

        Args:
            transitions: Ordered list of Transition objects for this game.
            winner: Player index (0 or 1) who won, or None for draws.
        """
        if not transitions:
            return
        start = len(self._transitions)
        end = start + len(transitions)
        self._game_ranges.append((start, end, winner))
        self._transitions.extend(transitions)
        self._n_games += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Sample a random batch of (obs, action) pairs for BC training.

        Args:
            batch_size: Number of transitions to sample.
            rng: Optional seeded RNG for reproducibility.

        Returns:
            Tuple (obs_batch, actions_batch) where:
              obs_batch  — dict of stacked numpy arrays, each (batch_size, ...)
              actions_batch — int32 array of shape (batch_size,)
        """
        if not self._transitions:
            raise ValueError("Buffer is empty.")
        if rng is None:
            rng = random.Random()

        n = len(self._transitions)
        indices = [rng.randint(0, n - 1) for _ in range(batch_size)]
        sampled = [self._transitions[i] for i in indices]

        obs_batch: dict[str, np.ndarray] = {}
        for key in sampled[0].obs:
            obs_batch[key] = np.stack([t.obs[key] for t in sampled], axis=0)

        actions_batch = np.array([t.action for t in sampled], dtype=np.int64)
        return obs_batch, actions_batch

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_winner(self, player_id: int = 0) -> "DemonstrationBuffer":
        """Return a new buffer containing only transitions from won games.

        Args:
            player_id: Keep games where this player was the winner.

        Returns:
            New DemonstrationBuffer with subset of transitions.
        """
        result = DemonstrationBuffer()
        for start, end, winner in self._game_ranges:
            if winner == player_id:
                result.add_game(self._transitions[start:end], winner=winner)
        logger.info(
            "filter_by_winner(player_id=%d): %d → %d transitions (%d games)",
            player_id, len(self._transitions), len(result), result.n_games,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialize buffer to a gzip-compressed pickle file.

        Args:
            path: Output file path (conventionally ends in .pkl.gz).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "transitions": self._transitions,
            "game_ranges": self._game_ranges,
            "n_games": self._n_games,
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %d transitions (%d games) to %s", len(self), self._n_games, path)

    @classmethod
    def load(cls, path: Path | str) -> "DemonstrationBuffer":
        """Deserialize buffer from a gzip-compressed pickle file.

        Args:
            path: Path to the .pkl.gz file created by save().

        Returns:
            Reconstructed DemonstrationBuffer.
        """
        path = Path(path)
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        buf = cls()
        buf._transitions = data["transitions"]
        buf._game_ranges = data["game_ranges"]
        buf._n_games = data["n_games"]
        logger.info("Loaded %d transitions (%d games) from %s", len(buf), buf._n_games, path)
        return buf

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_games(self) -> int:
        return self._n_games

    @property
    def win_count(self) -> int:
        """Number of games where player 0 won."""
        return sum(1 for _, _, w in self._game_ranges if w == 0)

    def __len__(self) -> int:
        return len(self._transitions)

    def __repr__(self) -> str:
        return (
            f"DemonstrationBuffer(transitions={len(self)}, "
            f"games={self._n_games}, p0_wins={self.win_count})"
        )


# ---------------------------------------------------------------------------
# Synthetic demonstration generator
# ---------------------------------------------------------------------------


class SyntheticDemoGenerator:
    """Generates expert demonstrations using GreedyAgent through WingspanEnv.

    This produces BC training data without requiring external log files.
    GreedyAgent serves as the expert: it consistently prefers high-value
    birds and structured play over random actions, providing a useful
    but imperfect supervisor for BC pre-training.

    Args:
        reward_mode: Reward mode passed to WingspanEnv ("dense", "terminal",
            or "shaped").
    """

    def __init__(self, reward_mode: str = "dense", game: str = "wingspan") -> None:
        self._reward_mode = reward_mode
        self._game = game

    def generate(
        self,
        n_games: int,
        seed: int = 0,
        only_wins: bool = False,
    ) -> DemonstrationBuffer:
        """Roll out GreedyAgent for n_games and collect Transitions.

        Args:
            n_games: Number of complete games to simulate.
            seed: Base seed for environment resets.
            only_wins: If True, return only transitions from games where
                player 0 (the GreedyAgent) won.

        Returns:
            Populated DemonstrationBuffer.
        """
        if self._game == "seven_wonders_duel":
            from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv
            env = SevenWondersDuelEnv(reward_mode=self._reward_mode)
            expert_fn = self._greedy_wingspan_action
        elif self._game == "splendor":
            from src.envs.splendor_env import SplendorEnv
            env = SplendorEnv(reward_mode=self._reward_mode)
            expert_fn = self._greedy_splendor_action
        else:
            from src.agents.baselines import GreedyAgent
            from src.envs.wingspan_env import WingspanEnv
            env = WingspanEnv(reward_mode=self._reward_mode)
            _greedy = GreedyAgent()
            expert_fn = lambda state, legal: _greedy.select_action(state, legal)  # noqa: E731
        rng = random.Random(seed)
        buffer = DemonstrationBuffer()

        for game_idx in range(n_games):
            game_seed = rng.randint(0, 100_000)
            obs, _ = env.reset(seed=game_seed)
            game_transitions: list[Transition] = []
            done = False

            while not done:
                legal = env._engine.get_legal_actions(env._state)
                expert_action = expert_fn(env._state, legal)

                if self._game == "splendor":
                    if expert_action is None:
                        mask = env.action_masks()
                        action_idx: int = int(np.argmax(mask))
                    else:
                        from src.games.splendor.actions import action_to_index
                        try:
                            action_idx = action_to_index(expert_action)
                        except (ValueError, KeyError):
                            mask = env.action_masks()
                            action_idx = int(np.argmax(mask))
                else:
                    action_idx = env._action_to_idx(expert_action)  # type: ignore[attr-defined]

                # _action_to_idx can return None for edge-case actions;
                # fall back to the first legal masked index
                if action_idx is None:
                    mask = env.action_masks()
                    action_idx = int(np.argmax(mask))
                    logger.debug(
                        "Game %d: expert action mapped to None, fallback idx=%d",
                        game_idx, action_idx,
                    )

                next_obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated

                game_transitions.append(Transition(
                    obs={k: v.copy() for k, v in obs.items()},
                    action=action_idx,
                    next_obs={k: v.copy() for k, v in next_obs.items()},
                    reward=float(reward),
                    done=done,
                    info=dict(info),
                ))
                obs = next_obs

            winner_from_info = info.get("winner")
            if winner_from_info is not None:
                winner: int | None = winner_from_info
            else:
                s0 = info.get("player_0_score", 0)
                s1 = info.get("player_1_score", 0)
                winner = 0 if s0 > s1 else (1 if s1 > s0 else None)
            buffer.add_game(game_transitions, winner=winner)

            if (game_idx + 1) % max(1, n_games // 5) == 0:
                logger.info(
                    "Generated %d / %d games  (buffer: %d transitions)",
                    game_idx + 1, n_games, len(buffer),
                )

        env.close()

        if only_wins:
            return buffer.filter_by_winner(player_id=0)
        return buffer

    @staticmethod
    def _greedy_wingspan_action(state: "Any", legal: list) -> "Any":
        """Delegate to GreedyAgent for Wingspan / SWD (action types match)."""
        from src.agents.baselines import GreedyAgent
        return GreedyAgent().select_action(state, legal)

    @staticmethod
    def _greedy_splendor_action(state: "Any", legal: list) -> "Any":
        """Simple greedy for Splendor: buy highest-VP card, else take gems.

        Priority:
          1. Buy board/reserved card with highest VP
          2. Take 2 of same gem (if possible)
          3. Take 3 different gems
          4. Any legal action
          Returns None if legal is empty (degenerate state — caller uses mask fallback).
        """
        if not legal:
            return None

        from src.games.splendor.actions import SplendorActionType
        from src.games.splendor.cards import CARDS_BY_ID

        buy_actions = [
            a for a in legal
            if a.action_type in (SplendorActionType.BUY_BOARD, SplendorActionType.BUY_RESERVED)
            and a.card_id and a.card_id in CARDS_BY_ID
        ]
        if buy_actions:
            return max(buy_actions, key=lambda a: CARDS_BY_ID[a.card_id].vp)

        take2 = [a for a in legal if a.action_type == SplendorActionType.TAKE_2_GEMS]
        if take2:
            return take2[0]

        take3 = [a for a in legal if a.action_type == SplendorActionType.TAKE_3_GEMS
                 and len(a.gems_taken) == 3]
        if take3:
            return take3[0]

        return legal[0]
