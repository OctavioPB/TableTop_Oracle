"""Wingspan gymnasium environment with action masking — Sprint 3."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from src.games.wingspan.engine import WingspanEngine
from src.games.wingspan.rewards import RewardMode

logger = logging.getLogger(__name__)

N_MAX_ACTIONS = 256


class WingspanEnv(gym.Env):
    """gym.Env wrapper for Wingspan, compatible with MaskablePPO.

    action_masks() is required by sb3_contrib.MaskablePPO.
    The observation space uses gymnasium.spaces.Dict with sub-spaces
    per board component (see Sprint 3 design doc in PLAN.md S3.2).
    """

    metadata = {"render_modes": ["text", "ansi"]}

    def __init__(
        self,
        num_players: int = 2,
        reward_mode: RewardMode = "dense",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self._engine = WingspanEngine()
        self._state = None

        # Defined in Sprint 3
        self.observation_space: gym.Space = gym.spaces.Discrete(1)  # placeholder
        self.action_space: gym.Space = gym.spaces.Discrete(N_MAX_ACTIONS)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        raise NotImplementedError("S3.1 — implement in Sprint 3")

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        raise NotImplementedError("S3.1 — implement in Sprint 3")

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of legal actions — required by MaskablePPO."""
        raise NotImplementedError("S3.3 — implement in Sprint 3")

    def render(self) -> str | None:
        raise NotImplementedError("S3.1 — implement in Sprint 3")
