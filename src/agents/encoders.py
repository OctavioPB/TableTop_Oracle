"""State encoder: Dict observation → tensor for MaskablePPO — Sprint 4."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WingspanFeaturesExtractor(BaseFeaturesExtractor):
    """Encodes the Dict observation space into a flat feature vector.

    Architecture:
    - MLP for scalar resources (food, eggs, round info)
    - Linear projection for birds_on_board (3 habitats × 5 slots)
    - Linear projection for hand and tray
    - Concatenation → shared MLP trunk
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        raise NotImplementedError("S4.1 — implement in Sprint 4")

    def forward(self, observations) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError("S4.1 — implement in Sprint 4")
