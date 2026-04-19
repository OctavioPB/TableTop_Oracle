"""Gymnasium wrappers for normalisation and frame stacking — Sprint 3.

NormaliseObsWrapper is provided for completeness; WingspanEnv and
SevenWondersDuelEnv already clip all observation values to [0, 1] internally,
so this wrapper is effectively a no-op for those environments.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NormaliseObsWrapper(gym.ObservationWrapper):
    """Clips all Box observations to [0, 1].

    Provided for environments that do not normalise internally.  For
    WingspanEnv and SevenWondersDuelEnv this is redundant because those
    environments already return observations in [0, 1].
    """

    def observation(self, obs):  # type: ignore[override]
        if isinstance(obs, dict):
            return {
                k: np.clip(v, 0.0, 1.0).astype(np.float32)
                for k, v in obs.items()
            }
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
