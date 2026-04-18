"""Gymnasium wrappers for normalisation and frame stacking — Sprint 3."""

from __future__ import annotations

import gymnasium as gym


class NormaliseObsWrapper(gym.ObservationWrapper):
    """Normalises continuous observation components to [0, 1]."""

    def observation(self, obs):  # type: ignore[override]
        raise NotImplementedError("S3 — implement in Sprint 3")
