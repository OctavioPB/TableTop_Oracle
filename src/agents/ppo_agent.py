"""MaskablePPO agent configuration and helpers — Sprint 4."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_maskable_ppo(env, features_dim: int = 256):  # type: ignore[return]
    """Construct a MaskablePPO model with project defaults.

    Args:
        env: Vectorised WingspanEnv.
        features_dim: Output dimension of WingspanFeaturesExtractor.

    Returns:
        Configured MaskablePPO instance ready for .learn().
    """
    raise NotImplementedError("S4.3 — implement in Sprint 4")
