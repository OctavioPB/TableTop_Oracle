"""Pytest fixtures shared across all test modules."""

from __future__ import annotations

import random

import numpy as np
import pytest

GLOBAL_SEED = 42


@pytest.fixture(autouse=True)
def fix_random_seeds() -> None:
    """Pin random seeds for reproducibility in every test."""
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    try:
        import torch
        torch.manual_seed(GLOBAL_SEED)
    except ImportError:
        pass
