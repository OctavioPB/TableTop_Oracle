"""Behavioural Cloning pre-trainer for the PPO policy — Sprint 5."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    bc_accuracy: float
    loss_per_epoch: list[float]
    n_transitions: int


class BehavioralCloningTrainer:
    """Pre-trains the actor network via supervised learning on expert demos.

    Cross-entropy loss between expert action and policy prediction.
    The trained actor weights are loaded into MaskablePPO before RL fine-tuning.
    """

    def train(self, demo_buffer, model, n_epochs: int = 50) -> TrainingMetrics:
        raise NotImplementedError("S5.3 — implement in Sprint 5")
