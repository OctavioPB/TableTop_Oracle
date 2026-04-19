"""Behavioural Cloning pre-trainer for the MaskablePPO actor — Sprint 5.

BC trains the policy network (feature extractor + MLP + action head) via
supervised cross-entropy loss on expert demonstrations. The trained weights
are then used as the initialisation point for PPO fine-tuning, providing
a warm start that improves sample efficiency.

Architecture notes:
  - The MaskablePPO MultiInputPolicy exposes extract_features(), mlp_extractor,
    and action_net. BC targets only the actor path (not the value head).
  - Action masks are NOT applied during BC training because expert actions are
    always legal by construction. Masks are re-enabled automatically when
    MaskablePPO takes over for RL fine-tuning.
  - A separate Adam optimizer is created over policy.parameters() so BC
    gradients don't interfere with SB3's internal PPO optimizer state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Summary of one BC training run.

    Attributes:
        bc_accuracy: Fraction of expert actions correctly predicted on the
            full training set after the final epoch.
        val_accuracy: Same metric on the held-out validation set, or -1.0
            if no validation split was used.
        loss_per_epoch: Mean cross-entropy loss recorded after each epoch.
        n_transitions: Total number of training transitions used.
    """

    bc_accuracy: float
    val_accuracy: float
    loss_per_epoch: list[float]
    n_transitions: int


# ---------------------------------------------------------------------------
# BehavioralCloningTrainer
# ---------------------------------------------------------------------------


class BehavioralCloningTrainer:
    """Pre-trains the MaskablePPO actor via supervised learning on expert demos.

    Args:
        model: A MaskablePPO instance (from build_maskable_ppo or loaded).
        device: Torch device string ("cpu" or "cuda").
        learning_rate: Adam learning rate for BC optimisation.
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        learning_rate: float = 1e-3,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._lr = learning_rate

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        demo_buffer: Any,
        n_epochs: int = 50,
        batch_size: int = 64,
        val_split: float = 0.1,
        log_every: int = 10,
    ) -> TrainingMetrics:
        """Train the policy actor on demonstrations via cross-entropy loss.

        Splits the buffer into train / validation, runs n_epochs of minibatch
        gradient descent, and reports accuracy after the final epoch.

        Args:
            demo_buffer: DemonstrationBuffer with (obs, action) transitions.
            n_epochs: Number of full passes over the training set.
            batch_size: Minibatch size for each gradient step.
            val_split: Fraction of transitions to hold out for validation.
                Set to 0.0 to skip validation (val_accuracy will be -1.0).
            log_every: Log loss to INFO every this many epochs.

        Returns:
            TrainingMetrics with accuracy and per-epoch loss history.

        Raises:
            ValueError: If the buffer contains fewer transitions than batch_size.
        """
        from src.imitation.demo_buffer import DemonstrationBuffer

        n_total = len(demo_buffer)
        if n_total < batch_size:
            raise ValueError(
                f"Buffer has only {n_total} transitions, "
                f"which is less than batch_size={batch_size}."
            )

        # --- Split into train / val ---
        indices = list(range(n_total))
        rng_split = np.random.default_rng(seed=0)
        rng_split.shuffle(indices)

        n_val = max(1, int(n_total * val_split)) if val_split > 0 else 0
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        policy = self._model.policy
        policy.to(self._device)
        policy.train()

        optimizer = torch.optim.Adam(policy.parameters(), lr=self._lr)
        loss_per_epoch: list[float] = []
        train_rng = np.random.default_rng(seed=42)

        for epoch in range(n_epochs):
            # Shuffle training indices each epoch
            train_rng.shuffle(train_idx)
            epoch_losses: list[float] = []

            for start in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[start: start + batch_size]
                if len(batch_idx) == 0:
                    continue

                obs_batch, actions_batch = self._gather_batch(demo_buffer, batch_idx)
                logits = self._forward(obs_batch, policy)
                actions_tensor = torch.tensor(
                    actions_batch, dtype=torch.long, device=self._device
                )
                loss = F.cross_entropy(logits, actions_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            loss_per_epoch.append(mean_loss)

            if (epoch + 1) % log_every == 0:
                logger.info(
                    "BC epoch %d/%d — loss=%.4f", epoch + 1, n_epochs, mean_loss
                )

        # --- Final accuracy ---
        policy.eval()
        train_acc = self._compute_accuracy(demo_buffer, train_idx, policy, batch_size)
        val_acc = (
            self._compute_accuracy(demo_buffer, val_idx, policy, batch_size)
            if n_val > 0
            else -1.0
        )

        logger.info(
            "BC training complete: train_acc=%.3f  val_acc=%.3f  "
            "transitions=%d  epochs=%d",
            train_acc, val_acc, len(train_idx), n_epochs,
        )
        return TrainingMetrics(
            bc_accuracy=train_acc,
            val_accuracy=val_acc,
            loss_per_epoch=loss_per_epoch,
            n_transitions=len(train_idx),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        demo_buffer: Any,
        batch_size: int = 256,
    ) -> float:
        """Compute action prediction accuracy over the entire buffer.

        Args:
            demo_buffer: DemonstrationBuffer to evaluate on.
            batch_size: Internal minibatch size for forward passes.

        Returns:
            Fraction of expert actions correctly predicted in [0, 1].
        """
        policy = self._model.policy
        policy.to(self._device)
        policy.eval()
        indices = list(range(len(demo_buffer)))
        return self._compute_accuracy(demo_buffer, indices, policy, batch_size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward(
        self,
        obs_batch: dict[str, np.ndarray],
        policy: Any,
    ) -> torch.Tensor:
        """Forward pass through policy actor path.

        Args:
            obs_batch: Dict of numpy arrays, each (batch_size, feature_dim).
            policy: The MaskablePPO policy (MultiInputPolicy).

        Returns:
            Action logits of shape (batch_size, N_MAX_ACTIONS).
        """
        obs_tensor = {
            k: torch.tensor(v, dtype=torch.float32).to(self._device)
            for k, v in obs_batch.items()
        }
        features = policy.extract_features(obs_tensor, policy.features_extractor)
        latent_pi, _ = policy.mlp_extractor(features)
        return policy.action_net(latent_pi)

    def _gather_batch(
        self,
        buffer: Any,
        indices: list[int],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Collect obs and actions for the given transition indices.

        Args:
            buffer: DemonstrationBuffer.
            indices: List of integer indices into buffer._transitions.

        Returns:
            (obs_batch_dict, actions_array)
        """
        transitions = [buffer._transitions[i] for i in indices]
        obs_batch: dict[str, np.ndarray] = {}
        for key in transitions[0].obs:
            obs_batch[key] = np.stack([t.obs[key] for t in transitions], axis=0)
        actions_batch = np.array([t.action for t in transitions], dtype=np.int64)
        return obs_batch, actions_batch

    @torch.no_grad()
    def _compute_accuracy(
        self,
        buffer: Any,
        indices: list[int],
        policy: Any,
        batch_size: int,
    ) -> float:
        """Compute top-1 accuracy over the given transition indices."""
        if not indices:
            return 0.0

        correct = 0
        total = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            obs_batch, actions_batch = self._gather_batch(buffer, batch_idx)
            logits = self._forward(obs_batch, policy)
            preds = logits.argmax(dim=-1).cpu().numpy()
            correct += int((preds == actions_batch).sum())
            total += len(actions_batch)

        return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Utility: transfer BC weights into a new MaskablePPO instance
# ---------------------------------------------------------------------------


def load_bc_weights_into_ppo(
    bc_model: Any,
    target_model: Any,
) -> None:
    """Copy actor weights from bc_model.policy into target_model.policy.

    Copies: features_extractor, mlp_extractor, action_net.
    Does NOT copy the value head (value_net) — PPO learns that from scratch.

    Args:
        bc_model: MaskablePPO instance whose policy was BC-trained.
        target_model: MaskablePPO instance to receive the BC weights.
    """
    src = bc_model.policy
    dst = target_model.policy

    actor_modules = ["features_extractor", "mlp_extractor", "action_net"]
    for module_name in actor_modules:
        src_mod = getattr(src, module_name, None)
        dst_mod = getattr(dst, module_name, None)
        if src_mod is None or dst_mod is None:
            logger.warning("Module '%s' not found on policy — skipping.", module_name)
            continue
        dst_mod.load_state_dict(src_mod.state_dict())

    logger.info(
        "Copied BC actor weights (%s) from bc_model → target_model.",
        ", ".join(actor_modules),
    )
