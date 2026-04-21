"""Splendor gymnasium environment with action masking.

Design mirrors WingspanEnv / SevenWondersDuelEnv:
  - Agent always controls player 0.
  - Player 1 takes a random legal action inside step().
  - Turn invariant: state.player_id == 0 at every step() and action_masks() call.
  - check_env() must pass without warnings.

Observation space (Dict of Box):
  "bank"       (6,)     gem counts in bank, normalised by supply max
  "board"      (132,)   3 tiers × 4 slots × 11 card features, flattened
  "deck_sizes" (3,)     remaining cards per tier, normalised
  "nobles"     (21,)    3 nobles × 7 features, flattened
  "player"     (46,)    player 0 board features
  "opponent"   (46,)    player 1 board features
  "game_state" (1,)     turn normalised

Action space: Discrete(45) with action_masks().
"""

from __future__ import annotations

import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.games.splendor.actions import (
    N_MAX_ACTIONS_SPLENDOR,
    SplendorAction,
    SplendorActionType,
    action_to_index,
    index_to_action_params,
)
from src.games.splendor.cards import (
    CARDS_BY_ID,
    GEM_SUPPLY_2P,
    GEM_TYPES,
    GOLD,
    N_BOARD_SLOTS,
    NOBLES_BY_ID,
    SplendorCard,
)
from src.games.splendor.engine import SplendorEngine
from src.games.splendor.rewards import RewardMode, compute_reward
from src.games.splendor.state import SplendorPlayerBoard, SplendorState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

BANK_FEATURES = 6            # 5 gem types + gold
CARD_FEATURES = 12           # occupied(1)+vp(1)+bonus_onehot(5)+costs(5) = 12
BOARD_FEATURES = 3 * N_BOARD_SLOTS * CARD_FEATURES   # 144
DECK_FEATURES = 3            # one per tier
NOBLE_FEATURES = 7           # per noble
NOBLES_TOTAL = 3 * NOBLE_FEATURES   # 21
PLAYER_FEATURES = 49         # gems(6)+bonus(5)+vp(1)+reserved(3×12)+cards(1)
GAME_STATE_FEATURES = 1

_BANK_MAX = max(GEM_SUPPLY_2P.values())
_GEM_IDX = {g: i for i, g in enumerate(GEM_TYPES)}
_TOTAL_CARDS_BY_TIER = {1: 25, 2: 15, 3: 15}
_MAX_VP = 15
_MAX_BONUS = 7
_MAX_GEMS = 10
_MAX_TURN = 60


def _encode_card(card_id: str | None) -> np.ndarray:
    """11-dim feature vector for a board slot."""
    vec = np.zeros(CARD_FEATURES, dtype=np.float32)
    if card_id is None:
        return vec
    card = CARDS_BY_ID[card_id]
    vec[0] = 1.0                                          # occupied
    vec[1] = float(card.vp) / _MAX_VP                    # vp norm
    bonus_idx = _GEM_IDX.get(card.bonus_color, -1)
    if bonus_idx >= 0:
        vec[2 + bonus_idx] = 1.0                          # bonus one-hot (2-6)
    for j, gem in enumerate(GEM_TYPES):
        vec[7 + j] = float(card.cost.get(gem, 0)) / 9.0  # cost norm (7-11)
    return vec


def _encode_noble(noble_id: str | None) -> np.ndarray:
    """7-dim feature vector for a noble slot."""
    vec = np.zeros(NOBLE_FEATURES, dtype=np.float32)
    if noble_id is None:
        return vec
    noble = NOBLES_BY_ID[noble_id]
    vec[0] = 1.0
    vec[1] = float(noble.vp) / 5.0
    for j, gem in enumerate(GEM_TYPES):
        vec[2 + j] = float(noble.requirements.get(gem, 0)) / _MAX_BONUS
    return vec


def _encode_player(board: SplendorPlayerBoard, reserved_board: list[str | None]) -> np.ndarray:
    """46-dim feature vector for one player.

    Layout:
      0-4   gems (5 types) / MAX_GEMS
      5     gold / MAX_GEMS
      6-10  bonus per color / MAX_BONUS
      11    vp / MAX_VP
      12-44 reserved cards (3 × 11 dims each)
      45    total cards owned / 40
    """
    vec = np.zeros(PLAYER_FEATURES, dtype=np.float32)
    for j, gem in enumerate(GEM_TYPES):
        vec[j] = float(board.gems.get(gem, 0)) / _MAX_GEMS
    vec[5] = float(board.gems.get(GOLD, 0)) / _MAX_GEMS
    bonus = board.bonus()
    for j, gem in enumerate(GEM_TYPES):
        vec[6 + j] = float(bonus.get(gem, 0)) / _MAX_BONUS
    vec[11] = float(board.vp()) / _MAX_VP
    for r_idx, card_id in enumerate(reserved_board):
        card_vec = _encode_card(card_id)
        vec[12 + r_idx * CARD_FEATURES: 12 + (r_idx + 1) * CARD_FEATURES] = card_vec
    vec[48] = float(len(board.cards_owned)) / 40.0
    return vec


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SplendorEnv(gym.Env):
    """Splendor gymnasium environment for 2 players (agent = P0)."""

    metadata: dict[str, Any] = {"render_modes": []}

    MAX_STEPS = 300  # truncation limit — prevents infinite loops in degenerate states

    def __init__(self, reward_mode: str = "dense") -> None:
        super().__init__()
        self._engine = SplendorEngine()
        self._reward_mode = RewardMode(reward_mode)
        self._state: SplendorState | None = None
        self._rng_seed: int | None = None
        self._step_count: int = 0

        self.action_space = spaces.Discrete(N_MAX_ACTIONS_SPLENDOR)
        self.observation_space = spaces.Dict({
            "bank":       spaces.Box(0.0, 1.0, (BANK_FEATURES,),       dtype=np.float32),
            "board":      spaces.Box(0.0, 1.0, (BOARD_FEATURES,),      dtype=np.float32),
            "deck_sizes": spaces.Box(0.0, 1.0, (DECK_FEATURES,),       dtype=np.float32),
            "nobles":     spaces.Box(0.0, 1.0, (NOBLES_TOTAL,),        dtype=np.float32),
            "player":     spaces.Box(0.0, 1.0, (PLAYER_FEATURES,),     dtype=np.float32),
            "opponent":   spaces.Box(0.0, 1.0, (PLAYER_FEATURES,),     dtype=np.float32),
            "game_state": spaces.Box(0.0, 1.0, (GAME_STATE_FEATURES,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng_seed = seed
        self._state = self._engine.reset(seed=seed)
        self._step_count = 0
        assert self._state.player_id == 0
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"
        assert self._state.player_id == 0, "Turn invariant violated"

        prev_state = self._state
        self._step_count += 1

        # --- Agent (P0) step ---
        p0_action = self._idx_to_action(action, self._state)
        result = self._engine.step(self._state, p0_action)
        self._state = result.new_state  # type: ignore[assignment]

        reward = float(compute_reward(prev_state, result, self._reward_mode))
        terminated = self._engine.is_terminal(self._state)

        # --- Opponent (P1) step ---
        if not terminated:
            assert self._state.player_id == 1
            self._state = self._run_opponent()
            if self._state.player_id != 0:
                self._state = self._state.model_copy(update={"player_id": 0})
            terminated = self._engine.is_terminal(self._state)

        truncated = not terminated and self._step_count >= self.MAX_STEPS

        info: dict[str, Any] = {}
        if terminated or truncated:
            p0_vp = self._state.get_board(0).vp()
            p1_vp = self._state.get_board(1).vp()
            info["player_0_score"] = p0_vp
            info["player_1_score"] = p1_vp
            info["winner"] = self._state.winner if terminated else (0 if p0_vp >= p1_vp else 1)

        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask over Discrete(45)."""
        assert self._state is not None
        mask = np.zeros(N_MAX_ACTIONS_SPLENDOR, dtype=bool)
        for a in self._engine.get_legal_actions(self._state):
            try:
                idx = action_to_index(a)
                if 0 <= idx < N_MAX_ACTIONS_SPLENDOR:
                    mask[idx] = True
            except (ValueError, KeyError):
                pass
        # Safety: ensure at least one action is valid
        if not mask.any():
            mask[0] = True
        return mask

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        state = self._state
        assert state is not None

        # Bank
        bank_vec = np.zeros(BANK_FEATURES, dtype=np.float32)
        for j, gem in enumerate(GEM_TYPES):
            bank_vec[j] = np.clip(float(state.bank.get(gem, 0)) / _BANK_MAX, 0.0, 1.0)
        bank_vec[5] = np.clip(float(state.bank.get(GOLD, 0)) / 5.0, 0.0, 1.0)

        # Board
        board_vec = np.zeros(BOARD_FEATURES, dtype=np.float32)
        for tier_idx in range(3):
            for slot_idx in range(N_BOARD_SLOTS):
                card_id = state.board[tier_idx][slot_idx]
                start = (tier_idx * N_BOARD_SLOTS + slot_idx) * CARD_FEATURES
                board_vec[start: start + CARD_FEATURES] = _encode_card(card_id)

        # Deck sizes
        deck_vec = np.array([
            np.clip(len(state.decks[i]) / _TOTAL_CARDS_BY_TIER[i + 1], 0.0, 1.0)
            for i in range(3)
        ], dtype=np.float32)

        # Nobles (pad to exactly 3)
        noble_vec = np.zeros(NOBLES_TOTAL, dtype=np.float32)
        for ni, noble_id in enumerate(state.nobles_available[:3]):
            noble_vec[ni * NOBLE_FEATURES: (ni + 1) * NOBLE_FEATURES] = _encode_noble(noble_id)

        # Players
        p0 = state.get_board(0)
        p1 = state.get_board(1)
        player_vec   = _encode_player(p0, p0.reserved)
        opponent_vec = _encode_player(p1, p1.reserved)

        # Game state
        game_vec = np.array(
            [np.clip(float(state.turn) / _MAX_TURN, 0.0, 1.0)],
            dtype=np.float32,
        )

        return {
            "bank":       bank_vec,
            "board":      board_vec,
            "deck_sizes": deck_vec,
            "nobles":     noble_vec,
            "player":     player_vec,
            "opponent":   opponent_vec,
            "game_state": game_vec,
        }

    def _run_opponent(self) -> SplendorState:
        """Run P1 random legal action using seeded np_random for determinism.

        If no legal actions exist (degenerate late-game state), skip the turn
        — the MAX_STEPS truncation in step() will end the episode.
        """
        legal = self._engine.get_legal_actions(self._state)
        if not legal:
            return self._state.model_copy(update={"player_id": 0})
        idx = int(self.np_random.integers(0, len(legal)))
        result = self._engine.step(self._state, legal[idx])
        return result.new_state  # type: ignore[return-value]

    def _idx_to_action(self, idx: int, state: SplendorState) -> SplendorAction:
        """Reconstruct a fully-specified SplendorAction from an index.

        Board actions need card_id and payment filled in from current state.
        """
        params = index_to_action_params(idx)
        params["player_id"] = state.player_id
        t = params["action_type"]

        if t == SplendorActionType.RESERVE_BOARD:
            tier_idx = params["tier"]
            slot_idx = params["slot"]
            card_id = state.board[tier_idx][slot_idx]
            params["card_id"] = card_id or ""

        elif t == SplendorActionType.BUY_BOARD:
            tier_idx = params["tier"]
            slot_idx = params["slot"]
            card_id = state.board[tier_idx][slot_idx]
            if card_id:
                board = state.get_board(state.player_id)
                params["card_id"] = card_id
                params["payment"] = board.payment_for(CARDS_BY_ID[card_id].cost)
            else:
                params["card_id"] = ""

        elif t == SplendorActionType.BUY_RESERVED:
            rslot = params["reserve_slot"]
            board = state.get_board(state.player_id)
            card_id = board.reserved[rslot]
            if card_id:
                params["card_id"] = card_id
                params["payment"] = board.payment_for(CARDS_BY_ID[card_id].cost)
            else:
                params["card_id"] = ""

        return SplendorAction(**{k: v for k, v in params.items() if k != "action_type"},
                              action_type=params["action_type"])
