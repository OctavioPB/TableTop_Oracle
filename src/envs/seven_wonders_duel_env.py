"""7 Wonders Duel gymnasium environment with action masking — Sprint 7.

Mirrors the WingspanEnv design:
  - Agent always controls player 0.
  - Player 1 takes a random legal action automatically inside step().
  - Turn invariant: state.player_id == 0 at every step() and action_masks() call.
  - check_env() must pass without warnings.

Observation space (flat dict of Box spaces):
  "pyramid"   — (23, CARD_FEATURES)  float32  current age pyramid state
  "player"    — (PLAYER_FEATURES,)   float32  player 0 board
  "opponent"  — (PLAYER_FEATURES,)   float32  player 1 board
  "tokens"    — (N_PROGRESS_TOKENS_AVAILABLE * TOKEN_FEATURES,) float32

Action space:
  Discrete(N_MAX_ACTIONS_7WD)  = Discrete(150)
  Index = card_position_in_deck * 6 + action_type_offset
"""

from __future__ import annotations

import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.games.seven_wonders_duel.actions import (
    N_ACTION_TYPES_PER_CARD,
    N_MAX_ACTIONS_7WD,
    SWDAction,
    SWDActionType,
)
from src.games.seven_wonders_duel.cards import (
    ALL_RESOURCES,
    ALL_SCIENCE_SYMBOLS,
)
from src.games.seven_wonders_duel.engine import SWDEngine
from src.games.seven_wonders_duel.rewards import RewardMode, compute_reward
from src.games.seven_wonders_duel.state import SWDState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation dimensions
# ---------------------------------------------------------------------------

N_MAX_CARDS_IN_AGE = 23       # max cards per age deck
N_CARD_TYPES = 7               # raw_material, manufactured, civilian, scientific, commercial, military, guild
N_RESOURCES = len(ALL_RESOURCES)   # 5
N_SCIENCE = len(ALL_SCIENCE_SYMBOLS)  # 6

# Per-card feature vector:
#  accessible(1) + card_type(7) + cost_coins_norm(1) + cost_resources(5) +
#  effect_vp(1) + effect_shields(1) + effect_coins(1) + effect_science_symbols(6)
#  = 23 dims
CARD_FEATURES = 23

# Per-player feature vector:
#  resources(5) + coins_norm(1) + shields_norm(1) + science_symbols(6) +
#  vp_norm(1) + n_built_cards_norm(1) + n_built_wonders_norm(1) + n_progress_tokens(1)
#  = 17 dims
PLAYER_FEATURES = 17

# Progress token features: available(1) + effect_vp(1) + effect_shields(1) +
#  effect_coins(1) + effect_science(1) = 5 dims per token, 5 tokens
N_TOKENS_VISIBLE = 5
TOKEN_FEATURES = 5
TOKENS_OBS_DIM = N_TOKENS_VISIBLE * TOKEN_FEATURES   # 25

CARD_TYPE_LIST = [
    "raw_material", "manufactured", "civilian",
    "scientific", "commercial", "military", "guild",
]


class SevenWondersDuelEnv(gym.Env):
    """Gymnasium environment for 7 Wonders Duel with action masking.

    Args:
        reward_mode: "sparse" or "dense".
        seed: RNG seed for the engine.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        reward_mode: str = RewardMode.DENSE.value,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._engine = SWDEngine(seed=seed)
        self._reward_mode = reward_mode
        self._seed = seed
        self._state: SWDState | None = None
        self._rng = random.Random(seed)

        self.observation_space = spaces.Dict({
            "pyramid": spaces.Box(
                low=0.0, high=1.0,
                shape=(N_MAX_CARDS_IN_AGE, CARD_FEATURES),
                dtype=np.float32,
            ),
            "player": spaces.Box(
                low=0.0, high=1.0,
                shape=(PLAYER_FEATURES,),
                dtype=np.float32,
            ),
            "opponent": spaces.Box(
                low=0.0, high=1.0,
                shape=(PLAYER_FEATURES,),
                dtype=np.float32,
            ),
            "tokens": spaces.Box(
                low=0.0, high=1.0,
                shape=(TOKENS_OBS_DIM,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Discrete(N_MAX_ACTIONS_7WD)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        engine_seed = seed if seed is not None else self._seed
        self._state = self._engine.reset(seed=engine_seed)
        self._rng = random.Random(engine_seed)
        return self._get_obs(), {}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"
        assert self._state.player_id == 0, "Turn invariant violated: player_id != 0"

        swd_action = self._idx_to_action(action, self._state)
        state_before = self._state

        result = self._engine.step(self._state, swd_action)
        self._state = result.new_state

        done = self._engine.is_terminal(self._state)
        winner = self._engine.get_winner(self._state) if done else None
        reward = compute_reward(
            state_before, self._state, player_id=0,
            done=done, winner=winner, mode=self._reward_mode,
        )

        # Let player 1 act if it's their turn and game isn't over
        while not done and self._state.player_id == 1:
            legal = self._engine.get_legal_actions(self._state)
            if not legal:
                break
            opp_action = self._rng.choice(legal)
            opp_result = self._engine.step(self._state, opp_action)
            self._state = opp_result.new_state
            done = self._engine.is_terminal(self._state)
            if done:
                winner = self._engine.get_winner(self._state)
                reward += compute_reward(
                    opp_result.new_state, self._state, player_id=0,
                    done=done, winner=winner, mode=self._reward_mode,
                )

        # Ensure turn invariant
        if not done and self._state.player_id != 0:
            self._state = self._state.model_copy(update={"player_id": 0})

        obs = self._get_obs()
        info: dict[str, Any] = {"winner": winner}
        return obs, float(reward), done, False, info

    def action_masks(self) -> np.ndarray:
        """Return binary mask of shape (N_MAX_ACTIONS_7WD,).

        1 = legal action, 0 = illegal.
        """
        assert self._state is not None
        assert self._state.player_id == 0

        mask = np.zeros(N_MAX_ACTIONS_7WD, dtype=bool)
        legal = self._engine.get_legal_actions(self._state)
        for action in legal:
            idx = self._action_to_idx(action)
            if 0 <= idx < N_MAX_ACTIONS_7WD:
                mask[idx] = True
        return mask

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Index ↔ Action conversion
    # ------------------------------------------------------------------

    def _action_to_idx(self, action: SWDAction) -> int:
        """Convert SWDAction to integer index in [0, N_MAX_ACTIONS_7WD)."""
        if not self._state or not self._state.age_deck:
            return 0

        deck = self._state.age_deck
        try:
            card_pos = deck.index(action.card_name)
        except ValueError:
            return 0

        if action.action_type == SWDActionType.BUILD_CARD.value:
            offset = 0
        elif action.action_type == SWDActionType.DISCARD_CARD.value:
            offset = 1
        elif action.action_type == SWDActionType.BUILD_WONDER.value:
            offset = 2 + max(0, action.wonder_slot)
        else:
            offset = 0

        idx = card_pos * N_ACTION_TYPES_PER_CARD + offset
        return min(idx, N_MAX_ACTIONS_7WD - 1)

    def _idx_to_action(self, idx: int, state: SWDState) -> SWDAction:
        """Convert integer index to SWDAction.

        Falls back to the first legal DISCARD action if the index is out of range
        or maps to an inaccessible card.
        """
        if not state.age_deck:
            return self._fallback_action(state)

        card_pos = idx // N_ACTION_TYPES_PER_CARD
        offset = idx % N_ACTION_TYPES_PER_CARD

        if card_pos >= len(state.age_deck):
            return self._fallback_action(state)

        card_name = state.age_deck[card_pos]

        if offset == 0:
            action_type = SWDActionType.BUILD_CARD.value
            wonder_slot = -1
        elif offset == 1:
            action_type = SWDActionType.DISCARD_CARD.value
            wonder_slot = -1
        else:
            action_type = SWDActionType.BUILD_WONDER.value
            wonder_slot = offset - 2

        return SWDAction(
            action_type=action_type,
            player_id=state.player_id,
            card_name=card_name,
            wonder_slot=wonder_slot,
        )

    def _fallback_action(self, state: SWDState) -> SWDAction:
        """Return the first legal action as a fallback."""
        legal = self._engine.get_legal_actions(state)
        if legal:
            return legal[0]
        accessible = state.accessible_cards(state.age_deck)
        card = accessible[0] if accessible else (state.age_deck[0] if state.age_deck else "")
        return SWDAction(
            action_type=SWDActionType.DISCARD_CARD.value,
            player_id=state.player_id,
            card_name=card,
        )

    # ------------------------------------------------------------------
    # Observation encoder
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        assert self._state is not None
        state = self._state

        pyramid_obs = self._encode_pyramid(state)
        player_obs = self._encode_player(state, 0)
        opponent_obs = self._encode_player(state, 1)
        tokens_obs = self._encode_tokens(state)

        return {
            "pyramid": np.clip(pyramid_obs, 0.0, 1.0).astype(np.float32),
            "player": np.clip(player_obs, 0.0, 1.0).astype(np.float32),
            "opponent": np.clip(opponent_obs, 0.0, 1.0).astype(np.float32),
            "tokens": np.clip(tokens_obs, 0.0, 1.0).astype(np.float32),
        }

    def _encode_pyramid(self, state: SWDState) -> np.ndarray:
        """Encode the current age pyramid as (N_MAX_CARDS_IN_AGE, CARD_FEATURES)."""
        obs = np.zeros((N_MAX_CARDS_IN_AGE, CARD_FEATURES), dtype=np.float32)
        accessible_set = set(state.accessible_cards(state.age_deck))

        for pos, card_name in enumerate(state.age_deck[:N_MAX_CARDS_IN_AGE]):
            if card_name in state.taken_cards:
                continue
            card = self._engine._card_catalog.get(card_name)
            if card is None:
                continue

            f = np.zeros(CARD_FEATURES, dtype=np.float32)
            # accessible
            f[0] = 1.0 if card_name in accessible_set else 0.0
            # card type one-hot (dims 1..7)
            if card.card_type in CARD_TYPE_LIST:
                f[1 + CARD_TYPE_LIST.index(card.card_type)] = 1.0
            # cost coins (normalised to max 10)
            f[8] = min(card.cost_coins / 10.0, 1.0)
            # cost resources (dims 9..13)
            for ri, res in enumerate(ALL_RESOURCES):
                f[9 + ri] = min(card.cost_resources.get(res, 0) / 3.0, 1.0)
            # effect VP (dim 14)
            f[14] = min(card.gives_vp() / 10.0, 1.0)
            # effect shields (dim 15)
            f[15] = min(card.gives_shields() / 3.0, 1.0)
            # effect coins (dim 16)
            f[16] = min(card.coin_effect() / 10.0, 1.0)
            # science symbols (dims 17..22)
            for si, sym in enumerate(ALL_SCIENCE_SYMBOLS):
                f[17 + si] = 1.0 if card.effect.get(sym, 0) > 0 else 0.0

            obs[pos] = f

        return obs

    def _encode_player(self, state: SWDState, pid: int) -> np.ndarray:
        """Encode a player board as (PLAYER_FEATURES,)."""
        board = state.get_board(pid)
        f = np.zeros(PLAYER_FEATURES, dtype=np.float32)
        # resources (dims 0..4)
        for ri, res in enumerate(ALL_RESOURCES):
            f[ri] = min(board.resources.get(res, 0) / 4.0, 1.0)
        # coins (dim 5)
        f[5] = min(board.coins / 50.0, 1.0)
        # shields (dim 6)
        f[6] = min(board.shields / 9.0, 1.0)
        # science symbols (dims 7..12)
        for si, sym in enumerate(ALL_SCIENCE_SYMBOLS):
            f[7 + si] = min(board.science_symbols.get(sym, 0) / 2.0, 1.0)
        # VP from cards + wonders (dim 13)
        f[13] = min((board.vp_from_cards + board.vp_from_wonders) / 50.0, 1.0)
        # n built cards (dim 14)
        f[14] = min(len(board.built_cards) / 23.0, 1.0)
        # n built wonders (dim 15)
        f[15] = min(board.n_built_wonders() / 4.0, 1.0)
        # n progress tokens (dim 16)
        f[16] = min(len(board.progress_tokens) / 5.0, 1.0)
        return f

    def _encode_tokens(self, state: SWDState) -> np.ndarray:
        """Encode available progress tokens as (TOKENS_OBS_DIM,)."""
        f = np.zeros(TOKENS_OBS_DIM, dtype=np.float32)
        tokens_dict = {t.name: t for t in self._engine._progress_tokens}

        for i, token_name in enumerate(state.progress_tokens_available[:N_TOKENS_VISIBLE]):
            token = tokens_dict.get(token_name)
            if token is None:
                continue
            base = i * TOKEN_FEATURES
            f[base] = 1.0  # available
            f[base + 1] = min(token.effect.get("vp", 0) / 7.0, 1.0)
            f[base + 2] = min(token.effect.get("shields", 0) / 2.0, 1.0)
            f[base + 3] = min(token.effect.get("coins", 0) / 12.0, 1.0)
            f[base + 4] = 1.0 if any(
                k in token.effect for k in ["balance", "law", "gear", "tablet", "compass"]
            ) else 0.0

        return f
