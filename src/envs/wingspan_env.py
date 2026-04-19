"""Wingspan gymnasium environment with action masking — Sprint 3.

Design decisions (documented here per PLAN.md D1, D2):
  D1 — Card representation: discrete feature vectors (28-dim per slot).
       Habitats (3-hot) + nest_type (6-hot) + food/pts/egg_limit scalars +
       power_timing (6-hot) + power_type (6-hot) + slot state (eggs/cache/tuck).
       Rationale: no LLM dependency at runtime, interpretable, fast.

  D2 — Opponent policy: single-agent mode.
       Agent always controls player 0.  Player 1 takes a random legal action
       automatically inside step(). The RL loop never sees player 1's turn.
       Rationale: simplest valid design for Sprint 3/4; self-play deferred to D2.

  Turn invariant maintained by env:
       state.player_id == 0 at the START of every step() and action_masks() call.
       After step(), state.player_id is restored to 0 before returning.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.games.wingspan.actions import WingspanAction, WingspanActionType
from src.games.wingspan.cards import ALL_ROUND_GOALS
from src.games.wingspan.engine import WingspanEngine
from src.games.wingspan.rewards import RewardMode, compute_reward
from src.games.wingspan.state import BirdSlotState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MAX_ACTIONS: int = 80
MAX_HAND_SIZE: int = 20
BIRD_FEATURES: int = 28
N_GOAL_TYPES: int = 8

FOOD_TYPES: list[str] = ["seed", "fruit", "invertebrate", "rodent", "fish"]
HABITATS: list[str] = ["forest", "grassland", "wetland"]
NEST_TYPES: list[str] = ["cup", "platform", "cavity", "ground", "star", "wild"]
POWER_TIMINGS: list[str] = [
    "when_played", "when_activated", "once_between_turns",
    "all_players", "end_of_round", "no_power",
]
POWER_TYPES: list[str] = ["accumulate", "productive", "predator", "flocking", "other", "none"]

# Action index layout
_GAIN_FOOD_START = 0   # 0-4: one per food type
_LAY_EGGS_IDX = 5
_DRAW_START = 6        # 6-9: tray_0, tray_1, tray_2, deck
_PLAY_BIRD_START = 10  # 10 + hand_pos*3 + hab_idx  (max: 10 + 20*3 - 1 = 69)

_DRAW_SOURCES: list[str] = ["tray_0", "tray_1", "tray_2", "deck"]


# ---------------------------------------------------------------------------
# Feature encoding helpers (module-level for performance)
# ---------------------------------------------------------------------------


def _encode_bird_slot(slot: "BirdSlotState | None", card: Any) -> np.ndarray:
    """Return 28-dim feature vector for a bird slot.

    Empty slot → zero vector. Occupied slot → card features + slot state.
    """
    vec = np.zeros(BIRD_FEATURES, dtype=np.float32)
    if slot is None:
        return vec

    vec[0] = 1.0  # occupied

    if card is not None:
        # habitats: multi-hot (indices 1-3)
        for i, h in enumerate(HABITATS):
            if h in card.habitats:
                vec[1 + i] = 1.0

        # nest_type: one-hot (indices 4-9)
        nt = getattr(card, "nest_type", "cup")
        if nt in NEST_TYPES:
            vec[4 + NEST_TYPES.index(nt)] = 1.0

        # food cost total, points, egg_limit (indices 10-12)
        total_cost = sum(card.food_cost.values()) if card.food_cost else 0
        vec[10] = min(total_cost, 8) / 8.0
        vec[11] = min(card.points, 9) / 9.0
        vec[12] = card.egg_limit / 6.0

        # power_timing: one-hot (indices 13-18)
        pt = getattr(card, "power_timing", "no_power")
        if pt in POWER_TIMINGS:
            vec[13 + POWER_TIMINGS.index(pt)] = 1.0

        # power_type: one-hot (indices 19-24)
        ptype = getattr(card, "power_type", "none")
        if ptype in POWER_TYPES:
            vec[19 + POWER_TYPES.index(ptype)] = 1.0

    # Slot state (indices 25-27)
    vec[25] = min(slot.eggs, 6) / 6.0
    vec[26] = min(sum(slot.cached_food.values()), 10) / 10.0
    vec[27] = min(slot.tucked_cards, 10) / 10.0

    return vec


def _encode_card(card: Any) -> np.ndarray:
    """Encode a card not yet placed (hand / tray) — slot state fields = 0."""
    if card is None:
        return np.zeros(BIRD_FEATURES, dtype=np.float32)
    dummy_slot = BirdSlotState(bird_name=card.name)
    return _encode_bird_slot(dummy_slot, card)


# ---------------------------------------------------------------------------
# WingspanEnv
# ---------------------------------------------------------------------------


class WingspanEnv(gym.Env):
    """Gymnasium-compatible Wingspan environment for MaskablePPO training.

    Action space: Discrete(N_MAX_ACTIONS = 80)
      0-4   GAIN_FOOD  (food type index into FOOD_TYPES)
      5     LAY_EGGS
      6-9   DRAW_CARDS (tray_0/1/2/deck)
      10+   PLAY_BIRD  (hand_pos * 3 + habitat_idx, one-hot over hand × habitat)

    Observation space: Dict of float32 boxes, all normalised to [0, 1].
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
        self._done: bool = True
        self._rng: random.Random = random.Random()

        # Build spaces
        board_shape = (3 * 5 * BIRD_FEATURES,)
        self.observation_space = spaces.Dict({
            "board":         spaces.Box(0.0, 1.0, shape=board_shape,                  dtype=np.float32),
            "opponent_board": spaces.Box(0.0, 1.0, shape=board_shape,                 dtype=np.float32),
            "food_supply":   spaces.Box(0.0, 1.0, shape=(len(FOOD_TYPES),),           dtype=np.float32),
            "hand":          spaces.Box(0.0, 1.0, shape=(MAX_HAND_SIZE * BIRD_FEATURES,), dtype=np.float32),
            "bird_tray":     spaces.Box(0.0, 1.0, shape=(3 * BIRD_FEATURES,),         dtype=np.float32),
            "game_state":    spaces.Box(0.0, 1.0, shape=(6,),                         dtype=np.float32),
            "round_goals":   spaces.Box(0.0, 1.0, shape=(4 * N_GOAL_TYPES,),          dtype=np.float32),
        })
        self.action_space = spaces.Discrete(N_MAX_ACTIONS)

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = random.Random(seed)
        self._state = self._engine.reset(seed=seed)
        # Ensure invariant: state.player_id == 0
        self._state = self._state.model_copy(update={"player_id": 0})
        self._done = False
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()."
        assert not self._done, "Episode is over. Call reset()."

        # ---- Player 0 (agent) acts ----------------------------------------
        wingspan_action = self._idx_to_action(int(action), player_id=0)
        state_before = self._state
        result0 = self._engine.step(self._state, wingspan_action)
        self._state = result0.new_state

        terminated = self._engine.is_terminal(self._state)

        reward = compute_reward(
            state_before,
            wingspan_action,
            self._state,
            terminated,
            reward_mode=self.reward_mode,
            player_id=0,
            engine=self._engine,
        )

        # ---- Player 1 (opponent) auto-acts --------------------------------
        if not terminated:
            # Switch to player 1 for their turn
            self._state = self._state.model_copy(update={"player_id": 1})
            legal1 = self._engine.get_legal_actions(self._state)
            opp_action = self._rng.choice(legal1)
            result1 = self._engine.step(self._state, opp_action)
            self._state = result1.new_state
            terminated = self._engine.is_terminal(self._state)

            # Restore invariant: player_id == 0 for next step
            # (_end_round already sets player_id=0; cover the non-round-end case)
            if not terminated and self._state.player_id != 0:
                self._state = self._state.model_copy(update={"player_id": 0})

        if terminated:
            self._done = True

        return self._get_obs(), float(reward), terminated, False, self._get_info()

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of legal actions — required by MaskablePPO.

        Indices that correspond to legal WingspanActions are True; all others False.
        Guaranteed to have at least one True entry in non-terminal states.
        """
        mask = np.zeros(N_MAX_ACTIONS, dtype=bool)
        if self._state is None or self._done:
            mask[0] = True  # dummy — shouldn't be sampled
            return mask

        legal = self._engine.get_legal_actions(self._state)
        for la in legal:
            idx = self._action_to_idx(la)
            if idx is not None and 0 <= idx < N_MAX_ACTIONS:
                mask[idx] = True

        if not mask.any():
            mask[0] = True  # safety fallback
        return mask

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        if self._state is None:
            return "Environment not initialised — call reset() first."

        board = self._state.get_board(0)
        opp = self._state.get_board(1)
        scores = self._engine.compute_scores(self._state)
        lines = [
            f"=== Wingspan — Round {self._state.round} / Turn {self._state.turn} ===",
            f"P0  birds={board.total_birds()}  eggs={board.total_eggs()}"
            f"  food={board.total_food()}  hand={len(board.hand)}"
            f"  cubes={board.action_cubes}  score={scores[0]}",
            f"P1  birds={opp.total_birds()}  eggs={opp.total_eggs()}"
            f"  food={opp.total_food()}  hand={len(opp.hand)}"
            f"  cubes={opp.action_cubes}  score={scores[1]}",
            f"Feeder: {self._state.bird_feeder}",
            f"Tray: {self._state.bird_tray}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        state = self._state
        catalog = self._engine._catalog
        board = state.get_board(0)
        opp_board = state.get_board(1)

        # --- Current player board (3 × 5 × BIRD_FEATURES) ---------------
        board_vecs: list[np.ndarray] = []
        for hab in HABITATS:
            for slot in board.get_habitat(hab):
                card = catalog.get(slot.bird_name) if slot else None
                board_vecs.append(_encode_bird_slot(slot, card))
        board_obs = np.concatenate(board_vecs, dtype=np.float32)

        # --- Opponent board -----------------------------------------------
        opp_vecs: list[np.ndarray] = []
        for hab in HABITATS:
            for slot in opp_board.get_habitat(hab):
                card = catalog.get(slot.bird_name) if slot else None
                opp_vecs.append(_encode_bird_slot(slot, card))
        opp_obs = np.concatenate(opp_vecs, dtype=np.float32)

        # --- Food supply (normalised) ------------------------------------
        food_obs = np.array(
            [min(board.food_supply.get(f, 0), 20) / 20.0 for f in FOOD_TYPES],
            dtype=np.float32,
        )

        # --- Hand (MAX_HAND_SIZE × BIRD_FEATURES) ------------------------
        hand_vecs: list[np.ndarray] = []
        for i in range(MAX_HAND_SIZE):
            if i < len(board.hand):
                card = catalog.get(board.hand[i])
                hand_vecs.append(_encode_card(card))
            else:
                hand_vecs.append(np.zeros(BIRD_FEATURES, dtype=np.float32))
        hand_obs = np.concatenate(hand_vecs, dtype=np.float32)

        # --- Bird tray (3 × BIRD_FEATURES) --------------------------------
        tray_vecs: list[np.ndarray] = []
        for i in range(3):
            if i < len(state.bird_tray):
                card = catalog.get(state.bird_tray[i])
                tray_vecs.append(_encode_card(card))
            else:
                tray_vecs.append(np.zeros(BIRD_FEATURES, dtype=np.float32))
        tray_obs = np.concatenate(tray_vecs, dtype=np.float32)

        # --- Game state scalars (6 values) --------------------------------
        game_state_obs = np.clip(
            np.array([
                state.round / 4.0,
                board.action_cubes / 8.0,
                board.total_birds() / 15.0,
                min(board.total_eggs(), 60) / 60.0,
                len(state.bird_feeder) / 5.0,
                len(state.bird_tray) / 3.0,
            ], dtype=np.float32),
            0.0, 1.0,
        )

        # --- Round goals (4 × N_GOAL_TYPES) -------------------------------
        goal_obs = np.zeros(4 * N_GOAL_TYPES, dtype=np.float32)
        for i, goal in enumerate(state.round_end_goals[:4]):
            if goal in ALL_ROUND_GOALS:
                g_idx = ALL_ROUND_GOALS.index(goal)
                goal_obs[i * N_GOAL_TYPES + g_idx] = 1.0

        return {
            "board":          board_obs,
            "opponent_board": opp_obs,
            "food_supply":    food_obs,
            "hand":           hand_obs,
            "bird_tray":      tray_obs,
            "game_state":     game_state_obs,
            "round_goals":    goal_obs,
        }

    def _get_info(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        scores = self._engine.compute_scores(self._state)
        board = self._state.get_board(0)
        return {
            "round":           self._state.round,
            "turn":            self._state.turn,
            "player_0_score":  scores.get(0, 0),
            "player_1_score":  scores.get(1, 0),
            "action_cubes":    board.action_cubes,
            "hand_size":       len(board.hand),
        }

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def _action_to_idx(self, action: WingspanAction) -> int | None:
        """Map a WingspanAction to its integer index, or None if out of range."""
        atype = action.action_type

        if atype == WingspanActionType.GAIN_FOOD.value:
            fc = action.food_choice
            if fc in FOOD_TYPES:
                return _GAIN_FOOD_START + FOOD_TYPES.index(fc)
            return None

        if atype == WingspanActionType.LAY_EGGS.value:
            return _LAY_EGGS_IDX

        if atype == WingspanActionType.DRAW_CARDS.value:
            if action.draw_source in _DRAW_SOURCES:
                return _DRAW_START + _DRAW_SOURCES.index(action.draw_source)
            return None

        if atype == WingspanActionType.PLAY_BIRD.value:
            board = self._state.get_board(self._state.player_id)
            if action.card_name not in board.hand:
                return None
            hand_pos = board.hand.index(action.card_name)
            if hand_pos >= MAX_HAND_SIZE:
                return None
            if action.target_habitat not in HABITATS:
                return None
            hab_idx = HABITATS.index(action.target_habitat)
            return _PLAY_BIRD_START + hand_pos * 3 + hab_idx

        return None

    def _idx_to_action(self, idx: int, player_id: int = 0) -> WingspanAction:
        """Decode integer action index to WingspanAction.

        Looks up the corresponding legal action (which has pre-computed
        food_payment and egg_payment for PLAY_BIRD). Falls back to the
        first legal action if the index is not found in the current legal set.
        """
        legal = self._engine.get_legal_actions(self._state)

        # Fast lookup: find matching legal action
        for la in legal:
            if self._action_to_idx(la) == idx:
                return la

        # Index not in legal set — take first legal action (shouldn't happen
        # if action_masks() was used correctly)
        logger.warning(
            "Action index %d not in legal action set — using fallback. "
            "Ensure action_masks() is applied before sampling.",
            idx,
        )
        return legal[0] if legal else WingspanAction(
            action_type=WingspanActionType.GAIN_FOOD.value,
            player_id=player_id,
            food_choice=FOOD_TYPES[0],
        )
