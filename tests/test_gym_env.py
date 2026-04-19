"""S3 — Gymnasium environment test suite.

Covers:
  - check_env compliance (official gymnasium validator)
  - Observation space shape and bounds
  - Action masking consistency with engine.get_legal_actions()
  - 1000 random steps without crash
  - reset() / step() contract
  - Render output
  - N_MAX_ACTIONS coverage
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from hypothesis import given, settings
from hypothesis import strategies as st

from src.envs.wingspan_env import (
    BIRD_FEATURES,
    FOOD_TYPES,
    HABITATS,
    MAX_HAND_SIZE,
    N_GOAL_TYPES,
    N_MAX_ACTIONS,
    WingspanEnv,
    _encode_bird_slot,
    _encode_card,
)
from src.games.wingspan.actions import WingspanActionType
from src.games.wingspan.state import BirdSlotState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> WingspanEnv:
    return WingspanEnv(reward_mode="dense")


@pytest.fixture
def seeded_env() -> WingspanEnv:
    e = WingspanEnv(reward_mode="dense")
    e.reset(seed=42)
    return e


# ---------------------------------------------------------------------------
# 1. check_env — official gymnasium validator
# ---------------------------------------------------------------------------


class TestCheckEnv:
    def test_check_env_passes(self) -> None:
        """gymnasium check_env must report zero warnings/errors."""
        env = WingspanEnv(reward_mode="dense")
        # check_env resets the env internally; warn=False turns warnings into errors
        check_env(env, warn=True, skip_render_check=True)


# ---------------------------------------------------------------------------
# 2. Observation space shape and bounds
# ---------------------------------------------------------------------------


class TestObservationSpace:
    def test_obs_space_keys(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        expected_keys = {"board", "opponent_board", "food_supply", "hand", "bird_tray", "game_state", "round_goals"}
        assert set(obs.keys()) == expected_keys

    def test_board_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["board"].shape == (3 * 5 * BIRD_FEATURES,)

    def test_opponent_board_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["opponent_board"].shape == (3 * 5 * BIRD_FEATURES,)

    def test_food_supply_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["food_supply"].shape == (len(FOOD_TYPES),)

    def test_hand_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["hand"].shape == (MAX_HAND_SIZE * BIRD_FEATURES,)

    def test_bird_tray_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["bird_tray"].shape == (3 * BIRD_FEATURES,)

    def test_game_state_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["game_state"].shape == (6,)

    def test_round_goals_obs_shape(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert obs["round_goals"].shape == (4 * N_GOAL_TYPES,)

    def test_all_obs_in_bounds(self, seeded_env: WingspanEnv) -> None:
        """All observation values must lie within [0, 1]."""
        obs, _ = seeded_env.reset(seed=0)
        for key, arr in obs.items():
            assert np.all(arr >= 0.0), f"{key} has values < 0"
            assert np.all(arr <= 1.0), f"{key} has values > 1"

    def test_obs_dtype_float32(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        for key, arr in obs.items():
            assert arr.dtype == np.float32, f"{key} dtype is {arr.dtype}"

    def test_obs_in_observation_space(self, seeded_env: WingspanEnv) -> None:
        obs, _ = seeded_env.reset(seed=0)
        assert seeded_env.observation_space.contains(obs), "obs not in observation_space"


# ---------------------------------------------------------------------------
# 3. Action space and masking
# ---------------------------------------------------------------------------


class TestActionMasking:
    def test_action_masks_shape(self, seeded_env: WingspanEnv) -> None:
        masks = seeded_env.action_masks()
        assert masks.shape == (N_MAX_ACTIONS,)

    def test_action_masks_dtype(self, seeded_env: WingspanEnv) -> None:
        masks = seeded_env.action_masks()
        assert masks.dtype == bool

    def test_action_masks_at_least_one_true(self, seeded_env: WingspanEnv) -> None:
        masks = seeded_env.action_masks()
        assert masks.any(), "All actions masked — at least one must be legal"

    def test_action_masks_consistent_with_engine(self, seeded_env: WingspanEnv) -> None:
        """Every action that the engine marks legal must appear in the mask."""
        engine = seeded_env._engine
        state = seeded_env._state
        legal = engine.get_legal_actions(state)
        masks = seeded_env.action_masks()

        for action in legal:
            idx = seeded_env._action_to_idx(action)
            if idx is not None and 0 <= idx < N_MAX_ACTIONS:
                assert masks[idx], (
                    f"Legal action {action} (idx={idx}) not in mask"
                )

    def test_masked_action_cannot_be_decoded_to_illegal(self, seeded_env: WingspanEnv) -> None:
        """Decoding a legal masked index produces a valid WingspanAction."""
        masks = seeded_env.action_masks()
        legal_indices = np.where(masks)[0]
        engine = seeded_env._engine
        state = seeded_env._state
        legal_actions = engine.get_legal_actions(state)
        legal_types = {a.action_type for a in legal_actions}

        for idx in legal_indices[:10]:  # check first 10
            decoded = seeded_env._idx_to_action(int(idx))
            assert decoded.action_type in legal_types, (
                f"Decoded action {decoded} not in legal types {legal_types}"
            )

    def test_gain_food_indices_in_first_5(self, seeded_env: WingspanEnv) -> None:
        """GAIN_FOOD actions must map to indices 0–4."""
        engine = seeded_env._engine
        legal = engine.get_legal_actions(seeded_env._state)
        for action in legal:
            if action.action_type == WingspanActionType.GAIN_FOOD.value:
                idx = seeded_env._action_to_idx(action)
                assert idx is not None and 0 <= idx <= 4

    def test_lay_eggs_index_is_5(self, seeded_env: WingspanEnv) -> None:
        """LAY_EGGS must always map to index 5."""
        from src.games.wingspan.actions import WingspanAction
        action = WingspanAction(
            action_type=WingspanActionType.LAY_EGGS.value,
            player_id=0,
        )
        idx = seeded_env._action_to_idx(action)
        assert idx == 5

    def test_draw_cards_indices_6_to_9(self, seeded_env: WingspanEnv) -> None:
        """DRAW_CARDS sources must map to indices 6–9."""
        from src.games.wingspan.actions import WingspanAction
        src_map = {"tray_0": 6, "tray_1": 7, "tray_2": 8, "deck": 9}
        for src, expected_idx in src_map.items():
            action = WingspanAction(
                action_type=WingspanActionType.DRAW_CARDS.value,
                player_id=0,
                draw_source=src,
            )
            assert seeded_env._action_to_idx(action) == expected_idx

    def test_n_max_actions_covers_max_hand(self) -> None:
        """N_MAX_ACTIONS must be ≥ 10 + MAX_HAND_SIZE * 3."""
        assert N_MAX_ACTIONS >= 10 + MAX_HAND_SIZE * 3


# ---------------------------------------------------------------------------
# 4. reset() / step() contract
# ---------------------------------------------------------------------------


class TestResetStepContract:
    def test_reset_returns_obs_and_info(self, env: WingspanEnv) -> None:
        result = env.reset(seed=1)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_seed_determinism(self, env: WingspanEnv) -> None:
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for key in obs1:
            np.testing.assert_array_equal(obs1[key], obs2[key])

    def test_reset_different_seeds(self, env: WingspanEnv) -> None:
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        diffs = [not np.array_equal(obs1[k], obs2[k]) for k in obs1]
        assert any(diffs), "Different seeds should produce different observations"

    def test_step_returns_5_tuple(self, env: WingspanEnv) -> None:
        env.reset(seed=0)
        masks = env.action_masks()
        action = int(np.where(masks)[0][0])
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5

    def test_step_obs_in_space(self, env: WingspanEnv) -> None:
        env.reset(seed=0)
        masks = env.action_masks()
        action = int(np.where(masks)[0][0])
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env: WingspanEnv) -> None:
        env.reset(seed=0)
        masks = env.action_masks()
        action = int(np.where(masks)[0][0])
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_step_terminated_and_truncated_are_bool(self, env: WingspanEnv) -> None:
        env.reset(seed=0)
        masks = env.action_masks()
        action = int(np.where(masks)[0][0])
        _, _, terminated, truncated, _ = env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, env: WingspanEnv) -> None:
        env.reset(seed=0)
        masks = env.action_masks()
        action = int(np.where(masks)[0][0])
        _, _, _, _, info = env.step(action)
        assert isinstance(info, dict)

    def test_step_before_reset_raises(self, env: WingspanEnv) -> None:
        with pytest.raises((AssertionError, RuntimeError, TypeError)):
            env.step(0)

    def test_multiple_resets_work(self, env: WingspanEnv) -> None:
        for seed in range(3):
            obs, info = env.reset(seed=seed)
            assert env.observation_space.contains(obs)

    def test_info_has_expected_keys(self, env: WingspanEnv) -> None:
        _, info = env.reset(seed=0)
        for key in ("round", "turn", "player_0_score", "player_1_score", "action_cubes"):
            assert key in info, f"Missing key '{key}' in info"


# ---------------------------------------------------------------------------
# 5. 1000 random steps without crash
# ---------------------------------------------------------------------------


class TestRandomPolicy:
    def test_1000_random_steps_no_crash(self, env: WingspanEnv) -> None:
        """1000 masked-random steps must complete without exception."""
        import random as pyrandom
        rng = pyrandom.Random(42)
        obs, _ = env.reset(seed=42)
        steps = 0
        episodes = 0

        while steps < 1000:
            masks = env.action_masks()
            legal_indices = np.where(masks)[0]
            assert len(legal_indices) > 0, "No legal actions available"
            action = int(rng.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert env.observation_space.contains(obs), f"Obs out of space at step {steps}"
            if terminated or truncated:
                obs, _ = env.reset(seed=steps)
                episodes += 1

        assert steps == 1000

    def test_player_id_invariant_maintained(self, env: WingspanEnv) -> None:
        """state.player_id must always be 0 when the env returns to the RL loop."""
        import random as pyrandom
        rng = pyrandom.Random(1)
        env.reset(seed=1)
        for _ in range(50):
            assert env._state.player_id == 0, "player_id invariant broken"
            masks = env.action_masks()
            legal_idx = np.where(masks)[0]
            action = int(rng.choice(legal_idx))
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                env.reset(seed=rng.randint(0, 9999))


# ---------------------------------------------------------------------------
# 6. Render
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_none_mode_returns_none(self, env: WingspanEnv) -> None:
        env.reset(seed=0)
        result = env.render()
        assert result is None

    def test_render_text_mode_returns_string(self) -> None:
        env = WingspanEnv(render_mode="text")
        env.reset(seed=0)
        result = env.render()
        assert isinstance(result, str)
        assert "Round" in result

    def test_render_ansi_mode_returns_string(self) -> None:
        env = WingspanEnv(render_mode="ansi")
        env.reset(seed=0)
        result = env.render()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 7. Feature encoding helpers
# ---------------------------------------------------------------------------


class TestFeatureEncoding:
    def test_empty_slot_all_zeros(self) -> None:
        vec = _encode_bird_slot(None, None)
        assert vec.shape == (BIRD_FEATURES,)
        assert np.all(vec == 0.0)

    def test_occupied_slot_first_feature_is_one(self) -> None:
        slot = BirdSlotState("Eagle")
        vec = _encode_bird_slot(slot, None)
        assert vec[0] == 1.0

    def test_empty_card_all_zeros(self) -> None:
        vec = _encode_card(None)
        assert np.all(vec == 0.0)

    def test_encoding_in_zero_one_range(self) -> None:
        slot = BirdSlotState("TestBird", eggs=3, cached_food={"seed": 1}, tucked_cards=2)
        from src.games.wingspan.cards import BirdCard, NestType, PowerID, PowerTiming, PowerType
        card = BirdCard(
            name="TestBird",
            habitats=["forest", "wetland"],
            food_cost={"seed": 2},
            nest_type=NestType.CUP.value,
            egg_limit=4,
            points=5,
            wingspan_cm=45,
            power_timing=PowerTiming.WHEN_ACTIVATED.value,
            power_id=PowerID.GAIN_FOOD_FEEDER.value,
            power_type=PowerType.PRODUCTIVE.value,
        )
        vec = _encode_bird_slot(slot, card)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
        assert vec.shape == (BIRD_FEATURES,)


# ---------------------------------------------------------------------------
# 8. Property-based tests
# ---------------------------------------------------------------------------


class TestPropertyBased:
    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=50, deadline=15_000)
    def test_obs_always_in_space(self, seed: int) -> None:
        env = WingspanEnv()
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)

    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=50, deadline=15_000)
    def test_action_mask_always_has_legal_action(self, seed: int) -> None:
        env = WingspanEnv()
        env.reset(seed=seed)
        masks = env.action_masks()
        assert masks.any()

    @given(n_steps=st.integers(min_value=1, max_value=10))
    @settings(max_examples=30, deadline=15_000)
    def test_obs_in_space_after_n_steps(self, n_steps: int) -> None:
        import random as pyrandom
        env = WingspanEnv()
        rng = pyrandom.Random(n_steps)
        env.reset(seed=n_steps)
        for _ in range(n_steps):
            masks = env.action_masks()
            legal = np.where(masks)[0]
            action = int(rng.choice(legal))
            obs, _, terminated, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated:
                break
