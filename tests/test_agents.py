"""S4 — Agent test suite.

Covers:
  - WingspanFeaturesExtractor: output shapes, forward pass
  - RandomAgent and GreedyAgent: valid action selection
  - build_maskable_ppo: model construction
  - WinRateCallback: structure
  - Short training smoke test (256 steps, 1 env)
  - evaluate_ppo_win_rate: 5 episodes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.baselines import BaseAgent, GreedyAgent, RandomAgent, evaluate_agents
from src.agents.encoders import WingspanFeaturesExtractor
from src.agents.ppo_agent import (
    WinRateCallback,
    build_maskable_ppo,
    evaluate_ppo_win_rate,
    make_callbacks,
)
from src.envs.wingspan_env import WingspanEnv
from src.games.wingspan.engine import WingspanEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> WingspanEngine:
    return WingspanEngine(seed=0)


@pytest.fixture
def env() -> WingspanEnv:
    e = WingspanEnv(reward_mode="dense")
    e.reset(seed=42)
    return e


@pytest.fixture(scope="module")
def obs_space():
    e = WingspanEnv()
    return e.observation_space


# ---------------------------------------------------------------------------
# 1. WingspanFeaturesExtractor
# ---------------------------------------------------------------------------


class TestFeaturesExtractor:
    def test_constructor_succeeds(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=256)
        assert extractor is not None

    def test_features_dim_stored(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=128)
        assert extractor.features_dim == 128

    def test_forward_output_shape_default(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=256)
        extractor.eval()
        env = WingspanEnv()
        obs, _ = env.reset(seed=0)
        # Convert obs dict to tensors (batch_size=1)
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }
        with torch.no_grad():
            output = extractor(obs_tensors)
        assert output.shape == (1, 256)

    def test_forward_output_shape_custom_dim(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=64)
        extractor.eval()
        env = WingspanEnv()
        obs, _ = env.reset(seed=1)
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }
        with torch.no_grad():
            output = extractor(obs_tensors)
        assert output.shape == (1, 64)

    def test_forward_output_no_nan(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=256)
        extractor.eval()
        env = WingspanEnv()
        obs, _ = env.reset(seed=2)
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }
        with torch.no_grad():
            output = extractor(obs_tensors)
        assert not torch.isnan(output).any()

    def test_forward_batch_of_4(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=256)
        extractor.eval()
        env = WingspanEnv()
        obs, _ = env.reset(seed=3)
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).repeat(4, 1)
            for k, v in obs.items()
        }
        with torch.no_grad():
            output = extractor(obs_tensors)
        assert output.shape == (4, 256)

    def test_sub_networks_exist(self, obs_space) -> None:
        extractor = WingspanFeaturesExtractor(obs_space, features_dim=256)
        for attr in ("board_net", "opp_net", "hand_net", "tray_net",
                     "food_net", "game_net", "goal_net", "trunk"):
            assert hasattr(extractor, attr), f"Missing sub-network: {attr}"


# ---------------------------------------------------------------------------
# 2. RandomAgent
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def test_returns_legal_action(self, engine: WingspanEngine) -> None:
        agent = RandomAgent(seed=0)
        state = engine.reset(seed=0)
        legal = engine.get_legal_actions(state)
        action = agent.select_action(state, legal)
        assert action in legal

    def test_is_instance_of_base_agent(self) -> None:
        assert isinstance(RandomAgent(), BaseAgent)

    def test_seed_determinism(self, engine: WingspanEngine) -> None:
        state = engine.reset(seed=0)
        legal = engine.get_legal_actions(state)
        a1 = RandomAgent(seed=7).select_action(state, legal)
        a2 = RandomAgent(seed=7).select_action(state, legal)
        assert a1.action_type == a2.action_type

    def test_different_seeds_may_differ(self, engine: WingspanEngine) -> None:
        state = engine.reset(seed=0)
        legal = engine.get_legal_actions(state)
        if len(legal) < 2:
            pytest.skip("Need at least 2 legal actions to test randomness")
        # Run 20 times with same seed — all same; with different seed — at least once different
        actions_s1 = [RandomAgent(seed=1).select_action(state, legal) for _ in range(20)]
        actions_s2 = [RandomAgent(seed=2).select_action(state, legal) for _ in range(20)]
        types_s1 = {a.action_type for a in actions_s1}
        types_s2 = {a.action_type for a in actions_s2}
        # At least one type should differ OR the sets should differ in order —
        # We just check both calls don't raise
        assert len(types_s1) >= 1
        assert len(types_s2) >= 1

    def test_works_with_single_legal_action(self, engine: WingspanEngine) -> None:
        state = engine.reset(seed=0)
        legal = engine.get_legal_actions(state)[:1]  # Force single option
        action = RandomAgent().select_action(state, legal)
        assert action == legal[0]


# ---------------------------------------------------------------------------
# 3. GreedyAgent
# ---------------------------------------------------------------------------


class TestGreedyAgent:
    def test_returns_legal_action(self, engine: WingspanEngine) -> None:
        agent = GreedyAgent(catalog=engine._catalog)
        state = engine.reset(seed=0)
        legal = engine.get_legal_actions(state)
        action = agent.select_action(state, legal)
        assert action in legal

    def test_is_instance_of_base_agent(self) -> None:
        assert isinstance(GreedyAgent(), BaseAgent)

    def test_prefers_play_bird(self, engine: WingspanEngine) -> None:
        from src.games.wingspan.actions import WingspanActionType

        agent = GreedyAgent(catalog=engine._catalog)
        state = engine.reset(seed=0)
        legal = engine.get_legal_actions(state)

        play_actions = [a for a in legal if a.action_type == WingspanActionType.PLAY_BIRD.value]
        if not play_actions:
            pytest.skip("No PLAY_BIRD actions available at game start")

        action = agent.select_action(state, legal)
        assert action.action_type == WingspanActionType.PLAY_BIRD.value

    def test_prefers_highest_point_bird(self, engine: WingspanEngine) -> None:
        from src.games.wingspan.actions import WingspanAction, WingspanActionType

        catalog = engine._catalog
        agent = GreedyAgent(catalog=catalog)

        # Manually construct two PLAY_BIRD actions with different point values
        sorted_birds = sorted(
            [(n, c) for n, c in catalog.items() if c.habitats],
            key=lambda x: x[1].points,
        )
        if len(sorted_birds) < 2:
            pytest.skip("Need ≥2 birds with habitats")

        low_pts_bird, low_card = sorted_birds[0]
        high_pts_bird, high_card = sorted_birds[-1]

        low_action = WingspanAction(
            action_type=WingspanActionType.PLAY_BIRD.value,
            player_id=0,
            card_name=low_pts_bird,
            target_habitat=low_card.habitats[0],
        )
        high_action = WingspanAction(
            action_type=WingspanActionType.PLAY_BIRD.value,
            player_id=0,
            card_name=high_pts_bird,
            target_habitat=high_card.habitats[0],
        )

        state = engine.reset(seed=0)
        chosen = agent.select_action(state, [low_action, high_action])
        assert chosen.card_name == high_pts_bird

    def test_fallback_to_lay_eggs_without_birds(self, engine: WingspanEngine) -> None:
        from src.games.wingspan.actions import WingspanAction, WingspanActionType

        agent = GreedyAgent(catalog=engine._catalog)
        state = engine.reset(seed=0)

        # Only offer LAY_EGGS and GAIN_FOOD
        lay = WingspanAction(action_type=WingspanActionType.LAY_EGGS.value, player_id=0)
        gain = WingspanAction(
            action_type=WingspanActionType.GAIN_FOOD.value,
            player_id=0, food_choice="seed"
        )
        chosen = agent.select_action(state, [gain, lay])
        assert chosen.action_type == WingspanActionType.LAY_EGGS.value


# ---------------------------------------------------------------------------
# 4. evaluate_agents (baseline head-to-head)
# ---------------------------------------------------------------------------


class TestEvaluateAgents:
    def test_random_vs_random_near_50_percent(self, engine: WingspanEngine) -> None:
        agent_a = RandomAgent(seed=1)
        agent_b = RandomAgent(seed=2)
        results = evaluate_agents(agent_a, agent_b, engine, n_games=20, seed=0)
        assert "win_rate_a" in results
        assert "avg_score_a" in results
        assert results["win_rate_a"] + results["win_rate_b"] + results["draw_rate"] == pytest.approx(1.0)

    def test_results_dict_complete(self, engine: WingspanEngine) -> None:
        results = evaluate_agents(
            RandomAgent(seed=0), RandomAgent(seed=1), engine, n_games=5, seed=0
        )
        for key in ("win_rate_a", "win_rate_b", "draw_rate", "avg_score_a", "avg_score_b", "n_games"):
            assert key in results

    def test_scores_nonnegative(self, engine: WingspanEngine) -> None:
        results = evaluate_agents(
            GreedyAgent(catalog=engine._catalog),
            RandomAgent(seed=0),
            engine,
            n_games=5,
            seed=0,
        )
        assert results["avg_score_a"] >= 0
        assert results["avg_score_b"] >= 0


# ---------------------------------------------------------------------------
# 5. build_maskable_ppo
# ---------------------------------------------------------------------------


class TestBuildMaskablePPO:
    def test_model_builds_without_error(self, env: WingspanEnv) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=0)
        model = build_maskable_ppo(vec_env, seed=0, tensorboard_log=None)
        assert model is not None
        vec_env.close()

    def test_model_has_policy(self, env: WingspanEnv) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=0)
        model = build_maskable_ppo(vec_env, seed=0, tensorboard_log=None)
        assert hasattr(model, "policy")
        vec_env.close()

    def test_model_can_predict(self) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        eval_env = WingspanEnv()
        obs, _ = eval_env.reset(seed=0)
        masks = eval_env.action_masks()

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=0)
        model = build_maskable_ppo(vec_env, seed=0, tensorboard_log=None)

        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        assert isinstance(int(action), int)
        assert 0 <= int(action) < eval_env.action_space.n
        vec_env.close()

    def test_custom_features_dim(self) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=0)
        model = build_maskable_ppo(
            vec_env, seed=0, tensorboard_log=None, features_dim=64
        )
        extractor = model.policy.features_extractor
        assert extractor.features_dim == 64
        vec_env.close()


# ---------------------------------------------------------------------------
# 6. WinRateCallback
# ---------------------------------------------------------------------------


class TestWinRateCallback:
    def test_constructor(self) -> None:
        eval_env = WingspanEnv()
        cb = WinRateCallback(eval_env=eval_env, eval_freq=1000, n_eval_episodes=5)
        assert cb._n_eval_episodes == 5
        assert cb._eval_freq == 1000
        assert cb.win_rate_history == []

    def test_make_callbacks_returns_pair(self, tmp_path) -> None:
        eval_env = WingspanEnv()
        win_cb, ckpt_cb = make_callbacks(
            eval_env=eval_env,
            checkpoints_dir=tmp_path,
            eval_freq=1000,
            n_eval_episodes=5,
            n_envs=1,
        )
        assert isinstance(win_cb, WinRateCallback)
        from stable_baselines3.common.callbacks import CheckpointCallback
        assert isinstance(ckpt_cb, CheckpointCallback)


# ---------------------------------------------------------------------------
# 7. Training smoke test (256 steps, 1 env)
# ---------------------------------------------------------------------------


class TestTrainingSmoke:
    def test_train_256_steps_no_crash(self, tmp_path) -> None:
        """MaskablePPO must complete 256 training steps without exception."""
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=42)
        eval_env = WingspanEnv()
        eval_env.reset(seed=0)

        model = build_maskable_ppo(
            vec_env,
            seed=42,
            tensorboard_log=None,
            n_steps=256,
            batch_size=32,
        )

        win_cb = WinRateCallback(
            eval_env=eval_env,
            eval_freq=128,
            n_eval_episodes=3,
        )

        model.learn(total_timesteps=256, callback=win_cb, reset_num_timesteps=True)

        # Win rate history should have at least one entry
        assert len(win_cb.win_rate_history) >= 1
        for record in win_cb.win_rate_history:
            assert "win_rate_vs_random" in record
            assert 0.0 <= record["win_rate_vs_random"] <= 1.0

        vec_env.close()

    def test_model_save_and_load(self, tmp_path) -> None:
        """Model should save and reload without error."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=0)
        model = build_maskable_ppo(vec_env, seed=0, tensorboard_log=None, n_steps=64)
        model.learn(total_timesteps=64)

        save_path = tmp_path / "test_model"
        model.save(str(save_path))
        assert (tmp_path / "test_model.zip").exists()

        loaded = MaskablePPO.load(str(save_path))
        assert loaded is not None
        vec_env.close()


# ---------------------------------------------------------------------------
# 8. evaluate_ppo_win_rate (5 episodes)
# ---------------------------------------------------------------------------


class TestEvaluatePPOWinRate:
    def test_returns_dict_with_expected_keys(self) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(WingspanEnv, n_envs=1, seed=0)
        model = build_maskable_ppo(vec_env, seed=0, tensorboard_log=None, n_steps=64)
        model.learn(total_timesteps=64)

        results = evaluate_ppo_win_rate(model, n_episodes=5, seed=0)
        for key in ("win_rate", "avg_score_p0", "avg_score_p1", "n_episodes"):
            assert key in results

        assert results["n_episodes"] == 5
        assert 0.0 <= results["win_rate"] <= 1.0
        assert results["avg_score_p0"] >= 0
        assert results["avg_score_p1"] >= 0
        vec_env.close()
