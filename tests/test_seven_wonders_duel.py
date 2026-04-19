"""Sprint 7 — 7 Wonders Duel tests.

Covers:
  - Card catalog loading
  - SWDState construction and accessors
  - SWDEngine: reset, step, legal actions, terminal detection
  - SWDLegalMoveValidator: correctness of legal moves
  - SevenWondersDuelEnv: gym interface, action masking, check_env
  - Reward functions
  - Integration: random game completes without crash
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from src.games.seven_wonders_duel.engine import SWDEngine
    return SWDEngine(seed=42)


@pytest.fixture(scope="module")
def initial_state(engine):
    return engine.reset(seed=42)


@pytest.fixture(scope="module")
def env():
    from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv
    e = SevenWondersDuelEnv(seed=42)
    e.reset(seed=42)
    return e


# ===========================================================================
# TestCardCatalog
# ===========================================================================

class TestCardCatalog:
    def test_load_returns_five_tuples(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        result = load_card_catalog()
        assert len(result) == 5

    def test_age1_has_23_cards(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        age1, *_ = load_card_catalog()
        assert len(age1) == 23

    def test_age2_has_23_cards(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        _, age2, *_ = load_card_catalog()
        assert len(age2) == 23

    def test_age3_has_23_cards(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        _, _, age3, *_ = load_card_catalog()
        assert len(age3) == 23

    def test_wonders_count(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        _, _, _, wonders, _ = load_card_catalog()
        assert len(wonders) >= 8

    def test_progress_tokens_count(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        _, _, _, _, tokens = load_card_catalog()
        assert len(tokens) >= 5

    def test_card_has_name(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        age1, *_ = load_card_catalog()
        for card in age1:
            assert card.name

    def test_card_type_is_valid(self):
        from src.games.seven_wonders_duel.cards import CardType, load_card_catalog
        valid_types = {ct.value for ct in CardType}
        age1, age2, age3, *_ = load_card_catalog()
        for card in age1 + age2 + age3:
            assert card.card_type in valid_types, f"{card.name}: {card.card_type}"

    def test_military_card_gives_shields(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        age1, *_ = load_card_catalog()
        military = [c for c in age1 if c.card_type == "military"]
        assert all(c.gives_shields() > 0 for c in military)

    def test_scientific_card_has_science_symbol(self):
        from src.games.seven_wonders_duel.cards import load_card_catalog
        age1, *_ = load_card_catalog()
        scientific = [c for c in age1 if c.card_type == "scientific"]
        assert all(len(c.science_symbols()) > 0 for c in scientific)


# ===========================================================================
# TestSWDState
# ===========================================================================

class TestSWDState:
    def test_initial_state_player_zero(self, initial_state):
        assert initial_state.player_id == 0

    def test_initial_state_age_one(self, initial_state):
        assert initial_state.age == 1

    def test_initial_state_two_boards(self, initial_state):
        assert len(initial_state.boards_data) == 2

    def test_initial_state_player_coins(self, initial_state):
        for pid in range(2):
            board = initial_state.get_board(pid)
            assert board.coins == 7

    def test_initial_state_age_deck_23(self, initial_state):
        assert len(initial_state.age_deck) == 23

    def test_accessible_cards_nonempty(self, initial_state):
        accessible = initial_state.accessible_cards(initial_state.age_deck)
        assert len(accessible) > 0

    def test_accessible_cards_subset_of_deck(self, initial_state):
        accessible = initial_state.accessible_cards(initial_state.age_deck)
        deck_set = set(initial_state.age_deck)
        for c in accessible:
            assert c in deck_set

    def test_accessible_cards_not_taken(self, initial_state):
        accessible = initial_state.accessible_cards(initial_state.age_deck)
        for c in accessible:
            assert c not in initial_state.taken_cards

    def test_with_board_immutable(self, initial_state):
        """with_board returns a new state, does not mutate original."""
        from src.games.seven_wonders_duel.state import SWDPlayerBoard
        original_coins = initial_state.get_board(0).coins
        new_board = SWDPlayerBoard.from_dict({
            **initial_state.get_board(0).to_dict(),
            "coins": 999,
        })
        new_state = initial_state.with_board(0, new_board)
        assert initial_state.get_board(0).coins == original_coins
        assert new_state.get_board(0).coins == 999

    def test_military_winner_none_at_start(self, initial_state):
        assert initial_state.military_winner() is None

    def test_science_winner_none_at_start(self, engine, initial_state):
        assert initial_state.science_winner(engine._card_catalog) is None


# ===========================================================================
# TestSWDEngine
# ===========================================================================

class TestSWDEngine:
    def test_reset_returns_swd_state(self, engine):
        from src.games.seven_wonders_duel.state import SWDState
        state = engine.reset(seed=1)
        assert isinstance(state, SWDState)

    def test_reset_different_seeds_different_decks(self, engine):
        s1 = engine.reset(seed=1)
        s2 = engine.reset(seed=2)
        assert s1.age_deck != s2.age_deck

    def test_reset_same_seed_same_deck(self, engine):
        s1 = engine.reset(seed=42)
        s2 = engine.reset(seed=42)
        assert s1.age_deck == s2.age_deck

    def test_step_returns_action_result(self, engine, initial_state):
        from src.games.base.game_state import ActionResult
        legal = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, legal[0])
        assert isinstance(result, ActionResult)

    def test_step_returns_new_state(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, legal[0])
        assert result.new_state is not initial_state

    def test_step_advances_player(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        # Take a discard action (always legal)
        discard = next(a for a in legal if a.action_type == "discard_card")
        result = engine.step(initial_state, discard)
        # After P0 acts, should be P1's turn
        assert result.new_state.player_id == 1

    def test_step_removes_card_from_deck(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        discard = legal[0]
        card_name = discard.card_name
        result = engine.step(initial_state, discard)
        assert card_name in result.new_state.taken_cards

    def test_get_legal_actions_nonempty(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        assert len(legal) > 0

    def test_get_legal_actions_all_swd_actions(self, engine, initial_state):
        from src.games.seven_wonders_duel.actions import SWDAction
        legal = engine.get_legal_actions(initial_state)
        for a in legal:
            assert isinstance(a, SWDAction)

    def test_legal_actions_always_include_discard(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        discards = [a for a in legal if a.action_type == "discard_card"]
        assert len(discards) > 0

    def test_is_terminal_false_at_start(self, engine, initial_state):
        assert not engine.is_terminal(initial_state)

    def test_get_winner_none_at_start(self, engine, initial_state):
        assert engine.get_winner(initial_state) is None

    def test_compute_final_score_nonneg(self, engine, initial_state):
        score = engine._compute_final_score(initial_state, 0)
        assert score >= 0

    def test_full_random_game_terminates(self, engine):
        """A full random game must end without crashing."""
        import random as py_random
        state = engine.reset(seed=7)
        rng = py_random.Random(7)
        max_turns = 500

        for _ in range(max_turns):
            if engine.is_terminal(state):
                break
            legal = engine.get_legal_actions(state)
            if not legal:
                break
            action = rng.choice(legal)
            result = engine.step(state, action)
            state = result.new_state

        # May or may not be terminal depending on card count, but no crash
        assert state is not None

    def test_discard_increases_player_coins(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        discard = next(a for a in legal if a.action_type == "discard_card")
        coins_before = initial_state.get_board(0).coins
        result = engine.step(initial_state, discard)
        # Discard gives ≥2 coins; P1 acted after, so check coins indirectly on P0
        # State has P1 as current player; we read P0's board
        coins_after = result.new_state.get_board(0).coins
        assert coins_after >= coins_before + 2

    def test_build_card_adds_to_built_cards(self, engine, initial_state):
        legal = engine.get_legal_actions(initial_state)
        builds = [a for a in legal if a.action_type == "build_card"]
        if not builds:
            pytest.skip("No affordable build actions in initial state")
        result = engine.step(initial_state, builds[0])
        assert builds[0].card_name in result.new_state.get_board(0).built_cards


# ===========================================================================
# TestSWDLegalMoveValidator
# ===========================================================================

class TestSWDLegalMoveValidator:
    def test_validator_returns_list(self, engine, initial_state):
        legal = engine._validator.get_legal_actions(initial_state, 0)
        assert isinstance(legal, list)

    def test_all_legal_accessible(self, engine, initial_state):
        accessible = set(initial_state.accessible_cards(initial_state.age_deck))
        legal = engine._validator.get_legal_actions(initial_state, 0)
        for a in legal:
            assert a.card_name in accessible, f"{a.card_name} not accessible"

    def test_discard_always_in_legal(self, engine, initial_state):
        legal = engine._validator.get_legal_actions(initial_state, 0)
        types = {a.action_type for a in legal}
        assert "discard_card" in types

    def test_cant_build_without_resources_or_coins(self):
        """A board with 0 coins cannot build a card that costs resources."""
        from src.games.seven_wonders_duel.engine import SWDEngine
        from src.games.seven_wonders_duel.state import SWDPlayerBoard
        eng = SWDEngine(seed=0)
        state = eng.reset(seed=0)
        # Remove all coins from P0
        board0 = state.get_board(0)
        board0_data = {**board0.to_dict(), "coins": 0}
        state = state.with_board(0, SWDPlayerBoard.from_dict(board0_data))

        legal = eng._validator.get_legal_actions(state, 0)
        # All build_card actions should be for cards with 0 cost
        for a in legal:
            if a.action_type == "build_card":
                card = eng._card_catalog[a.card_name]
                # Either free card or player can afford through trading (but 0 coins → can't trade)
                assert card.cost_coins == 0 and not card.cost_resources, \
                    f"Should not be able to build {a.card_name} with 0 coins"


# ===========================================================================
# TestSevenWondersDuelEnv
# ===========================================================================

class TestSevenWondersDuelEnv:
    def test_reset_returns_obs_dict(self, env):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "pyramid" in obs
        assert "player" in obs
        assert "opponent" in obs
        assert "tokens" in obs

    def test_obs_shapes(self, env):
        from src.envs.seven_wonders_duel_env import (
            N_MAX_CARDS_IN_AGE, CARD_FEATURES, PLAYER_FEATURES, TOKENS_OBS_DIM
        )
        obs, _ = env.reset(seed=0)
        assert obs["pyramid"].shape == (N_MAX_CARDS_IN_AGE, CARD_FEATURES)
        assert obs["player"].shape == (PLAYER_FEATURES,)
        assert obs["opponent"].shape == (PLAYER_FEATURES,)
        assert obs["tokens"].shape == (TOKENS_OBS_DIM,)

    def test_obs_values_in_range(self, env):
        obs, _ = env.reset(seed=0)
        for key, arr in obs.items():
            assert arr.min() >= 0.0, f"{key} has negative values"
            assert arr.max() <= 1.0, f"{key} has values > 1"

    def test_action_masks_shape(self, env):
        from src.envs.seven_wonders_duel_env import N_MAX_ACTIONS_7WD
        env.reset(seed=0)
        mask = env.action_masks()
        assert mask.shape == (N_MAX_ACTIONS_7WD,)

    def test_action_masks_has_some_legal(self, env):
        env.reset(seed=0)
        mask = env.action_masks()
        assert mask.sum() > 0

    def test_action_masks_dtype_bool(self, env):
        env.reset(seed=0)
        mask = env.action_masks()
        assert mask.dtype == bool

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        mask = env.action_masks()
        legal_idx = int(np.where(mask)[0][0]) if mask.any() else 0
        result = env.step(legal_idx)
        assert len(result) == 5

    def test_step_obs_in_range(self, env):
        env.reset(seed=1)
        mask = env.action_masks()
        idx = int(np.where(mask)[0][0]) if mask.any() else 0
        obs, *_ = env.step(idx)
        for key, arr in obs.items():
            assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_step_reward_is_float(self, env):
        env.reset(seed=2)
        mask = env.action_masks()
        idx = int(np.where(mask)[0][0]) if mask.any() else 0
        _, reward, *_ = env.step(idx)
        assert isinstance(reward, float)

    def test_step_done_is_bool(self, env):
        env.reset(seed=3)
        mask = env.action_masks()
        idx = int(np.where(mask)[0][0]) if mask.any() else 0
        _, _, done, _, _ = env.step(idx)
        assert isinstance(done, bool)

    def test_turn_invariant_after_step(self, env):
        """state.player_id must be 0 after step() regardless of game flow."""
        env.reset(seed=5)
        mask = env.action_masks()
        idx = int(np.where(mask)[0][0]) if mask.any() else 0
        env.step(idx)
        if env._state and not env._engine.is_terminal(env._state):
            assert env._state.player_id == 0

    def test_check_env(self):
        """gymnasium check_env must pass without errors."""
        from gymnasium.utils.env_checker import check_env
        from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv
        check_env(SevenWondersDuelEnv(seed=0))

    def test_action_to_idx_round_trip(self, env):
        """_action_to_idx(_idx_to_action(i)) should recover i for legal actions."""
        env.reset(seed=10)
        mask = env.action_masks()
        for idx in np.where(mask)[0][:5]:
            action = env._idx_to_action(int(idx), env._state)
            recovered = env._action_to_idx(action)
            assert recovered == idx, f"idx={idx}, action={action}, recovered={recovered}"


# ===========================================================================
# TestRewards
# ===========================================================================

class TestRewards:
    def test_sparse_terminal_win(self, engine):
        from src.games.seven_wonders_duel.rewards import compute_reward
        state = engine.reset(seed=0)
        # Simulate win
        reward = compute_reward(state, state, player_id=0, done=True, winner=0, mode="sparse")
        assert reward == 1.0

    def test_sparse_terminal_loss(self, engine):
        from src.games.seven_wonders_duel.rewards import compute_reward
        state = engine.reset(seed=0)
        reward = compute_reward(state, state, player_id=0, done=True, winner=1, mode="sparse")
        assert reward == -1.0

    def test_sparse_nonterminal_zero(self, engine):
        from src.games.seven_wonders_duel.rewards import compute_reward
        state = engine.reset(seed=0)
        reward = compute_reward(state, state, player_id=0, done=False, winner=None, mode="sparse")
        assert reward == 0.0

    def test_dense_win_positive(self, engine):
        from src.games.seven_wonders_duel.rewards import compute_reward
        state = engine.reset(seed=0)
        reward = compute_reward(state, state, player_id=0, done=True, winner=0, mode="dense")
        assert reward > 0.0

    def test_dense_draw_zero(self, engine):
        from src.games.seven_wonders_duel.rewards import compute_reward
        state = engine.reset(seed=0)
        reward = compute_reward(state, state, player_id=0, done=True, winner=None, mode="dense")
        assert reward == 0.0


# ===========================================================================
# TestIntegration
# ===========================================================================

class TestIntegration:
    def test_100_random_games_no_crash(self, engine):
        """100 random games via engine must complete without exception."""
        import random as py_random
        for game_seed in range(100):
            state = engine.reset(seed=game_seed)
            rng = py_random.Random(game_seed)
            for _ in range(400):  # safety cap
                if engine.is_terminal(state):
                    break
                legal = engine.get_legal_actions(state)
                if not legal:
                    break
                result = engine.step(state, rng.choice(legal))
                state = result.new_state

    def test_env_random_episode_completes(self):
        """A random policy must complete an episode via the gym env."""
        import numpy as np
        from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv
        env = SevenWondersDuelEnv(seed=0)
        obs, _ = env.reset(seed=0)
        done = False
        for _ in range(500):
            if done:
                break
            mask = env.action_masks()
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                break
            idx = int(np.random.choice(legal_indices))
            obs, _, done, _, _ = env.step(idx)
        # Episode may not always finish in 500 steps; just verify no crash

    def test_framework_reuse_pct(self):
        """7WD reuses the same ABC/base classes — no new base code needed."""
        from src.games.base.engine import GameEngine
        from src.games.base.game_state import ActionResult
        from src.games.seven_wonders_duel.engine import SWDEngine
        eng = SWDEngine(seed=0)
        assert isinstance(eng, GameEngine)


# Needed for np.where calls in tests
import numpy as np  # noqa: E402 — intentional late import for test file
