"""Tests for the Splendor game engine and environment."""

from __future__ import annotations

import pytest
from gymnasium.utils.env_checker import check_env


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_import_engine() -> None:
    from src.games.splendor.engine import SplendorEngine
    engine = SplendorEngine()
    assert engine is not None


def test_import_env() -> None:
    from src.envs.splendor_env import SplendorEnv
    env = SplendorEnv()
    assert env is not None
    env.close()


def test_import_extractor() -> None:
    from src.agents.splendor_extractor import SplendorFeaturesExtractor
    assert SplendorFeaturesExtractor is not None


# ---------------------------------------------------------------------------
# Engine — reset
# ---------------------------------------------------------------------------


def test_engine_reset_initial_state() -> None:
    from src.games.splendor.engine import SplendorEngine

    engine = SplendorEngine()
    state = engine.reset(seed=42)

    assert state.player_id == 0
    assert state.turn == 0
    assert len(state.board) == 3
    for row in state.board:
        assert len(row) == 4
    assert len(state.nobles_available) == 3
    assert sum(state.bank.values()) > 0


def test_engine_reset_board_cards_are_valid() -> None:
    from src.games.splendor.cards import CARDS_BY_ID
    from src.games.splendor.engine import SplendorEngine

    engine = SplendorEngine()
    state = engine.reset(seed=0)

    for row in state.board:
        for card_id in row:
            if card_id is not None:
                assert card_id in CARDS_BY_ID


def test_engine_legal_actions_nonempty() -> None:
    from src.games.splendor.engine import SplendorEngine

    engine = SplendorEngine()
    state = engine.reset(seed=42)
    actions = engine.get_legal_actions(state)
    assert len(actions) > 0


# ---------------------------------------------------------------------------
# Engine — action invariants
# ---------------------------------------------------------------------------


def test_take_3_gems_reduces_bank() -> None:
    from src.games.splendor.actions import SplendorActionType
    from src.games.splendor.engine import SplendorEngine

    engine = SplendorEngine()
    state = engine.reset(seed=42)
    actions = engine.get_legal_actions(state)
    take3 = [a for a in actions if a.action_type == SplendorActionType.TAKE_3_GEMS]
    assert take3, "Should have take-3 actions at game start"

    act = take3[0]
    result = engine.step(state, act)
    new_state = result.new_state

    for gem in act.gems_taken:
        assert new_state.bank[gem] == state.bank[gem] - 1


def test_buy_card_increases_vp() -> None:
    from src.games.splendor.actions import SplendorAction, SplendorActionType
    from src.games.splendor.cards import CARDS_BY_ID
    from src.games.splendor.engine import SplendorEngine
    from src.games.splendor.state import SplendorPlayerBoard

    engine = SplendorEngine()
    state = engine.reset(seed=42)

    # Manually give player enough gems to buy a tier-1 card
    board = state.get_board(0)
    board.gems = {"white": 4, "blue": 4, "green": 4, "red": 4, "black": 4, "gold": 0}
    state = state.with_board(0, board)

    actions = engine.get_legal_actions(state)
    buy_actions = [a for a in actions if a.action_type == SplendorActionType.BUY_BOARD]
    assert buy_actions, "Should be able to buy at least one card with 4 of each gem"

    act = buy_actions[0]
    prev_vp = state.get_board(0).vp()
    result = engine.step(state, act)
    new_vp = result.new_state.get_board(0).vp()
    card = CARDS_BY_ID[act.card_id]
    assert new_vp == prev_vp + card.vp


def test_player_gems_never_exceed_10() -> None:
    """After any take action, player total gems ≤ 10."""
    from src.games.splendor.actions import SplendorActionType
    from src.games.splendor.engine import SplendorEngine

    engine = SplendorEngine()
    state = engine.reset(seed=7)

    for _ in range(20):
        if engine.is_terminal(state):
            break
        actions = engine.get_legal_actions(state)
        take_actions = [a for a in actions if a.action_type in (
            SplendorActionType.TAKE_3_GEMS, SplendorActionType.TAKE_2_GEMS
        )]
        act = take_actions[0] if take_actions else actions[0]
        result = engine.step(state, act)
        state = result.new_state

        # Check the player who just acted (player_id has flipped after step)
        prev_pid = 1 - state.player_id
        assert state.get_board(prev_pid).total_gems() <= 10


# ---------------------------------------------------------------------------
# Engine — full random game
# ---------------------------------------------------------------------------


def test_random_game_completes() -> None:
    """A full game with two random agents must complete without error."""
    import random
    from src.games.splendor.engine import SplendorEngine

    engine = SplendorEngine()
    rng = random.Random(42)

    for _ in range(10):
        state = engine.reset(seed=rng.randint(0, 10_000))
        max_steps = 500  # safety cutoff

        for step in range(max_steps):
            if engine.is_terminal(state):
                break
            actions = engine.get_legal_actions(state)
            if not actions:
                # Degenerate state — env truncates; engine returning [] is valid
                break
            act = rng.choice(actions)
            result = engine.step(state, act)
            assert result.success
            state = result.new_state


def test_winner_has_15_vp() -> None:
    """When the game ends, at least one player must have ≥ 15 VP."""
    import random
    from src.games.splendor.engine import SplendorEngine
    from src.games.splendor.cards import VICTORY_POINTS_TARGET

    engine = SplendorEngine()
    rng = random.Random(0)
    state = engine.reset(seed=0)
    max_steps = 500

    for _ in range(max_steps):
        if engine.is_terminal(state):
            break
        actions = engine.get_legal_actions(state)
        act = rng.choice(actions)
        result = engine.step(state, act)
        state = result.new_state

    if engine.is_terminal(state):
        boards = state.get_boards()
        max_vp = max(b.vp() for b in boards)
        assert max_vp >= VICTORY_POINTS_TARGET


# ---------------------------------------------------------------------------
# Environment — gym check
# ---------------------------------------------------------------------------


def test_gym_env_check() -> None:
    """check_env() must pass without warnings."""
    from src.envs.splendor_env import SplendorEnv

    env = SplendorEnv()
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def test_env_reset_returns_valid_obs() -> None:
    from src.envs.splendor_env import (
        BANK_FEATURES, BOARD_FEATURES, DECK_FEATURES,
        NOBLES_TOTAL, PLAYER_FEATURES, SplendorEnv,
    )

    env = SplendorEnv()
    obs, info = env.reset(seed=42)

    assert obs["bank"].shape == (BANK_FEATURES,)
    assert obs["board"].shape == (BOARD_FEATURES,)
    assert obs["deck_sizes"].shape == (DECK_FEATURES,)
    assert obs["nobles"].shape == (NOBLES_TOTAL,)
    assert obs["player"].shape == (PLAYER_FEATURES,)
    assert obs["opponent"].shape == (PLAYER_FEATURES,)
    assert obs["game_state"].shape == (1,)

    for key, arr in obs.items():
        assert arr.min() >= 0.0, f"{key} has negative value"
        assert arr.max() <= 1.0, f"{key} exceeds 1.0"
    env.close()


def test_env_action_mask_consistent() -> None:
    """Every masked action must be in the engine's legal action list."""
    import numpy as np
    from src.envs.splendor_env import SplendorEnv
    from src.games.splendor.actions import action_to_index

    env = SplendorEnv()
    env.reset(seed=42)
    mask = env.action_masks()

    engine_legal_indices = set()
    for a in env._engine.get_legal_actions(env._state):
        try:
            engine_legal_indices.add(action_to_index(a))
        except (ValueError, KeyError):
            pass

    masked_true = set(np.where(mask)[0])
    # Every masked action must be in legal set (mask may be subset if duplicates)
    extra = masked_true - engine_legal_indices
    assert not extra, f"Mask has {len(extra)} indices not in legal actions: {extra}"
    env.close()


def test_env_step_episode_runs() -> None:
    """A full episode using random masked actions must complete."""
    import numpy as np
    from src.envs.splendor_env import SplendorEnv

    env = SplendorEnv()
    obs, _ = env.reset(seed=42)

    for _ in range(300):
        mask = env.action_masks()
        legal_idxs = np.where(mask)[0]
        action = int(np.random.choice(legal_idxs))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["bank"].min() >= 0.0
        if terminated or truncated:
            assert "winner" in info
            break

    env.close()


def test_env_player_id_invariant() -> None:
    """state.player_id must be 0 at the start of every step() call."""
    import numpy as np
    from src.envs.splendor_env import SplendorEnv

    env = SplendorEnv()
    env.reset(seed=123)

    for _ in range(50):
        assert env._state.player_id == 0, "Turn invariant violated"
        mask = env.action_masks()
        legal = np.where(mask)[0]
        action = int(np.random.choice(legal))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
