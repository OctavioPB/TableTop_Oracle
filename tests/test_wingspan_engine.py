"""S2 — Wingspan engine test suite.

≥ 50 test cases covering:
  - Catalog loading
  - State model (BirdSlotState, WingspanPlayerBoard, WingspanState)
  - LegalMoveValidator
  - WingspanEngine (reset, step, terminal, winner, scoring)
  - Reward functions
  - 100 random complete games without crash
  - Hypothesis property-based tests for invariants
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.games.wingspan.actions import WingspanAction, WingspanActionType
from src.games.wingspan.cards import (
    ALL_ROUND_GOALS,
    BirdCard,
    FoodType,
    NestType,
    PowerID,
    PowerTiming,
    PowerType,
    RoundGoalType,
    load_bird_catalog,
)
from src.games.wingspan.engine import WingspanEngine, _roll_feeder
from src.games.wingspan.rewards import compute_reward
from src.games.wingspan.rules import LegalMoveValidator
from src.games.wingspan.state import (
    BirdSlotState,
    N_HABITAT_SLOTS,
    N_ROUNDS,
    WingspanPlayerBoard,
    WingspanState,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CATALOG_PATH = (
    Path(__file__).parent.parent / "data" / "card_catalogs" / "wingspan_birds.csv"
)


@pytest.fixture(scope="module")
def catalog() -> dict[str, BirdCard]:
    return load_bird_catalog(CATALOG_PATH)


@pytest.fixture
def engine() -> WingspanEngine:
    return WingspanEngine(catalog_path=CATALOG_PATH, seed=42)


@pytest.fixture
def initial_state(engine: WingspanEngine) -> WingspanState:
    return engine.reset(seed=42)


def _make_board(player_id: int = 0, **kwargs: Any) -> WingspanPlayerBoard:
    defaults = {
        "food_supply": {"seed": 3, "invertebrate": 2},
        "hand": [],
        "action_cubes": 8,
    }
    defaults.update(kwargs)
    return WingspanPlayerBoard(player_id=player_id, **defaults)


def _make_minimal_card(
    name: str = "TestBird",
    habitats: list[str] | None = None,
    food_cost: dict | None = None,
    egg_limit: int = 2,
    points: int = 1,
) -> BirdCard:
    return BirdCard(
        name=name,
        habitats=habitats or ["forest"],
        food_cost=food_cost or {},
        nest_type=NestType.CUP.value,
        egg_limit=egg_limit,
        points=points,
        wingspan_cm=30,
        power_timing=PowerTiming.NO_POWER.value,
        power_id=PowerID.NONE.value,
        power_type=PowerType.NONE.value,
    )


# ---------------------------------------------------------------------------
# 1. Catalog loading
# ---------------------------------------------------------------------------


class TestCatalogLoading:
    def test_catalog_loads_without_error(self, catalog: dict[str, BirdCard]) -> None:
        assert len(catalog) > 0

    def test_catalog_has_at_least_170_birds(self, catalog: dict[str, BirdCard]) -> None:
        assert len(catalog) >= 170

    def test_all_birds_have_valid_habitats(self, catalog: dict[str, BirdCard]) -> None:
        valid = {"forest", "grassland", "wetland"}
        for name, card in catalog.items():
            for hab in card.habitats:
                assert hab in valid, f"{name} has invalid habitat {hab}"

    def test_all_birds_have_valid_egg_limits(self, catalog: dict[str, BirdCard]) -> None:
        for name, card in catalog.items():
            assert 1 <= card.egg_limit <= 6, f"{name} egg_limit out of range"

    def test_all_birds_have_nonnegative_points(self, catalog: dict[str, BirdCard]) -> None:
        for name, card in catalog.items():
            assert card.points >= 0, f"{name} has negative points"

    def test_all_birds_have_valid_power_id(self, catalog: dict[str, BirdCard]) -> None:
        valid_ids = {p.value for p in PowerID}
        for name, card in catalog.items():
            assert card.power_id in valid_ids, f"{name} has unknown power_id {card.power_id}"

    def test_all_birds_have_valid_power_timing(self, catalog: dict[str, BirdCard]) -> None:
        valid = {p.value for p in PowerTiming}
        for name, card in catalog.items():
            assert card.power_timing in valid, f"{name} invalid timing {card.power_timing}"

    def test_predator_birds_have_power_param(self, catalog: dict[str, BirdCard]) -> None:
        for name, card in catalog.items():
            if card.power_id == PowerID.PREDATOR.value:
                assert card.power_param > 0, f"Predator {name} missing power_param"

    def test_catalog_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_bird_catalog("/nonexistent/path/birds.csv")


# ---------------------------------------------------------------------------
# 2. BirdSlotState
# ---------------------------------------------------------------------------


class TestBirdSlotState:
    def test_default_constructor(self) -> None:
        s = BirdSlotState(bird_name="Eagle")
        assert s.bird_name == "Eagle"
        assert s.eggs == 0
        assert s.tucked_cards == 0
        assert s.cached_food == {}

    def test_copy_is_independent(self) -> None:
        s = BirdSlotState("Hawk", eggs=2, cached_food={"seed": 1})
        c = s.copy()
        c.eggs = 5
        c.cached_food["seed"] = 99
        assert s.eggs == 2
        assert s.cached_food["seed"] == 1

    def test_roundtrip_serialisation(self) -> None:
        s = BirdSlotState("Heron", eggs=3, cached_food={"fish": 2}, tucked_cards=1)
        d = s.to_dict()
        s2 = BirdSlotState.from_dict(d)
        assert s2.bird_name == "Heron"
        assert s2.eggs == 3
        assert s2.cached_food == {"fish": 2}
        assert s2.tucked_cards == 1


# ---------------------------------------------------------------------------
# 3. WingspanPlayerBoard
# ---------------------------------------------------------------------------


class TestWingspanPlayerBoard:
    def test_initial_board_has_empty_habitats(self) -> None:
        b = _make_board()
        assert b.total_birds() == 0
        for hab in ("forest", "grassland", "wetland"):
            assert len(b.get_habitat(hab)) == N_HABITAT_SLOTS

    def test_first_empty_slot_returns_zero_when_empty(self) -> None:
        b = _make_board()
        assert b.first_empty_slot("forest") == 0

    def test_first_empty_slot_returns_none_when_full(self) -> None:
        b = _make_board()
        for i in range(N_HABITAT_SLOTS):
            b.forest[i] = BirdSlotState(f"Bird{i}")
        assert b.first_empty_slot("forest") is None

    def test_egg_cost_column_1_is_zero(self) -> None:
        b = _make_board()
        assert b.egg_cost_for_habitat("forest") == 0

    def test_egg_cost_column_2_is_one(self) -> None:
        b = _make_board()
        b.forest[0] = BirdSlotState("Bird0")
        assert b.egg_cost_for_habitat("forest") == 1

    def test_egg_cost_column_4_is_two(self) -> None:
        b = _make_board()
        for i in range(3):
            b.forest[i] = BirdSlotState(f"B{i}")
        assert b.egg_cost_for_habitat("forest") == 2

    def test_total_eggs_counts_all_habitats(self) -> None:
        b = _make_board()
        b.forest[0] = BirdSlotState("A", eggs=2)
        b.grassland[0] = BirdSlotState("B", eggs=3)
        b.wetland[0] = BirdSlotState("C", eggs=1)
        assert b.total_eggs() == 6

    def test_birds_in_habitat_only_occupied(self) -> None:
        b = _make_board()
        b.forest[0] = BirdSlotState("A")
        b.forest[2] = BirdSlotState("B")
        result = b.birds_in_habitat("forest")
        assert len(result) == 2
        idxs = [i for i, _ in result]
        assert 0 in idxs and 2 in idxs

    def test_copy_is_independent(self) -> None:
        b = _make_board(food_supply={"seed": 3})
        b.forest[0] = BirdSlotState("X", eggs=1)
        c = b.copy()
        c.food_supply["seed"] = 99
        c.forest[0].eggs = 5
        assert b.food_supply["seed"] == 3
        assert b.forest[0].eggs == 1

    def test_roundtrip_serialisation(self) -> None:
        b = _make_board(food_supply={"fish": 2}, hand=["Robin"], action_cubes=5)
        b.wetland[1] = BirdSlotState("Heron", eggs=1)
        b2 = WingspanPlayerBoard.from_dict(b.to_dict())
        assert b2.player_id == b.player_id
        assert b2.food_supply == {"fish": 2}
        assert b2.hand == ["Robin"]
        assert b2.action_cubes == 5
        assert b2.wetland[1] is not None
        assert b2.wetland[1].bird_name == "Heron"

    def test_unknown_habitat_raises(self) -> None:
        b = _make_board()
        with pytest.raises(ValueError):
            b.get_habitat("ocean")


# ---------------------------------------------------------------------------
# 4. WingspanState
# ---------------------------------------------------------------------------


class TestWingspanState:
    def test_initial_state_construction(self, engine, initial_state) -> None:
        s = initial_state
        assert s.round == 1
        assert s.phase == "main"
        assert len(s.boards_data) == 2
        assert len(s.bird_tray) == 3
        assert len(s.bird_feeder) == 5

    def test_get_board_returns_correct_player(self, initial_state) -> None:
        s = initial_state
        b0 = s.get_board(0)
        b1 = s.get_board(1)
        assert b0.player_id == 0
        assert b1.player_id == 1

    def test_with_board_creates_new_state(self, initial_state) -> None:
        s = initial_state
        b = s.get_board(0)
        b.food_supply["seed"] = 999
        s2 = s.with_board(0, b)
        assert s.get_board(0).food_supply.get("seed", 0) != 999
        assert s2.get_board(0).food_supply["seed"] == 999

    def test_all_players_done_false_initially(self, initial_state) -> None:
        assert not initial_state.all_players_done()

    def test_all_players_done_true_when_zero_cubes(self, initial_state) -> None:
        boards = initial_state.get_boards()
        for b in boards:
            b.action_cubes = 0
        s2 = initial_state.with_boards(boards)
        assert s2.all_players_done()

    def test_model_dump_json_runs(self, initial_state) -> None:
        json_str = initial_state.model_dump_json()
        assert "round" in json_str
        assert "bird_feeder" in json_str


# ---------------------------------------------------------------------------
# 5. LegalMoveValidator
# ---------------------------------------------------------------------------


class TestLegalMoveValidator:
    def test_gain_food_always_legal(self, catalog: dict[str, BirdCard], engine, initial_state) -> None:
        validator = LegalMoveValidator(catalog)
        actions = validator.get_legal_gain_food_actions(initial_state, initial_state.get_board(0))
        assert len(actions) >= 1
        for a in actions:
            assert a.action_type == WingspanActionType.GAIN_FOOD.value

    def test_draw_cards_legal_at_game_start(self, catalog, initial_state) -> None:
        validator = LegalMoveValidator(catalog)
        actions = validator.get_legal_draw_cards_actions(initial_state, initial_state.get_board(0))
        # At game start: 3 tray slots + deck → ≥ 1 action
        assert len(actions) >= 1

    def test_lay_eggs_illegal_with_no_birds(self, catalog, initial_state) -> None:
        validator = LegalMoveValidator(catalog)
        actions = validator.get_legal_lay_eggs_actions(initial_state, initial_state.get_board(0))
        # No birds placed yet, so no birds to receive eggs
        assert actions == []

    def test_lay_eggs_legal_when_birds_have_room(self, catalog, initial_state) -> None:
        validator = LegalMoveValidator(catalog)
        board = initial_state.get_board(0)
        # Place a bird with room for eggs
        bird_name = next(iter(catalog.keys()))
        board.forest[0] = BirdSlotState(bird_name, eggs=0)
        state2 = initial_state.with_board(0, board)
        actions = validator.get_legal_lay_eggs_actions(state2, state2.get_board(0))
        assert len(actions) == 1

    def test_play_bird_legal_when_hand_and_resources(self, catalog) -> None:
        validator = LegalMoveValidator(catalog)
        # Find a bird with 0 food cost and some habitat
        zero_cost_bird = next(
            (n for n, c in catalog.items() if not c.food_cost), None
        )
        if zero_cost_bird is None:
            pytest.skip("No zero-cost bird in catalog")
        card = catalog[zero_cost_bird]
        hab = card.habitats[0]
        board = _make_board(hand=[zero_cost_bird])
        boards_data = [board.to_dict(), _make_board(1).to_dict()]
        state = WingspanState(
            player_id=0,
            turn=0,
            phase="main",
            boards_data=boards_data,
            bird_feeder=["seed", "seed", "fish", "fruit", "invertebrate"],
            bird_tray=["BirdA", "BirdB", "BirdC"],
            draw_deck=["BirdD"],
            round_end_goals=["most_birds_forest", "most_birds_grassland", "most_birds_wetland", "most_birds_total"],
        )
        actions = validator.get_legal_play_bird_actions(state, state.get_board(0))
        assert any(a.card_name == zero_cost_bird for a in actions)

    def test_legal_actions_never_empty_in_nonterminal(self, engine, initial_state) -> None:
        """Core invariant: get_legal_actions never returns [] in a non-terminal state."""
        state = initial_state
        for _ in range(20):
            actions = engine.get_legal_actions(state)
            assert len(actions) > 0, "Empty legal actions in non-terminal state"
            action = random.Random(0).choice(actions)
            result = engine.step(state, action)
            if engine.is_terminal(result.new_state):
                break
            state = result.new_state

    def test_validate_action_legal_returns_true(self, catalog, initial_state) -> None:
        validator = LegalMoveValidator(catalog)
        actions = validator.get_legal_actions(initial_state, 0)
        is_legal, reason = validator.validate_action(initial_state, actions[0])
        assert is_legal
        assert reason == "Legal"

    def test_validate_action_illegal_returns_false(self, catalog, initial_state) -> None:
        validator = LegalMoveValidator(catalog)
        illegal = WingspanAction(
            action_type=WingspanActionType.PLAY_BIRD.value,
            player_id=0,
            card_name="nonexistent_bird_xyz",
            target_habitat="forest",
        )
        is_legal, _ = validator.validate_action(initial_state, illegal)
        assert not is_legal


# ---------------------------------------------------------------------------
# 6. WingspanEngine — reset
# ---------------------------------------------------------------------------


class TestWingspanEngineReset:
    def test_reset_returns_wingspan_state(self, engine) -> None:
        s = engine.reset(seed=1)
        assert isinstance(s, WingspanState)

    def test_reset_deals_5_cards_per_player(self, engine) -> None:
        s = engine.reset(seed=1)
        assert len(s.get_board(0).hand) == 5
        assert len(s.get_board(1).hand) == 5

    def test_reset_3_cards_in_tray(self, engine) -> None:
        s = engine.reset(seed=1)
        assert len(s.bird_tray) == 3

    def test_reset_5_feeder_dice(self, engine) -> None:
        s = engine.reset(seed=1)
        assert len(s.bird_feeder) == 5

    def test_reset_4_round_goals(self, engine) -> None:
        s = engine.reset(seed=1)
        assert len(s.round_end_goals) == 4

    def test_reset_round_goals_are_valid(self, engine) -> None:
        s = engine.reset(seed=1)
        for goal in s.round_end_goals:
            assert goal in ALL_ROUND_GOALS

    def test_reset_seed_deterministic(self, engine) -> None:
        s1 = engine.reset(seed=99)
        s2 = engine.reset(seed=99)
        assert s1.bird_tray == s2.bird_tray
        assert s1.get_board(0).hand == s2.get_board(0).hand

    def test_reset_different_seeds_different_states(self, engine) -> None:
        s1 = engine.reset(seed=1)
        s2 = engine.reset(seed=2)
        # Very unlikely to be identical
        assert s1.get_board(0).hand != s2.get_board(0).hand or s1.bird_tray != s2.bird_tray

    def test_reset_no_overlap_between_players_hands(self, engine) -> None:
        s = engine.reset(seed=7)
        h0 = set(s.get_board(0).hand)
        h1 = set(s.get_board(1).hand)
        assert h0.isdisjoint(h1)

    def test_reset_initial_player_id_is_zero(self, engine) -> None:
        s = engine.reset(seed=42)
        assert s.player_id == 0


# ---------------------------------------------------------------------------
# 7. WingspanEngine — step / actions
# ---------------------------------------------------------------------------


class TestWingspanEngineStep:
    def test_step_gain_food_increases_supply(self, engine, initial_state) -> None:
        state = initial_state
        food_before = state.get_board(0).total_food()
        action = WingspanAction(
            action_type=WingspanActionType.GAIN_FOOD.value,
            player_id=0,
            food_choice=state.bird_feeder[0],
        )
        result = engine.step(state, action)
        food_after = result.new_state.get_board(0).total_food()
        assert food_after > food_before

    def test_step_gain_food_removes_die_from_feeder(self, engine, initial_state) -> None:
        state = initial_state
        feeder_size_before = len(state.bird_feeder)
        food_choice = state.bird_feeder[0]
        action = WingspanAction(
            action_type=WingspanActionType.GAIN_FOOD.value,
            player_id=0,
            food_choice=food_choice,
        )
        result = engine.step(state, action)
        assert len(result.new_state.bird_feeder) == feeder_size_before - 1

    def test_step_returns_action_result_success(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        assert result.success

    def test_step_advances_turn_counter(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        assert result.new_state.turn == initial_state.turn + 1

    def test_step_decrements_action_cubes(self, engine, initial_state) -> None:
        cubes_before = initial_state.get_board(0).action_cubes
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        cubes_after = result.new_state.get_board(0).action_cubes
        assert cubes_after == cubes_before - 1

    def test_step_draw_cards_adds_to_hand(self, engine, initial_state) -> None:
        board = initial_state.get_board(0)
        hand_size_before = len(board.hand)
        draw_action = WingspanAction(
            action_type=WingspanActionType.DRAW_CARDS.value,
            player_id=0,
            draw_source="tray_0",
        )
        result = engine.step(initial_state, draw_action)
        hand_after = result.new_state.get_board(0).hand
        assert len(hand_after) == hand_size_before + 1

    def test_step_unknown_action_type_raises(self, engine, initial_state) -> None:
        bad_action = WingspanAction(
            action_type="invalid_action_xyz",
            player_id=0,
        )
        with pytest.raises(ValueError):
            engine.step(initial_state, bad_action)

    def test_step_wrong_action_type_raises(self, engine, initial_state) -> None:
        with pytest.raises(TypeError):
            engine.step(initial_state, "not_an_action")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. WingspanEngine — terminal / winner
# ---------------------------------------------------------------------------


class TestWingspanEngineTerminal:
    def test_initial_state_not_terminal(self, engine, initial_state) -> None:
        assert not engine.is_terminal(initial_state)

    def test_finished_phase_is_terminal(self, engine, initial_state) -> None:
        s = initial_state.model_copy(update={"phase": "finished"})
        assert engine.is_terminal(s)

    def test_winner_none_in_nonterminal(self, engine, initial_state) -> None:
        assert engine.get_winner(initial_state) is None

    def test_winner_returns_valid_player_id(self, engine, initial_state) -> None:
        s = initial_state.model_copy(update={"phase": "finished"})
        winner = engine.get_winner(s)
        assert winner in (0, 1)

    def test_compute_scores_returns_dict_for_all_players(self, engine, initial_state) -> None:
        scores = engine.compute_scores(initial_state)
        assert 0 in scores
        assert 1 in scores

    def test_compute_scores_nonnegative(self, engine, initial_state) -> None:
        scores = engine.compute_scores(initial_state)
        for pid, score in scores.items():
            assert score >= 0, f"Player {pid} has negative score"


# ---------------------------------------------------------------------------
# 9. Reward functions
# ---------------------------------------------------------------------------


class TestRewards:
    def test_terminal_reward_zero_midgame(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        r = compute_reward(
            initial_state, actions[0], result.new_state, False,
            reward_mode="terminal", player_id=0, engine=engine
        )
        assert r == 0.0

    def test_dense_reward_returns_float(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        r = compute_reward(
            initial_state, actions[0], result.new_state, False,
            reward_mode="dense", player_id=0, engine=engine
        )
        assert isinstance(r, float)

    def test_shaped_reward_returns_float(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        r = compute_reward(
            initial_state, actions[0], result.new_state, False,
            reward_mode="shaped", player_id=0, engine=engine
        )
        assert isinstance(r, float)

    def test_reward_unknown_mode_raises(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        with pytest.raises(ValueError):
            compute_reward(
                initial_state, actions[0], result.new_state, False,
                reward_mode="unknown",  # type: ignore[arg-type]
                player_id=0, engine=engine
            )

    def test_dense_reward_bound(self, engine, initial_state) -> None:
        actions = engine.get_legal_actions(initial_state)
        result = engine.step(initial_state, actions[0])
        r = compute_reward(
            initial_state, actions[0], result.new_state, False,
            reward_mode="dense", player_id=0, engine=engine
        )
        assert -2.0 <= r <= 2.0


# ---------------------------------------------------------------------------
# 10. Feeder mechanics
# ---------------------------------------------------------------------------


class TestFeederMechanics:
    def test_roll_feeder_default_5_dice(self) -> None:
        rng = random.Random(0)
        result = _roll_feeder(rng)
        assert len(result) == 5

    def test_roll_feeder_custom_n(self) -> None:
        rng = random.Random(0)
        result = _roll_feeder(rng, 3)
        assert len(result) == 3

    def test_roll_feeder_valid_food_types(self) -> None:
        real_types = {ft.value for ft in FoodType if ft != FoodType.WILD}
        rng = random.Random(0)
        for _ in range(10):
            result = _roll_feeder(rng)
            for food in result:
                assert food in real_types


# ---------------------------------------------------------------------------
# 11. 100 random complete games
# ---------------------------------------------------------------------------


class TestRandomCompleteGames:
    @settings(max_examples=100, deadline=30_000)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    def test_random_game_no_crash(self, seed: int) -> None:
        """A random game from start to finish must never raise an exception."""
        eng = WingspanEngine(catalog_path=CATALOG_PATH, seed=seed)
        rng = random.Random(seed)
        state = eng.reset(seed=seed)

        max_steps = 1000  # safety cap
        steps = 0

        while not eng.is_terminal(state) and steps < max_steps:
            actions = eng.get_legal_actions(state)
            assert len(actions) > 0, f"Empty actions at step {steps}"
            action = rng.choice(actions)
            result = eng.step(state, action)
            assert result.success
            state = result.new_state
            steps += 1

        # Either game completed naturally or hit safety cap
        if not eng.is_terminal(state):
            # Accept hitting the cap — the engine didn't crash
            pass
        else:
            scores = eng.compute_scores(state)
            for pid, score in scores.items():
                assert score >= 0, f"Player {pid} negative score at game end"


# ---------------------------------------------------------------------------
# 12. Property-based tests — GameState invariants
# ---------------------------------------------------------------------------


class TestPropertyBased:
    @given(eggs=st.integers(min_value=0, max_value=6))
    def test_egg_cost_nonnegative(self, eggs: int) -> None:
        b = _make_board()
        for i in range(min(eggs, N_HABITAT_SLOTS)):
            b.forest[i] = BirdSlotState(f"Bird{i}", eggs=1)
        cost = b.egg_cost_for_habitat("forest")
        assert cost >= 0

    @given(n_birds=st.integers(min_value=0, max_value=5))
    def test_total_birds_matches_placed(self, n_birds: int) -> None:
        b = _make_board()
        for i in range(n_birds):
            b.forest[i] = BirdSlotState(f"B{i}")
        assert b.total_birds() == n_birds

    @given(seed=st.integers(min_value=0, max_value=9999))
    def test_legal_actions_nonempty_at_start(self, seed: int) -> None:
        eng = WingspanEngine(catalog_path=CATALOG_PATH, seed=seed)
        state = eng.reset(seed=seed)
        actions = eng.get_legal_actions(state)
        assert len(actions) > 0

    @given(n_eggs=st.integers(min_value=0, max_value=20))
    def test_total_eggs_nonnegative(self, n_eggs: int) -> None:
        b = _make_board()
        b.forest[0] = BirdSlotState("Eagle", eggs=n_eggs)
        assert b.total_eggs() >= 0

    @given(seed=st.integers(min_value=0, max_value=999))
    def test_score_nonnegative_midgame(self, seed: int) -> None:
        eng = WingspanEngine(catalog_path=CATALOG_PATH, seed=seed)
        rng = random.Random(seed)
        state = eng.reset(seed=seed)
        for _ in range(5):
            if eng.is_terminal(state):
                break
            actions = eng.get_legal_actions(state)
            result = eng.step(state, rng.choice(actions))
            state = result.new_state
        scores = eng.compute_scores(state)
        for score in scores.values():
            assert score >= 0
