"""S0.6 — Smoke tests: all modules importable without error."""

from __future__ import annotations


def test_import_game_state() -> None:
    from src.games.base.game_state import Action, ActionResult, GameState, PlayerState

    assert Action is not None
    assert ActionResult is not None
    assert GameState is not None
    assert PlayerState is not None


def test_import_game_engine() -> None:
    from src.games.base.engine import GameEngine

    assert GameEngine is not None


def test_import_wingspan_state() -> None:
    from src.games.wingspan.cards import FoodType, Habitat
    from src.games.wingspan.state import WingspanState

    assert Habitat.FOREST == "forest"
    assert FoodType.SEED == "seed"


def test_import_wingspan_actions() -> None:
    from src.games.wingspan.actions import WingspanAction, WingspanActionType

    assert WingspanActionType.PLAY_BIRD == "play_bird"


def test_import_wingspan_engine() -> None:
    from src.games.wingspan.engine import WingspanEngine

    assert WingspanEngine is not None


def test_import_wingspan_cards() -> None:
    from src.games.wingspan.cards import BirdCard, load_bird_catalog

    assert BirdCard is not None


def test_import_wingspan_rules() -> None:
    from src.games.wingspan.rules import LegalMoveValidator

    assert LegalMoveValidator is not None


def test_import_wingspan_rewards() -> None:
    from src.games.wingspan.rewards import compute_reward

    assert compute_reward is not None


def test_import_oracle_client() -> None:
    from src.oracle.claude_client import ClaudeClient, DEFAULT_MODEL

    assert DEFAULT_MODEL == "claude-sonnet-4-6"


def test_import_oracle_ingestion() -> None:
    from src.oracle.ingestion import RulebookIngester, RuleChunk

    assert RulebookIngester is not None


def test_import_oracle_retriever() -> None:
    from src.oracle.retriever import RuleRetriever, RetrievedChunk

    assert RuleRetriever is not None


def test_import_oracle_rule_oracle() -> None:
    from src.oracle.rule_oracle import RuleOracle, RuleAnswer, ValidationResult

    assert RuleOracle is not None


def test_import_envs() -> None:
    from src.envs.wingspan_env import WingspanEnv, N_MAX_ACTIONS

    assert N_MAX_ACTIONS >= 70  # 5+1+4+MAX_HAND_SIZE*3; tighter than 256 placeholder


def test_import_agents() -> None:
    from src.agents.baselines import RandomAgent, GreedyAgent, BaseAgent
    from src.agents.bc_agent import BehavioralCloningTrainer
    from src.agents.ppo_agent import build_maskable_ppo

    assert RandomAgent is not None


def test_import_imitation() -> None:
    from src.imitation.demo_buffer import DemonstrationBuffer, Transition
    from src.imitation.bga_parser import BGALogParser
    from src.imitation.tts_parser import TTSLogParser

    assert DemonstrationBuffer is not None


def test_import_eval() -> None:
    from src.eval.metrics import win_rate, avg_score
    from src.eval.tournament import Tournament, EloTable
    from src.eval.llm_judge import LLMJudge, PlayQualityReport

    assert Tournament is not None


def test_action_dataclass() -> None:
    """Action is constructable with defaults."""
    from src.games.base.game_state import Action

    a = Action(action_type="gain_food", params={"food": "seed"}, player_id=0)
    assert a.action_type == "gain_food"
    assert a.params["food"] == "seed"
    d = a.to_dict()
    assert d["action_type"] == "gain_food"


def test_player_state_defaults() -> None:
    """PlayerState initialises with empty hand and zero score."""
    from src.games.base.game_state import PlayerState

    ps = PlayerState(player_id=0)
    assert ps.score == 0
    assert ps.hand == []
    assert ps.resources == {}


def test_random_agent_selects_legal_action() -> None:
    """RandomAgent returns one of the provided legal actions."""
    from src.agents.baselines import RandomAgent
    from src.games.base.game_state import Action, PlayerState

    class _DummyState:
        player_id = 0
        turn = 1
        phase = "main"
        players = [PlayerState(player_id=0)]
        shared_board = {}

    legal = [Action("gain_food"), Action("lay_eggs"), Action("draw_cards")]
    agent = RandomAgent(seed=42)
    chosen = agent.select_action(_DummyState(), legal)  # type: ignore[arg-type]
    assert chosen in legal


def test_demo_buffer_len() -> None:
    """DemonstrationBuffer tracks transition count."""
    from src.imitation.demo_buffer import DemonstrationBuffer, Transition

    buf = DemonstrationBuffer()
    assert len(buf) == 0
    buf.add_game([Transition(obs={}, action=0, next_obs={}, reward=0.0, done=False)])
    assert len(buf) == 1
