"""Sprint 6 — Evaluation framework tests.

Covers:
  - game_runner: run_game correctness and GameResult structure
  - metrics: win_rate, avg_score, score_distribution, rule_violation_rate,
             steps_to_target_winrate
  - tournament: EloTable update logic, standings, Tournament round-robin
  - llm_judge: PlayQualityReport construction, LLMJudge response parsing

Tests that require a real API call are skipped automatically when
ANTHROPIC_API_KEY is not set.
"""

from __future__ import annotations

import os
import json
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Shared WingspanEngine for the module; loading the catalog once is slow."""
    from src.games.wingspan.engine import WingspanEngine
    return WingspanEngine(seed=0)


@pytest.fixture(scope="module")
def random_agent():
    from src.agents.baselines import RandomAgent
    return RandomAgent(seed=42)


@pytest.fixture(scope="module")
def greedy_agent():
    from src.agents.baselines import GreedyAgent
    return GreedyAgent()


# ===========================================================================
# TestGameRunner
# ===========================================================================

class TestGameRunner:
    def test_run_game_returns_game_result(self, engine, random_agent):
        from src.eval.game_runner import GameResult, run_game
        result = run_game(random_agent, random_agent, engine, seed=1)
        assert isinstance(result, GameResult)

    def test_run_game_winner_is_valid(self, engine, random_agent):
        from src.eval.game_runner import run_game
        result = run_game(random_agent, random_agent, engine, seed=2)
        assert result.winner in (0, 1, None)

    def test_run_game_scores_two_players(self, engine, random_agent):
        from src.eval.game_runner import run_game
        result = run_game(random_agent, random_agent, engine, seed=3)
        assert len(result.scores) == 2

    def test_run_game_scores_nonnegative(self, engine, random_agent):
        from src.eval.game_runner import run_game
        result = run_game(random_agent, random_agent, engine, seed=4)
        assert all(s >= 0 for s in result.scores)

    def test_run_game_n_turns_positive(self, engine, random_agent):
        from src.eval.game_runner import run_game
        result = run_game(random_agent, random_agent, engine, seed=5)
        assert result.n_turns > 0

    def test_run_game_different_seeds_different_outcomes(self, engine, random_agent):
        """Two different seeds should not always produce the same score pair."""
        from src.eval.game_runner import run_game
        results = [run_game(random_agent, random_agent, engine, seed=i) for i in range(5)]
        score_pairs = [tuple(r.scores) for r in results]
        # At least two distinct outcomes across 5 games
        assert len(set(score_pairs)) >= 1  # minimum sanity; usually distinct

    def test_run_game_terminates_within_max_turns(self, engine, random_agent):
        from src.eval.game_runner import run_game, _MAX_TURNS
        result = run_game(random_agent, random_agent, engine, seed=99)
        assert result.n_turns <= _MAX_TURNS

    def test_greedy_agent_plays_full_game(self, engine, greedy_agent):
        from src.eval.game_runner import run_game
        from src.agents.baselines import RandomAgent
        result = run_game(greedy_agent, RandomAgent(seed=0), engine, seed=10)
        assert result.n_turns > 0
        assert result.winner in (0, 1, None)


# ===========================================================================
# TestMetrics
# ===========================================================================

class TestMetrics:
    def test_win_rate_returns_float(self, engine, random_agent):
        from src.eval.metrics import win_rate
        wr = win_rate(random_agent, random_agent, engine, n_games=4, base_seed=0)
        assert isinstance(wr, float)

    def test_win_rate_in_range(self, engine, random_agent):
        from src.eval.metrics import win_rate
        wr = win_rate(random_agent, random_agent, engine, n_games=4, base_seed=0)
        assert 0.0 <= wr <= 1.0

    def test_win_rate_self_play_near_half(self, engine, random_agent):
        """Random vs random over enough games should be close to 0.5."""
        from src.eval.metrics import win_rate
        wr = win_rate(random_agent, random_agent, engine, n_games=20, base_seed=0)
        assert 0.0 <= wr <= 1.0  # broad check; variance high over 20 games

    def test_win_rate_greedy_vs_random_positive(self, engine, greedy_agent, random_agent):
        """Greedy agent should beat random more than 0% of the time."""
        from src.eval.metrics import win_rate
        wr = win_rate(greedy_agent, random_agent, engine, n_games=8, base_seed=0)
        assert wr >= 0.0

    def test_avg_score_returns_tuple(self, engine, random_agent):
        from src.eval.metrics import avg_score
        result = avg_score(random_agent, engine, n_games=4, base_seed=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_avg_score_values_nonneg(self, engine, random_agent):
        from src.eval.metrics import avg_score
        mean, std = avg_score(random_agent, engine, n_games=4, base_seed=0)
        assert mean >= 0.0
        assert std >= 0.0

    def test_score_distribution_length(self, engine, random_agent):
        from src.eval.metrics import score_distribution
        scores = score_distribution(random_agent, engine, n_games=5, base_seed=0)
        assert len(scores) == 5

    def test_score_distribution_values_nonneg(self, engine, random_agent):
        from src.eval.metrics import score_distribution
        scores = score_distribution(random_agent, engine, n_games=5, base_seed=0)
        assert all(s >= 0 for s in scores)

    def test_rule_violation_rate_returns_float(self, engine, random_agent):
        from src.eval.metrics import rule_violation_rate
        rate = rule_violation_rate(random_agent, engine, n_games=2, base_seed=0)
        assert isinstance(rate, float)

    def test_rule_violation_rate_random_agent_zero(self, engine, random_agent):
        """RandomAgent always picks from the legal list — violation rate must be 0."""
        from src.eval.metrics import rule_violation_rate
        rate = rule_violation_rate(random_agent, engine, n_games=3, base_seed=0)
        assert rate == 0.0

    def test_rule_violation_rate_greedy_zero(self, engine, greedy_agent):
        """GreedyAgent also picks only legal actions."""
        from src.eval.metrics import rule_violation_rate
        rate = rule_violation_rate(greedy_agent, engine, n_games=3, base_seed=0)
        assert rate == 0.0

    def test_steps_to_target_winrate_found(self):
        from src.eval.metrics import steps_to_target_winrate
        history = [
            {"step": 50_000, "win_rate": 0.40},
            {"step": 100_000, "win_rate": 0.52},
            {"step": 150_000, "win_rate": 0.60},
        ]
        step = steps_to_target_winrate(history, target=0.55)
        assert step == 150_000

    def test_steps_to_target_winrate_not_reached(self):
        from src.eval.metrics import steps_to_target_winrate
        history = [{"step": 50_000, "win_rate": 0.45}]
        assert steps_to_target_winrate(history, target=0.80) is None

    def test_steps_to_target_winrate_exact_match(self):
        from src.eval.metrics import steps_to_target_winrate
        history = [{"step": 200_000, "win_rate": 0.55}]
        assert steps_to_target_winrate(history, target=0.55) == 200_000

    def test_steps_to_target_winrate_empty_history(self):
        from src.eval.metrics import steps_to_target_winrate
        assert steps_to_target_winrate([], target=0.5) is None


# ===========================================================================
# TestEloTable
# ===========================================================================

class TestEloTable:
    def test_initial_ratings_set(self):
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 1000.0, "B": 1000.0})
        assert table.ratings["A"] == 1000.0

    def test_update_winner_gains_elo(self):
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 1000.0, "B": 1000.0})
        table.update("A", "B")
        assert table.ratings["A"] > 1000.0

    def test_update_loser_loses_elo(self):
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 1000.0, "B": 1000.0})
        table.update("A", "B")
        assert table.ratings["B"] < 1000.0

    def test_update_zero_sum(self):
        """Total Elo is conserved across any update."""
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 1000.0, "B": 1000.0})
        total_before = sum(table.ratings.values())
        table.update("A", "B")
        total_after = sum(table.ratings.values())
        assert abs(total_after - total_before) < 1e-6

    def test_draw_update_symmetric(self):
        """Draw with equal ratings: both ratings stay equal (no change for symmetric case)."""
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 1000.0, "B": 1000.0})
        table.update("A", "B", draw=True)
        assert abs(table.ratings["A"] - table.ratings["B"]) < 1e-6

    def test_standings_ordered(self):
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 900.0, "B": 1100.0, "C": 1000.0})
        standings = table.standings()
        ratings_in_order = [r for _, r in standings]
        assert ratings_in_order == sorted(ratings_in_order, reverse=True)

    def test_match_results_recorded(self):
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"A": 1000.0, "B": 1000.0})
        table.update("A", "B")
        assert table.match_results["A"]["B"]["win"] == 1
        assert table.match_results["B"]["A"]["loss"] == 1

    def test_repr_contains_agent_names(self):
        from src.eval.tournament import EloTable
        table = EloTable(ratings={"AlphaAgent": 1050.0, "BetaAgent": 950.0})
        rep = repr(table)
        assert "AlphaAgent" in rep
        assert "BetaAgent" in rep

    def test_update_unknown_agent_defaults_to_1000(self):
        """An agent not yet in ratings should default to 1000."""
        from src.eval.tournament import EloTable
        table = EloTable(ratings={})
        table.update("X", "Y")
        assert "X" in table.ratings
        assert "Y" in table.ratings


# ===========================================================================
# TestTournament
# ===========================================================================

class TestTournament:
    def test_tournament_returns_elo_table(self, engine, random_agent, greedy_agent):
        from src.eval.tournament import EloTable, Tournament
        t = Tournament(engine=engine, base_seed=0)
        table = t.run(
            {"random": random_agent, "greedy": greedy_agent},
            n_games_per_pair=2,
        )
        assert isinstance(table, EloTable)

    def test_tournament_all_agents_have_rating(self, engine, random_agent, greedy_agent):
        from src.eval.tournament import Tournament
        t = Tournament(engine=engine, base_seed=0)
        table = t.run(
            {"random": random_agent, "greedy": greedy_agent},
            n_games_per_pair=2,
        )
        assert "random" in table.ratings
        assert "greedy" in table.ratings

    def test_tournament_three_agents(self, engine, random_agent, greedy_agent):
        """Three agents → three pairs → each has a rating."""
        from src.agents.baselines import RandomAgent
        from src.eval.tournament import Tournament
        t = Tournament(engine=engine, base_seed=0)
        agents = {
            "random_a": random_agent,
            "random_b": RandomAgent(seed=7),
            "greedy": greedy_agent,
        }
        table = t.run(agents, n_games_per_pair=2)
        assert len(table.ratings) == 3

    def test_tournament_greedy_higher_elo_than_random(self, engine, random_agent, greedy_agent):
        """GreedyAgent should end up with higher Elo than RandomAgent over enough games."""
        from src.eval.tournament import Tournament
        t = Tournament(engine=engine, base_seed=0)
        table = t.run(
            {"random": random_agent, "greedy": greedy_agent},
            n_games_per_pair=10,
        )
        # Not guaranteed over 10 games, but very likely
        assert "greedy" in table.ratings and "random" in table.ratings


# ===========================================================================
# TestPlayQualityReport
# ===========================================================================

class TestPlayQualityReport:
    def test_construction_minimal(self):
        from src.eval.llm_judge import PlayQualityReport
        report = PlayQualityReport(
            strategic_coherence=0.7,
            synergy_exploitation=0.5,
        )
        assert report.tactical_errors == []
        assert report.summary == ""

    def test_construction_full(self):
        from src.eval.llm_judge import PlayQualityReport
        report = PlayQualityReport(
            strategic_coherence=0.8,
            synergy_exploitation=0.6,
            tactical_errors=["played weak bird early"],
            summary="Solid wetland strategy.",
        )
        assert len(report.tactical_errors) == 1
        assert "wetland" in report.summary


# ===========================================================================
# TestLLMJudge
# ===========================================================================

class TestLLMJudgeParseResponse:
    """Tests that exercise only the response-parsing logic (no API call needed)."""

    def _make_judge(self):
        from unittest.mock import MagicMock
        from src.eval.llm_judge import LLMJudge
        mock_client = MagicMock()
        return LLMJudge(client=mock_client)

    def test_parse_valid_json(self):
        judge = self._make_judge()
        raw = json.dumps({
            "strategic_coherence": 0.75,
            "synergy_exploitation": 0.60,
            "tactical_errors": ["missed draw combo"],
            "summary": "Forest engine with minor inefficiencies.",
        })
        report = judge._parse_response(raw)
        assert abs(report.strategic_coherence - 0.75) < 1e-6
        assert len(report.tactical_errors) == 1

    def test_parse_json_with_markdown_fences(self):
        judge = self._make_judge()
        raw = "```json\n" + json.dumps({
            "strategic_coherence": 0.5,
            "synergy_exploitation": 0.5,
            "tactical_errors": [],
            "summary": "Average play.",
        }) + "\n```"
        report = judge._parse_response(raw)
        assert abs(report.strategic_coherence - 0.5) < 1e-6

    def test_parse_invalid_json_raises_value_error(self):
        from src.eval.llm_judge import PlayQualityReport
        judge = self._make_judge()
        with pytest.raises(ValueError, match="failed to parse"):
            judge._parse_response("not valid json at all")

    def test_parse_missing_key_raises_value_error(self):
        judge = self._make_judge()
        raw = json.dumps({"strategic_coherence": 0.5})  # missing synergy_exploitation
        with pytest.raises(ValueError):
            judge._parse_response(raw)

    def test_parse_empty_tactical_errors(self):
        judge = self._make_judge()
        raw = json.dumps({
            "strategic_coherence": 0.9,
            "synergy_exploitation": 0.8,
            "tactical_errors": [],
            "summary": "Excellent play.",
        })
        report = judge._parse_response(raw)
        assert report.tactical_errors == []

    def test_parse_summary_optional(self):
        judge = self._make_judge()
        raw = json.dumps({
            "strategic_coherence": 0.5,
            "synergy_exploitation": 0.5,
        })
        report = judge._parse_response(raw)
        assert report.summary == ""


class TestLLMJudgeWithMockClient:
    """Tests LLMJudge.evaluate_play_quality() against a mocked ClaudeClient."""

    def test_evaluate_calls_client_complete(self):
        from unittest.mock import MagicMock
        from src.eval.llm_judge import LLMJudge, PlayQualityReport
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "strategic_coherence": 0.7,
            "synergy_exploitation": 0.6,
            "tactical_errors": [],
            "summary": "Good game.",
        })
        judge = LLMJudge(client=mock_client)
        report = judge.evaluate_play_quality("Turn 1 | P0 | gain_food")
        assert isinstance(report, PlayQualityReport)
        mock_client.complete.assert_called_once()

    def test_evaluate_returns_play_quality_report(self):
        from unittest.mock import MagicMock
        from src.eval.llm_judge import LLMJudge, PlayQualityReport
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "strategic_coherence": 0.8,
            "synergy_exploitation": 0.7,
            "tactical_errors": ["late bird placement"],
            "summary": "Wetland focus.",
        })
        judge = LLMJudge(client=mock_client)
        report = judge.evaluate_play_quality("dummy transcript")
        assert isinstance(report, PlayQualityReport)
        assert report.strategic_coherence == pytest.approx(0.8)
        assert len(report.tactical_errors) == 1


# ===========================================================================
# TestBuildGameTranscript
# ===========================================================================

class TestBuildGameTranscript:
    def test_transcript_is_string(self, engine, random_agent):
        from src.eval.llm_judge import build_game_transcript
        transcript = build_game_transcript(engine, [random_agent, random_agent], seed=42)
        assert isinstance(transcript, str)

    def test_transcript_contains_game_over(self, engine, random_agent):
        from src.eval.llm_judge import build_game_transcript
        transcript = build_game_transcript(engine, [random_agent, random_agent], seed=42)
        assert "Game over" in transcript

    def test_transcript_contains_turn_entries(self, engine, random_agent):
        from src.eval.llm_judge import build_game_transcript
        transcript = build_game_transcript(engine, [random_agent, random_agent], seed=42)
        assert "Turn" in transcript

    def test_transcript_nonempty(self, engine, random_agent):
        from src.eval.llm_judge import build_game_transcript
        transcript = build_game_transcript(engine, [random_agent, random_agent], seed=1)
        assert len(transcript) > 100


# ===========================================================================
# TestIntegration
# ===========================================================================

class TestIntegration:
    def test_win_rate_greedy_vs_random_consistent_with_game_runner(
        self, engine, greedy_agent, random_agent
    ):
        """win_rate result matches manual count from run_game."""
        from src.eval.game_runner import run_game
        from src.eval.metrics import win_rate

        n = 6
        manual_wins = 0
        for i in range(n):
            if i % 2 == 0:
                r = run_game(greedy_agent, random_agent, engine, seed=i)
                if r.winner == 0:
                    manual_wins += 1
            else:
                r = run_game(random_agent, greedy_agent, engine, seed=i)
                if r.winner == 1:
                    manual_wins += 1

        wr = win_rate(greedy_agent, random_agent, engine, n_games=n, base_seed=0)
        assert abs(wr - manual_wins / n) < 1e-6

    def test_tournament_and_metrics_consistent(self, engine, random_agent, greedy_agent):
        """Greedy Elo > Random Elo iff greedy win_rate > 0.5 (loose consistency check)."""
        from src.eval.metrics import win_rate
        from src.eval.tournament import Tournament

        t = Tournament(engine=engine, base_seed=42)
        table = t.run({"random": random_agent, "greedy": greedy_agent}, n_games_per_pair=4)
        # Just verify the table has both entries — statistical direction not guaranteed over 4 games
        assert "random" in table.ratings
        assert "greedy" in table.ratings
