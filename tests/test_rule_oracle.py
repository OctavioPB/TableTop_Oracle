"""S1 — Rule Oracle, ingestion, and retriever tests.

Unit tests run without a PDF or API key (mocked dependencies).
Integration tests are skipped unless ChromaDB is populated and ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.oracle.ingestion import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    RuleChunk,
    RulebookIngester,
)
from src.oracle.retriever import RetrievedChunk, RuleRetriever
from src.oracle.rule_oracle import RuleAnswer, RuleOracle, ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ingester(tmp_path: Path) -> RulebookIngester:
    return RulebookIngester(chroma_persist_dir=tmp_path / "chroma")


@pytest.fixture
def sample_pages() -> list[tuple[int, str]]:
    """Synthetic rulebook pages for chunking tests."""
    return [
        (
            1,
            "TURN STRUCTURE\n\n"
            "On your turn you must take exactly one of four actions: "
            "gain food, lay eggs, draw bird cards, or play a bird. "
            "You indicate your chosen action by placing one of your action cubes "
            "on the leftmost exposed space of that action's row on your player mat.\n\n"
            "GAIN FOOD\n\n"
            "When you choose this action you gain food from the bird feeder dice. "
            "Take any 1 die from the bird feeder and gain the food shown. "
            "Then activate the brown powers on all birds in your forest habitat, "
            "from right to left.",
        ),
        (
            2,
            "LAY EGGS\n\n"
            "When you choose this action, gain 2 eggs from the supply, "
            "plus 1 additional egg per bird in your grassland. "
            "Place eggs on any birds that have room under their egg limit. "
            "For example, if you have 3 birds in your grassland, you lay 5 eggs total.\n\n"
            "DRAW BIRD CARDS\n\n"
            "When you choose this action, draw 1 card from the deck or face-up tray, "
            "plus 1 additional card per bird in your wetland. "
            "Exception: if the deck is empty, you may only draw from the tray.",
        ),
    ]


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="On your turn you must take exactly one of four actions.",
        page=1,
        section="Turn Structure",
        game="wingspan",
        chunk_type="rule",
        score=0.92,
        chunk_id="abc123",
    )


@pytest.fixture
def mock_retriever(sample_chunk: RetrievedChunk) -> MagicMock:
    retriever = MagicMock(spec=RuleRetriever)
    retriever.query.return_value = [sample_chunk]
    retriever.query_with_filter.return_value = [sample_chunk]
    return retriever


@pytest.fixture
def mock_client_qa() -> MagicMock:
    client = MagicMock()
    client.complete_json.return_value = {
        "answer": "You must take exactly one of four actions on your turn.",
        "confidence": 0.95,
        "sources": ["page 1 — Turn Structure"],
        "verbatim_quotes": ["you must take exactly one of four actions"],
    }
    return client


@pytest.fixture
def mock_client_validation() -> MagicMock:
    client = MagicMock()
    client.complete_json.return_value = {
        "is_legal": True,
        "reason": "Gaining food is always a legal action on your turn.",
        "rule_quote": "On your turn you must take exactly one of four actions.",
        "confidence": 0.98,
    }
    return client


@pytest.fixture
def oracle_qa(mock_client_qa: MagicMock, mock_retriever: MagicMock) -> RuleOracle:
    return RuleOracle(
        client=mock_client_qa,
        retriever=mock_retriever,
        prompt_dir=Path("src/oracle/prompts"),
    )


@pytest.fixture
def oracle_validation(
    mock_client_validation: MagicMock, mock_retriever: MagicMock
) -> RuleOracle:
    return RuleOracle(
        client=mock_client_validation,
        retriever=mock_retriever,
        prompt_dir=Path("src/oracle/prompts"),
    )


# ---------------------------------------------------------------------------
# Ingestion unit tests (no PDF, no ChromaDB required)
# ---------------------------------------------------------------------------


class TestRulebookIngesterChunking:
    def test_chunk_text_produces_non_empty_chunks(
        self, ingester: RulebookIngester, sample_pages: list
    ) -> None:
        chunks = ingester._chunk_text(sample_pages, "wingspan")
        assert len(chunks) > 0

    def test_chunk_text_assigns_correct_game(
        self, ingester: RulebookIngester, sample_pages: list
    ) -> None:
        chunks = ingester._chunk_text(sample_pages, "wingspan")
        assert all(c.game == "wingspan" for c in chunks)

    def test_chunk_text_size_within_bounds(
        self, ingester: RulebookIngester, sample_pages: list
    ) -> None:
        chunks = ingester._chunk_text(sample_pages, "wingspan")
        for chunk in chunks[:-1]:  # last chunk may be smaller
            word_count = len(chunk.text.split())
            assert word_count <= CHUNK_SIZE_WORDS + 5, (
                f"Chunk too large: {word_count} words"
            )

    def test_chunk_ids_are_unique(
        self, ingester: RulebookIngester, sample_pages: list
    ) -> None:
        chunks = ingester._chunk_text(sample_pages, "wingspan")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs"

    def test_chunk_page_numbers_are_valid(
        self, ingester: RulebookIngester, sample_pages: list
    ) -> None:
        chunks = ingester._chunk_text(sample_pages, "wingspan")
        valid_pages = {p for p, _ in sample_pages}
        for chunk in chunks:
            assert chunk.page in valid_pages

    def test_chunk_type_detection_rule(self, ingester: RulebookIngester) -> None:
        text = "You must take exactly one action on your turn."
        assert ingester._detect_chunk_type(text) == "rule"

    def test_chunk_type_detection_example(self, ingester: RulebookIngester) -> None:
        text = "For example, if you have 3 birds in your grassland, you lay 5 eggs."
        assert ingester._detect_chunk_type(text) == "example"

    def test_chunk_type_detection_exception(self, ingester: RulebookIngester) -> None:
        text = "Exception: if the deck is empty, you may only draw from the tray."
        assert ingester._detect_chunk_type(text) == "exception"

    def test_chunk_type_detection_card_power(self, ingester: RulebookIngester) -> None:
        text = "When played: gain 1 food from the bird feeder."
        assert ingester._detect_chunk_type(text) == "card_power"

    def test_chunk_type_detection_card_power_once_between_turns(
        self, ingester: RulebookIngester
    ) -> None:
        text = "Once between turns: tuck 1 card from the deck."
        assert ingester._detect_chunk_type(text) == "card_power"

    def test_detect_section_returns_empty_for_normal_text(
        self, ingester: RulebookIngester
    ) -> None:
        text = "When you choose this action you gain food from the bird feeder dice."
        assert ingester._detect_section(text) == ""

    def test_detect_section_detects_all_caps_heading(
        self, ingester: RulebookIngester
    ) -> None:
        heading = "TURN STRUCTURE"
        result = ingester._detect_section(heading)
        assert result != ""

    def test_detect_section_ignores_long_text(self, ingester: RulebookIngester) -> None:
        long_text = "A" * 100
        assert ingester._detect_section(long_text) == ""

    def test_extract_segments_preserves_section_context(
        self, ingester: RulebookIngester, sample_pages: list
    ) -> None:
        segments = ingester._extract_segments(sample_pages)
        sections = {s for _, s, _ in segments}
        # Should detect at least one section heading from the synthetic pages
        assert len(sections) >= 1


class TestRulebookIngesterIngest:
    def test_ingest_raises_on_missing_pdf(
        self, ingester: RulebookIngester, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            ingester.ingest(tmp_path / "nonexistent.pdf", "wingspan")


# ---------------------------------------------------------------------------
# Retriever unit tests
# ---------------------------------------------------------------------------


class TestRuleRetriever:
    def test_retriever_raises_on_missing_collection(self, tmp_path: Path) -> None:
        retriever = RuleRetriever(chroma_persist_dir=tmp_path / "chroma")
        with pytest.raises(RuntimeError, match="not found"):
            retriever.query("How do I gain food?", "wingspan")

    def test_collection_exists_returns_false_for_missing(self, tmp_path: Path) -> None:
        retriever = RuleRetriever(chroma_persist_dir=tmp_path / "chroma")
        assert retriever.collection_exists("wingspan") is False

    def test_parse_results_handles_empty(self) -> None:
        empty = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        result = RuleRetriever._parse_results(empty, "wingspan")
        assert result == []

    def test_parse_results_converts_distance_to_score(self) -> None:
        raw = {
            "documents": [["Some rule text"]],
            "metadatas": [[{"page": 5, "section": "S", "game": "wingspan", "chunk_type": "rule"}]],
            "distances": [[0.2]],
        }
        chunks = RuleRetriever._parse_results(raw, "wingspan")
        assert len(chunks) == 1
        assert abs(chunks[0].score - 0.8) < 1e-6  # 1.0 - 0.2

    def test_retrieved_chunk_fields(self, sample_chunk: RetrievedChunk) -> None:
        assert sample_chunk.game == "wingspan"
        assert 0.0 <= sample_chunk.score <= 1.0
        assert sample_chunk.chunk_type in ("rule", "example", "exception", "card_power")


# ---------------------------------------------------------------------------
# RuleOracle unit tests (mocked client and retriever)
# ---------------------------------------------------------------------------


class TestRuleOracleAnswerQuestion:
    def test_answer_rule_question_returns_rule_answer(
        self, oracle_qa: RuleOracle
    ) -> None:
        result = oracle_qa.answer_rule_question("What actions can I take?", "wingspan")
        assert isinstance(result, RuleAnswer)

    def test_answer_rule_question_answer_is_string(
        self, oracle_qa: RuleOracle
    ) -> None:
        result = oracle_qa.answer_rule_question("How many actions per turn?", "wingspan")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_answer_rule_question_confidence_is_float_in_range(
        self, oracle_qa: RuleOracle
    ) -> None:
        result = oracle_qa.answer_rule_question("What is the egg limit?", "wingspan")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_answer_rule_question_sources_is_list(
        self, oracle_qa: RuleOracle
    ) -> None:
        result = oracle_qa.answer_rule_question("When does the round end?", "wingspan")
        assert isinstance(result.sources, list)

    def test_answer_rule_question_calls_retriever(
        self, oracle_qa: RuleOracle, mock_retriever: MagicMock
    ) -> None:
        oracle_qa.answer_rule_question("Some question", "wingspan")
        mock_retriever.query.assert_called_once()

    def test_answer_rule_question_calls_client(
        self, oracle_qa: RuleOracle, mock_client_qa: MagicMock
    ) -> None:
        oracle_qa.answer_rule_question("Some question", "wingspan")
        mock_client_qa.complete_json.assert_called_once()

    def test_answer_rule_question_no_chunks_returns_graceful(
        self, mock_client_qa: MagicMock
    ) -> None:
        empty_retriever = MagicMock(spec=RuleRetriever)
        empty_retriever.query.return_value = []
        oracle = RuleOracle(
            client=mock_client_qa,
            retriever=empty_retriever,
            prompt_dir=Path("src/oracle/prompts"),
        )
        result = oracle.answer_rule_question("Anything", "wingspan")
        assert isinstance(result, RuleAnswer)
        assert result.confidence == 0.0
        # Client should NOT be called when there are no chunks
        mock_client_qa.complete_json.assert_not_called()


class TestRuleOracleValidateAction:
    def test_validate_action_returns_validation_result(
        self, oracle_validation: RuleOracle
    ) -> None:
        from src.games.base.game_state import Action
        from src.games.wingspan.state import WingspanState

        state = WingspanState(player_id=0, turn=1, phase="main")
        action = Action(action_type="gain_food", params={"food": "seed"})
        result = oracle_validation.validate_action(state, action, "wingspan")
        assert isinstance(result, ValidationResult)

    def test_validate_action_is_legal_is_bool(
        self, oracle_validation: RuleOracle
    ) -> None:
        from src.games.base.game_state import Action
        from src.games.wingspan.state import WingspanState

        state = WingspanState(player_id=0, turn=1, phase="main")
        action = Action(action_type="gain_food")
        result = oracle_validation.validate_action(state, action, "wingspan")
        assert isinstance(result.is_legal, bool)

    def test_validate_action_confidence_in_range(
        self, oracle_validation: RuleOracle
    ) -> None:
        from src.games.base.game_state import Action
        from src.games.wingspan.state import WingspanState

        state = WingspanState(player_id=0, turn=1, phase="main")
        action = Action(action_type="lay_eggs")
        result = oracle_validation.validate_action(state, action, "wingspan")
        assert 0.0 <= result.confidence <= 1.0


class TestRuleOracleResolveConflict:
    def test_resolve_conflict_returns_string(self) -> None:
        client = MagicMock()
        client.complete_json.return_value = {
            "ruling": "Card text overrides general rules",
            "reason": "Specific always beats general in Wingspan.",
            "confidence": 0.9,
        }
        retriever = MagicMock(spec=RuleRetriever)
        oracle = RuleOracle(
            client=client,
            retriever=retriever,
            prompt_dir=Path("src/oracle/prompts"),
        )
        result = oracle.resolve_conflict("Rule A text", "Rule B text")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Prompt loading tests
# ---------------------------------------------------------------------------


class TestPromptLoading:
    def test_load_rule_question_prompt_exists(self) -> None:
        oracle = RuleOracle(
            client=MagicMock(), retriever=MagicMock(),
            prompt_dir=Path("src/oracle/prompts"),
        )
        template = oracle._load_prompt("rule_question.txt")
        assert "{question}" in template
        assert "{retrieved_chunks}" in template
        assert "{game}" in template

    def test_load_rule_validator_prompt_exists(self) -> None:
        oracle = RuleOracle(
            client=MagicMock(), retriever=MagicMock(),
            prompt_dir=Path("src/oracle/prompts"),
        )
        template = oracle._load_prompt("rule_validator.txt")
        assert "{game_state_json}" in template
        assert "{action_json}" in template

    def test_load_missing_prompt_raises_file_not_found(self) -> None:
        oracle = RuleOracle(
            client=MagicMock(), retriever=MagicMock(),
            prompt_dir=Path("src/oracle/prompts"),
        )
        with pytest.raises(FileNotFoundError):
            oracle._load_prompt("nonexistent_prompt.txt")

    def test_prompt_caching(self) -> None:
        oracle = RuleOracle(
            client=MagicMock(), retriever=MagicMock(),
            prompt_dir=Path("src/oracle/prompts"),
        )
        t1 = oracle._load_prompt("rule_question.txt")
        t2 = oracle._load_prompt("rule_question.txt")
        assert t1 is t2  # same object due to caching


# ---------------------------------------------------------------------------
# Golden dataset structure validation
# ---------------------------------------------------------------------------


class TestGoldenDataset:
    def test_golden_dataset_exists(self) -> None:
        path = Path("data/golden_rules/wingspan_rules_qa.json")
        assert path.exists(), f"Golden dataset not found: {path}"

    def test_golden_dataset_has_minimum_pairs(self) -> None:
        path = Path("data/golden_rules/wingspan_rules_qa.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["qa_pairs"]) >= 50

    def test_golden_dataset_required_fields(self) -> None:
        path = Path("data/golden_rules/wingspan_rules_qa.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"id", "category", "question", "expected_answer", "source_page"}
        for qa in data["qa_pairs"]:
            missing = required - set(qa.keys())
            assert not missing, f"QA {qa.get('id')} missing fields: {missing}"

    def test_golden_dataset_categories(self) -> None:
        path = Path("data/golden_rules/wingspan_rules_qa.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        valid_cats = {"basic_turn", "bird_power", "end_of_round", "edge_case", "exception"}
        for qa in data["qa_pairs"]:
            assert qa["category"] in valid_cats, f"Unknown category: {qa['category']}"

    def test_golden_dataset_has_all_categories(self) -> None:
        path = Path("data/golden_rules/wingspan_rules_qa.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        present = {qa["category"] for qa in data["qa_pairs"]}
        required = {"basic_turn", "bird_power", "end_of_round", "edge_case", "exception"}
        assert present == required

    def test_golden_dataset_unique_ids(self) -> None:
        path = Path("data/golden_rules/wingspan_rules_qa.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = [qa["id"] for qa in data["qa_pairs"]]
        assert len(ids) == len(set(ids)), "Duplicate IDs in golden dataset"


# ---------------------------------------------------------------------------
# Integration tests (require ChromaDB collection + API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.skipif(
    not Path("data/chroma_db").exists(),
    reason="ChromaDB not populated — run ingest_rulebook.py first",
)
class TestRuleOracleIntegration:
    """These tests call the real API and require the PDF to be ingested."""

    def test_integration_answer_rule_question(self) -> None:
        from src.oracle.claude_client import ClaudeClient
        from src.oracle.retriever import RuleRetriever

        retriever = RuleRetriever(chroma_persist_dir="data/chroma_db")
        if not retriever.collection_exists("wingspan"):
            pytest.skip("wingspan collection not ingested")

        client = ClaudeClient()
        oracle = RuleOracle(client=client, retriever=retriever)
        result = oracle.answer_rule_question(
            "What are the four actions a player can take?", "wingspan"
        )
        assert isinstance(result, RuleAnswer)
        assert result.confidence > 0.5
        assert len(result.answer) > 20
