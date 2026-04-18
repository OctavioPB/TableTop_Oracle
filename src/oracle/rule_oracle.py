"""Rule Oracle: LLM + RAG → natural-language rule validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.games.base.game_state import Action, GameState
from src.oracle.claude_client import ClaudeClient
from src.oracle.retriever import RuleRetriever

logger = logging.getLogger(__name__)


@dataclass
class RuleAnswer:
    """Response from the Rule Oracle to a rule question."""

    answer: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    verbatim_quotes: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Whether an action is legal given the current game state."""

    is_legal: bool
    reason: str
    rule_references: list[str] = field(default_factory=list)
    confidence: float = 1.0


class RuleOracle:
    """Answers rule questions and validates actions via RAG + LLM.

    The LLM is only called during development (Sprint 1–2) and as a
    fallback during RL training for uncovered edge cases. It is NOT
    called on every RL step — that would cost ~$3,000 USD.
    """

    def __init__(
        self,
        client: ClaudeClient,
        retriever: RuleRetriever,
        prompt_dir: str | Path | None = None,
    ) -> None:
        self._client = client
        self._retriever = retriever
        self._prompt_dir = Path(prompt_dir or "src/oracle/prompts")

    def answer_rule_question(self, question: str, game: str) -> RuleAnswer:
        """Answer a natural-language rule question with source references.

        Args:
            question: The rule question in natural language.
            game: Game identifier for retrieval.

        Returns:
            RuleAnswer with answer text, confidence, and source references.
        """
        raise NotImplementedError("S1.4 — implement in Sprint 1")

    def validate_action(
        self, state: GameState, action: Action, game: str
    ) -> ValidationResult:
        """Determine whether action is legal given state.

        Args:
            state: Current serialised game state.
            action: Proposed action.
            game: Game identifier for retrieval.

        Returns:
            ValidationResult with legality flag and rule citations.
        """
        raise NotImplementedError("S1.4 — implement in Sprint 1")

    def resolve_conflict(self, rule_a: str, rule_b: str) -> str:
        """Resolve an apparent contradiction between two rule texts.

        Args:
            rule_a: First rule text.
            rule_b: Second rule text.

        Returns:
            Explanation of which rule takes precedence and why.
        """
        raise NotImplementedError("S1.4 — implement in Sprint 1")
