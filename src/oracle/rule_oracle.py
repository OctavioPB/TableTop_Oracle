"""Rule Oracle: LLM + RAG → natural-language rule validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.games.base.game_state import Action, GameState
from src.oracle.claude_client import ClaudeClient
from src.oracle.retriever import RetrievedChunk, RuleRetriever

logger = logging.getLogger(__name__)

_RETRIEVAL_K = 10         # chunks returned per query
_VALIDATION_K = 6         # smaller set for action validation (state already provides context)
_SYSTEM_PROMPT = (
    "You are a precise board game rules arbiter. "
    "Always base your answers strictly on the provided rulebook excerpts. "
    "Never invent rules. Respond only with the JSON format requested."
)


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

    Usage during development (Sprint 1–2):
        oracle.answer_rule_question("How many eggs can I lay?", "wingspan")
        oracle.validate_action(state, action, "wingspan")

    NOT called during RL training — only for edge cases not covered by the
    deterministic LegalMoveValidator. See PLAN.md D4 and architecture notes.
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
        self._prompts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_rule_question(self, question: str, game: str) -> RuleAnswer:
        """Answer a natural-language rule question with source references.

        Args:
            question: The rule question in natural language.
            game: Game identifier for retrieval (e.g. "wingspan").

        Returns:
            RuleAnswer with answer, confidence, sources, and verbatim quotes.
        """
        chunks = self._retriever.query(question, game, k=_RETRIEVAL_K)
        if not chunks:
            logger.warning("No chunks retrieved for question: %s", question)
            return RuleAnswer(
                answer="No relevant rules found in the rulebook.",
                confidence=0.0,
            )

        prompt_template = self._load_prompt("rule_question.txt")
        user_content = prompt_template.format(
            game=game,
            retrieved_chunks=self._format_chunks(chunks),
            question=question,
        )

        raw = self._client.complete_json(
            messages=[{"role": "user", "content": user_content}],
            system=_SYSTEM_PROMPT,
        )

        return RuleAnswer(
            answer=str(raw.get("answer", "")),
            confidence=float(raw.get("confidence", 0.5)),
            sources=list(raw.get("sources", [])),
            verbatim_quotes=list(raw.get("verbatim_quotes", [])),
        )

    def validate_action(
        self, state: GameState, action: Action, game: str
    ) -> ValidationResult:
        """Determine whether action is legal given state.

        Args:
            state: Current game state (serialised to JSON for the prompt).
            action: Proposed action.
            game: Game identifier.

        Returns:
            ValidationResult with legality flag and rule citations.
        """
        import json

        state_json = state.model_dump_json(indent=2)
        action_json = json.dumps(action.to_dict(), indent=2)

        # Retrieve rules relevant to the action type
        query = f"{game} rules for action: {action.action_type}"
        chunks = self._retriever.query(query, game, k=_VALIDATION_K)

        prompt_template = self._load_prompt("rule_validator.txt")
        user_content = prompt_template.format(
            game=game,
            retrieved_chunks=self._format_chunks(chunks),
            game_state_json=state_json,
            action_json=action_json,
        )

        raw = self._client.complete_json(
            messages=[{"role": "user", "content": user_content}],
            system=_SYSTEM_PROMPT,
        )

        return ValidationResult(
            is_legal=bool(raw.get("is_legal", False)),
            reason=str(raw.get("reason", "")),
            rule_references=[str(raw.get("rule_quote", ""))],
            confidence=float(raw.get("confidence", 0.5)),
        )

    def resolve_conflict(self, rule_a: str, rule_b: str) -> str:
        """Resolve an apparent contradiction between two rule texts.

        Args:
            rule_a: First rule text.
            rule_b: Second rule text.

        Returns:
            Explanation of which rule takes precedence and why.
        """
        prompt_template = self._load_prompt("edge_case_resolver.txt")
        user_content = prompt_template.format(rule_a=rule_a, rule_b=rule_b)

        raw = self._client.complete_json(
            messages=[{"role": "user", "content": user_content}],
            system=_SYSTEM_PROMPT,
        )

        return str(raw.get("ruling", "")) + " — " + str(raw.get("reason", ""))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_prompt(self, filename: str) -> str:
        """Load and cache a prompt template from the prompts directory."""
        if filename not in self._prompts:
            path = self._prompt_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Prompt file not found: {path}. "
                    f"Expected in {self._prompt_dir}"
                )
            self._prompts[filename] = path.read_text(encoding="utf-8")
        return self._prompts[filename]

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Page {chunk.page} — {chunk.section} "
                f"(type: {chunk.chunk_type}, score: {chunk.score:.2f})\n"
                f"{chunk.text}"
            )
        return "\n\n".join(parts)
