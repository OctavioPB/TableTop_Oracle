"""LLM-as-Judge for qualitative play quality evaluation — Sprint 6."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.oracle.claude_client import ClaudeClient


@dataclass
class PlayQualityReport:
    strategic_coherence: float
    synergy_exploitation: float
    tactical_errors: list[str] = field(default_factory=list)
    summary: str = ""


class LLMJudge:
    """Uses Claude to evaluate strategic quality of a game transcript.

    For qualitative evaluation only — never called during RL training.
    """

    def __init__(self, client: ClaudeClient) -> None:
        self._client = client

    def evaluate_play_quality(self, game_transcript: str) -> PlayQualityReport:
        raise NotImplementedError("S6.3 — implement in Sprint 6")
