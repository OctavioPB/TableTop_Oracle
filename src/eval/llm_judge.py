"""LLM-as-Judge for qualitative play quality evaluation — Sprint 6.

Never called during RL training. Used post-hoc for qualitative error analysis
and as a sanity check on whether the agent's moves are strategically coherent.

Cost note: each evaluate_play_quality() call uses ~1 000 input tokens +
~300 output tokens on claude-sonnet-4-6 ≈ $0.005. Disk cache in ClaudeClient
ensures repeated calls on identical transcripts are free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.oracle.claude_client import ClaudeClient

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are an expert Wingspan board game analyst.
You will be given a transcript of one game and asked to evaluate the
strategic quality of player 0's decisions.

Respond ONLY with a valid JSON object with these exact keys:
{
  "strategic_coherence": <float 0-1>,
  "synergy_exploitation": <float 0-1>,
  "tactical_errors": [<short string>, ...],
  "summary": "<one sentence>"
}

Scoring guide:
  strategic_coherence: Does the player pursue a coherent engine-building
    strategy (e.g., forest birds, egg-laying chain, wetland draw engine)?
    0 = random, 1 = perfectly consistent.
  synergy_exploitation: Does the player activate bird powers that complement
    each other? 0 = ignores all synergies, 1 = maximises chain effects.
  tactical_errors: Concrete mistakes such as "played low-point bird when
    higher-point bird was affordable" or "wasted egg-laying turn with no
    birds on board". Empty list if no clear errors observed.
  summary: One sentence characterising the player's overall strategy.

Do not include any text outside the JSON object.
"""


@dataclass
class PlayQualityReport:
    """Structured output from LLMJudge.evaluate_play_quality().

    Attributes:
        strategic_coherence: Float in [0, 1]. How consistent is the strategy.
        synergy_exploitation: Float in [0, 1]. How well synergies are used.
        tactical_errors: List of human-readable mistake descriptions.
        summary: One-sentence characterisation of the player's strategy.
    """

    strategic_coherence: float
    synergy_exploitation: float
    tactical_errors: list[str] = field(default_factory=list)
    summary: str = ""


class LLMJudge:
    """Uses Claude to evaluate strategic quality of a game transcript.

    For qualitative evaluation only — never called during RL training.

    Args:
        client: ClaudeClient instance (with disk cache enabled by default).
    """

    def __init__(self, client: ClaudeClient) -> None:
        self._client = client

    def evaluate_play_quality(self, game_transcript: str) -> PlayQualityReport:
        """Evaluate the strategic quality of player 0's moves in a transcript.

        Sends the transcript to Claude with a structured prompt and parses the
        JSON response into a PlayQualityReport.

        Args:
            game_transcript: Human-readable game log with player 0's moves
                labelled. Typically produced by build_game_transcript().

        Returns:
            PlayQualityReport with scores and error list.

        Raises:
            ValueError: If the API response cannot be parsed as valid JSON.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "Please evaluate the strategic quality of player 0's moves "
                    "in the following Wingspan game transcript:\n\n"
                    + game_transcript
                ),
            }
        ]

        raw = self._client.complete(
            messages=messages,
            system=_JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> PlayQualityReport:
        """Parse Claude's JSON response into a PlayQualityReport.

        Args:
            raw: Raw string returned by ClaudeClient.complete().

        Returns:
            PlayQualityReport populated from the parsed JSON.

        Raises:
            ValueError: If raw cannot be decoded as JSON or is missing required keys.
        """
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLMJudge: failed to parse Claude response as JSON. "
                f"Raw response: {raw!r}"
            ) from exc

        try:
            return PlayQualityReport(
                strategic_coherence=float(data["strategic_coherence"]),
                synergy_exploitation=float(data["synergy_exploitation"]),
                tactical_errors=list(data.get("tactical_errors", [])),
                summary=str(data.get("summary", "")),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"LLMJudge: response JSON missing required keys. Data: {data}"
            ) from exc


def build_game_transcript(
    engine: Any,
    agents: list[Any],
    seed: int = 42,
    max_turns: int = 200,
) -> str:
    """Play one game and return a human-readable transcript.

    Produces a plain-text log of every action taken, suitable for passing
    to LLMJudge.evaluate_play_quality().

    Args:
        engine: WingspanEngine instance.
        agents: List of two BaseAgent instances; index = player_id.
        seed: RNG seed for engine.reset().
        max_turns: Safety cap on game length.

    Returns:
        Multi-line string with one action per line, labelled by player.
    """
    state = engine.reset(seed=seed)
    lines: list[str] = [f"=== Wingspan game (seed={seed}) ==="]
    turn = 0

    while not engine.is_terminal(state) and turn < max_turns:
        pid = state.player_id
        legal = engine.get_legal_actions(state)
        if not legal:
            break

        action = agents[pid].select_action(state, legal)
        lines.append(
            f"Turn {turn + 1:3d} | P{pid} | round={state.round} | "
            f"{action.action_type}"
            + (f" food={action.food_choice}" if action.food_choice else "")
            + (f" bird={action.card_name}" if action.card_name else "")
            + (f" habitat={action.target_habitat}" if action.target_habitat else "")
            + (f" draw={action.draw_source}" if action.draw_source else "")
        )

        result = engine.step(state, action)
        state = result.new_state
        turn += 1

    # Final scores
    scores = [engine._compute_final_score(state, pid) for pid in range(2)]
    winner = engine.get_winner(state)
    lines.append(
        f"=== Game over after {turn} turns | "
        f"P0={scores[0]} P1={scores[1]} | "
        f"Winner: P{winner} ==="
    )

    return "\n".join(lines)
