"""Oracle-guided reward shaping for RL training — Sprint 6.

Pre-computes strategic guidance from the Rule Oracle ONCE (offline, cached),
then exposes a zero-cost lookup table used by OracleShapedWingspanEnv during
training. This design satisfies the $15 API budget constraint: ~20 queries at
setup, 0 queries during 1M training steps.

Usage:
    bonuses = build_oracle_bonuses(oracle, game="wingspan", cache_path=Path(...))
    # -> {"play_bird": 0.045, "gain_food": 0.040, "lay_eggs": 0.048, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum additive bonus per training step.
# Kept small relative to the base dense reward (~0.5–2.0) to avoid
# overwhelming the environment's own reward signal.
_BONUS_SCALE = 0.05

# Strategic questions — one per Wingspan action type.
# Answers are parsed for confidence, which scales the bonus.
_WINGSPAN_STRATEGIC_QUERIES: list[tuple[str, str]] = [
    (
        "play_bird",
        "In Wingspan, how strategically valuable is playing a new bird card to a habitat "
        "compared to other possible actions? Is it generally a high-priority move?",
    ),
    (
        "gain_food",
        "In Wingspan, how important is the gain-food action relative to other actions? "
        "Should players prioritise gathering food early in the game?",
    ),
    (
        "lay_eggs",
        "In Wingspan, how strategically important is laying eggs? Is laying eggs considered "
        "one of the most efficient actions for scoring in Wingspan?",
    ),
    (
        "draw_cards",
        "In Wingspan, when is drawing bird cards strategically valuable? "
        "Is drawing cards a high-priority action compared to playing birds or laying eggs?",
    ),
]


def build_oracle_bonuses(
    oracle: object,
    game: str,
    cache_path: Path | None = None,
) -> dict[str, float]:
    """Query the Rule Oracle offline and return action_type -> reward bonus.

    Results are written to cache_path (JSON). On subsequent calls with the
    same cache_path the file is read directly — no API calls are made.

    Args:
        oracle: A RuleOracle instance.
        game: Game identifier (currently only "wingspan" is supported).
        cache_path: Optional path to persist/load the bonus dict as JSON.

    Returns:
        Dict mapping action_type strings to float bonuses in [0, _BONUS_SCALE].
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading oracle bonuses from cache: %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if game != "wingspan":
        logger.warning(
            "Oracle reward shaping is only implemented for 'wingspan' — "
            "returning zero bonuses for game='%s'.",
            game,
        )
        return {}

    bonuses: dict[str, float] = {}

    for action_type, question in _WINGSPAN_STRATEGIC_QUERIES:
        try:
            answer = oracle.answer_rule_question(question, game)  # type: ignore[attr-defined]
            # Scale confidence linearly into [0, _BONUS_SCALE]
            bonus = round(float(answer.confidence) * _BONUS_SCALE, 5)
            bonuses[action_type] = bonus
            logger.info(
                "Oracle bonus  %-12s = %.5f  (confidence=%.3f)",
                action_type,
                bonus,
                answer.confidence,
            )
        except Exception as exc:
            logger.warning("Oracle query failed for action_type=%s: %s", action_type, exc)
            bonuses[action_type] = 0.0

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(bonuses, indent=2), encoding="utf-8")
        logger.info("Oracle bonuses persisted to: %s", cache_path)

    return bonuses
