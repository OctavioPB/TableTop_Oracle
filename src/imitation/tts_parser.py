"""Tabletop Simulator log parser — Sprint 5.

Parses Tabletop Simulator (TTS) Wingspan save-state JSON logs into
DemonstrationBuffer Transitions.

TTS Wingspan logs follow this structure (community mod format):
{
  "game_id": "...",
  "winner": 0,
  "seed": 42,
  "moves": [
    {
      "player_id": 0,
      "action_type": "gain_food",
      "food_type": "seed",
      "turn": 1
    },
    ...
  ]
}

Unlike BGA logs, TTS logs may include additional metadata fields per move
(card name localisation, board snapshot) — these are ignored if unrecognised.

This parser mirrors BGALogParser: it replays moves through WingspanEngine
to reconstruct observations, and only records player-0 transitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.imitation.demo_buffer import DemonstrationBuffer, Transition

logger = logging.getLogger(__name__)


class TTSLogParser:
    """Parses Tabletop Simulator Wingspan history JSON into DemonstrationBuffer Transitions.

    Replays the recorded moves through WingspanEngine to reconstruct the
    observation at each step.  Only player-0 turns produce Transitions.
    """

    def parse_game_log(self, raw_log: dict[str, Any]) -> list[Transition]:
        """Parse one TTS game log dict into a list of Transitions.

        Args:
            raw_log: Parsed JSON dict conforming to the TTS Wingspan log format.

        Returns:
            List of Transition objects (one per player-0 action).

        Raises:
            ValueError: If raw_log is missing the 'moves' key or it is not a list.
        """
        from src.games.wingspan.engine import WingspanEngine
        from src.envs.wingspan_env import WingspanEnv

        moves = raw_log.get("moves")
        if moves is None or not isinstance(moves, list):
            raise ValueError("TTS log must contain a 'moves' list")

        if not moves:
            return []

        seed = raw_log.get("seed", 0)
        winner = raw_log.get("winner")

        engine = WingspanEngine(seed=seed)
        env = WingspanEnv()
        env.reset(seed=seed)

        transitions: list[Transition] = []

        for move in moves:
            if not isinstance(move, dict):
                logger.warning("Skipping non-dict move entry: %s", move)
                continue

            player_id = move.get("player_id", 0)

            # Build obs for the current player's perspective
            # (env always shows state from P0's perspective)
            obs_before = env._get_obs()

            # Map the TTS move to a WingspanAction
            action = self._map_tts_action(move, env)
            if action is None:
                logger.debug("Skipping unrecognised TTS move: %s", move)
                # Still need to advance engine state — skip step for now
                continue

            # Find action index in gym space
            action_idx = env._action_to_idx(action)

            # Step the env
            obs_after, reward, done, _, info = env.step(action_idx)

            if player_id == 0:
                transitions.append(Transition(
                    obs=obs_before,
                    action=action_idx,
                    next_obs=obs_after,
                    reward=float(reward),
                    done=done,
                    info={"move": move, "winner": winner},
                ))

            if done:
                break

        return transitions

    def parse_file(self, path: str | Path) -> list[Transition]:
        """Load a TTS log JSON file and parse it.

        Args:
            path: Path to a .json file in TTS Wingspan log format.

        Returns:
            List of Transition objects.
        """
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.parse_game_log(data)

    def parse_into_buffer(
        self,
        path: str | Path,
        buffer: DemonstrationBuffer | None = None,
    ) -> DemonstrationBuffer:
        """Parse a TTS log file and add its transitions to a DemonstrationBuffer.

        Args:
            path: Path to a .json TTS log file.
            buffer: Existing buffer to append to; creates a new one if None.

        Returns:
            DemonstrationBuffer with the parsed transitions added.
        """
        import json
        if buffer is None:
            buffer = DemonstrationBuffer()

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        transitions = self.parse_game_log(data)
        winner = data.get("winner")
        if transitions:
            buffer.add_game(transitions, winner=winner)
        return buffer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_tts_action(self, move: dict[str, Any], env: Any) -> Any | None:
        """Map a TTS move dict to a WingspanAction.

        TTS field names may differ slightly from BGA; both 'food_type' and
        'food_choice' are accepted for GAIN_FOOD actions.
        """
        from src.games.wingspan.actions import WingspanAction, WingspanActionType

        atype = move.get("action_type", "")

        if atype == WingspanActionType.GAIN_FOOD.value:
            food = move.get("food_type") or move.get("food_choice", "")
            legal = env._engine.get_legal_actions(env._state)
            for a in legal:
                if a.action_type == atype and a.food_choice == food:
                    return a
            # Fallback: first legal gain_food
            for a in legal:
                if a.action_type == atype:
                    return a

        elif atype == WingspanActionType.LAY_EGGS.value:
            legal = env._engine.get_legal_actions(env._state)
            for a in legal:
                if a.action_type == atype:
                    return a

        elif atype == WingspanActionType.DRAW_CARDS.value:
            source = move.get("draw_source", "deck")
            legal = env._engine.get_legal_actions(env._state)
            for a in legal:
                if a.action_type == atype and a.draw_source == source:
                    return a
            for a in legal:
                if a.action_type == atype:
                    return a

        elif atype == WingspanActionType.PLAY_BIRD.value:
            card = move.get("card_name", "")
            habitat = move.get("habitat") or move.get("target_habitat", "")
            legal = env._engine.get_legal_actions(env._state)
            for a in legal:
                if (a.action_type == atype
                        and a.card_name == card
                        and a.target_habitat == habitat):
                    return a
            # Partial match on card_name only
            for a in legal:
                if a.action_type == atype and a.card_name == card:
                    return a

        return None
