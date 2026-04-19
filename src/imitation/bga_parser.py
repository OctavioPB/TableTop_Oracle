"""BoardGameArena log parser — Sprint 5.

Parses Wingspan game logs in the TabletopOracle BGA format into
Transition objects compatible with DemonstrationBuffer.

BGA Log Format (JSON):
----------------------
{
  "game_id": "12345",
  "winner": 0,
  "seed": 42,
  "moves": [
    {"player_id": 0, "action_type": "gain_food", "food_type": "seed"},
    {"player_id": 1, "action_type": "lay_eggs"},
    {"player_id": 0, "action_type": "play_bird",
     "card_name": "American Robin", "habitat": "forest"},
    {"player_id": 0, "action_type": "draw_cards", "draw_source": "tray_0"},
    ...
  ]
}

Notes:
  - "winner" is the player_id of the winning player (0 or 1), or null for draws.
  - Each move belongs to exactly one player.
  - The parser replays moves through WingspanEngine to reconstruct states.
  - Only player-0 moves produce Transition objects (BC trains player-0 policy).
  - For PLAY_BIRD actions, the parser uses the first legal matching action from
    the engine to resolve food_payment and egg_payment automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.imitation.demo_buffer import DemonstrationBuffer, Transition

logger = logging.getLogger(__name__)

# Supported BGA action_type strings
_VALID_ACTION_TYPES = frozenset({
    "gain_food", "lay_eggs", "draw_cards", "play_bird",
})


class BGALogParser:
    """Parses BGA Wingspan game logs into DemonstrationBuffer transitions.

    Reconstructs game states by replaying moves through WingspanEngine
    and WingspanEnv to obtain gym-compatible observations for each step.

    Usage:
        parser = BGALogParser()
        buffer = parser.parse_directory(Path("data/game_logs/bga/"))
    """

    def __init__(self) -> None:
        pass  # Stateless — one parser instance can handle multiple logs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse_game_log(self, raw_log: dict[str, Any]) -> list[Transition]:
        """Parse a single BGA game log into Transition objects.

        Only player-0 decisions are converted to Transitions; player-1
        moves are replayed to advance the environment state correctly.

        Args:
            raw_log: Parsed JSON dict matching the BGA log format above.

        Returns:
            List of Transition objects for player 0's decisions in this game.

        Raises:
            ValueError: If the log is missing required fields or contains
                unknown action types.
        """
        self._validate_log(raw_log)

        from src.envs.wingspan_env import WingspanEnv
        from src.games.wingspan.actions import WingspanAction, WingspanActionType

        seed = raw_log.get("seed", 0)
        env = WingspanEnv(reward_mode="dense")
        obs, _ = env.reset(seed=seed)

        transitions: list[Transition] = []
        moves: list[dict[str, Any]] = raw_log.get("moves", [])

        for move_idx, move in enumerate(moves):
            player_id = int(move.get("player_id", 0))
            action_type_str = move.get("action_type", "")

            if action_type_str not in _VALID_ACTION_TYPES:
                logger.warning(
                    "Move %d: unknown action_type '%s' — skipping.",
                    move_idx, action_type_str,
                )
                continue

            # Map the BGA move to a WingspanAction
            wingspan_action = self._map_action(move, env, player_id)
            if wingspan_action is None:
                logger.warning(
                    "Move %d: could not map action %s for player %d — skipping.",
                    move_idx, action_type_str, player_id,
                )
                continue

            action_idx = env._action_to_idx(wingspan_action) if player_id == 0 else None

            # For player 1, temporarily switch state so the env step works
            if player_id == 1:
                # Manually advance using the engine (player 1 uses engine directly)
                try:
                    result = env._engine.step(env._state, wingspan_action)
                    env._state = result.new_state
                    # Restore invariant: player_id = 0
                    if not env._engine.is_terminal(env._state):
                        env._state = env._state.model_copy(update={"player_id": 0})
                    obs = env._get_obs()
                except Exception as exc:
                    logger.warning(
                        "Move %d: player-1 step failed (%s) — stopping game replay.",
                        move_idx, exc,
                    )
                    break
                continue

            # Player 0: record (obs, action_idx) and advance via env.step()
            if action_idx is None:
                # Fallback: use first legal masked action
                import numpy as np
                mask = env.action_masks()
                action_idx = int(np.argmax(mask))
                logger.debug(
                    "Move %d: action index is None, using fallback idx=%d",
                    move_idx, action_idx,
                )

            if env._done:
                logger.info("Game ended before all moves were replayed.")
                break

            try:
                next_obs, reward, terminated, truncated, info = env.step(action_idx)
            except Exception as exc:
                logger.warning(
                    "Move %d: env.step(%d) failed (%s) — stopping game replay.",
                    move_idx, action_idx, exc,
                )
                break

            done = terminated or truncated
            transitions.append(Transition(
                obs={k: v.copy() for k, v in obs.items()},
                action=action_idx,
                next_obs={k: v.copy() for k, v in next_obs.items()},
                reward=float(reward),
                done=done,
                info=dict(info),
            ))
            obs = next_obs
            if done:
                break

        env.close()
        logger.debug(
            "Game %s: %d player-0 transitions extracted from %d moves.",
            raw_log.get("game_id", "?"), len(transitions), len(moves),
        )
        return transitions

    def parse_file(self, path: Path | str) -> list[Transition]:
        """Parse a single JSON log file.

        Args:
            path: Path to a JSON file in BGA log format.

        Returns:
            List of Transitions from this game.
        """
        import json
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self.parse_game_log(raw)

    def parse_directory(
        self,
        directory: Path | str,
        glob_pattern: str = "*.json",
    ) -> DemonstrationBuffer:
        """Parse all JSON log files in a directory into a DemonstrationBuffer.

        Args:
            directory: Directory containing BGA log JSON files.
            glob_pattern: Glob pattern to match log files.

        Returns:
            DemonstrationBuffer populated from all parsed games.
        """
        import json
        directory = Path(directory)
        buffer = DemonstrationBuffer()

        log_files = sorted(directory.glob(glob_pattern))
        if not log_files:
            logger.warning("No files matching '%s' in %s.", glob_pattern, directory)
            return buffer

        for path in log_files:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                transitions = self.parse_game_log(raw)
                winner = raw.get("winner")
                buffer.add_game(transitions, winner=winner)
                logger.debug("Parsed %s: %d transitions.", path.name, len(transitions))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s — skipping.", path.name, exc)

        logger.info(
            "Parsed %d log files → %d transitions (%d games).",
            len(log_files), len(buffer), buffer.n_games,
        )
        return buffer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_log(self, raw_log: dict[str, Any]) -> None:
        """Raise ValueError if required fields are missing."""
        if "moves" not in raw_log:
            raise ValueError("BGA log missing required field 'moves'.")
        if not isinstance(raw_log["moves"], list):
            raise ValueError("'moves' must be a list.")

    def _map_action(
        self,
        move: dict[str, Any],
        env: Any,
        player_id: int,
    ) -> Any | None:
        """Convert a BGA move dict to a WingspanAction.

        Searches the engine's current legal actions to find the matching
        action (which has pre-computed payment fields). Falls back to the
        first legal action of the requested type if no exact match exists.

        Args:
            move: Single move dict from the BGA log.
            env: Active WingspanEnv (used for legal action lookup).
            player_id: Player whose turn it is.

        Returns:
            Matching WingspanAction, or None if no legal match found.
        """
        from src.games.wingspan.actions import WingspanAction, WingspanActionType

        action_type_str = move.get("action_type", "")

        # Temporarily set player_id on the state for legal action lookup
        original_pid = env._state.player_id
        if env._state.player_id != player_id:
            env._state = env._state.model_copy(update={"player_id": player_id})

        try:
            legal = env._engine.get_legal_actions(env._state)
        finally:
            # Always restore
            if env._state.player_id != original_pid:
                env._state = env._state.model_copy(update={"player_id": original_pid})

        if action_type_str == "gain_food":
            food_type = move.get("food_type", "")
            for la in legal:
                if (la.action_type == WingspanActionType.GAIN_FOOD.value
                        and la.food_choice == food_type):
                    return la
            # Fallback: any GAIN_FOOD
            for la in legal:
                if la.action_type == WingspanActionType.GAIN_FOOD.value:
                    return la

        elif action_type_str == "lay_eggs":
            for la in legal:
                if la.action_type == WingspanActionType.LAY_EGGS.value:
                    return la

        elif action_type_str == "draw_cards":
            draw_source = move.get("draw_source", "deck")
            for la in legal:
                if (la.action_type == WingspanActionType.DRAW_CARDS.value
                        and la.draw_source == draw_source):
                    return la
            for la in legal:
                if la.action_type == WingspanActionType.DRAW_CARDS.value:
                    return la

        elif action_type_str == "play_bird":
            card_name = move.get("card_name", "")
            habitat = move.get("habitat", "")
            # Exact match: card + habitat
            for la in legal:
                if (la.action_type == WingspanActionType.PLAY_BIRD.value
                        and la.card_name == card_name
                        and la.target_habitat == habitat):
                    return la
            # Partial match: same card, any habitat
            for la in legal:
                if (la.action_type == WingspanActionType.PLAY_BIRD.value
                        and la.card_name == card_name):
                    return la
            # Fallback: any PLAY_BIRD
            for la in legal:
                if la.action_type == WingspanActionType.PLAY_BIRD.value:
                    return la

        return None


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def generate_synthetic_bga_log(seed: int = 42, n_moves: int = 20) -> dict[str, Any]:
    """Generate a synthetic BGA-format log for testing.

    Runs a real WingspanEngine game for n_moves turns and records the
    moves in BGA format. The seed is stored so the parser can replay it.

    Args:
        seed: Engine seed for reproducibility.
        n_moves: Maximum number of moves to record (total, both players).

    Returns:
        BGA log dict suitable for BGALogParser.parse_game_log().
    """
    import random as pyrandom

    from src.envs.wingspan_env import WingspanEnv
    from src.games.wingspan.actions import WingspanActionType

    env = WingspanEnv(reward_mode="dense")
    env.reset(seed=seed)

    rng = pyrandom.Random(seed)
    moves: list[dict[str, Any]] = []
    winner: int | None = None

    # Alternate P0 and P1 directly via the engine (bypass env's auto-P1 step)
    for move_num in range(n_moves):
        if env._engine.is_terminal(env._state):
            scores = env._engine.compute_scores(env._state)
            s0, s1 = scores.get(0, 0), scores.get(1, 0)
            winner = 0 if s0 > s1 else (1 if s1 > s0 else None)
            break

        player_id = move_num % 2  # alternate players
        env._state = env._state.model_copy(update={"player_id": player_id})
        legal = env._engine.get_legal_actions(env._state)
        if not legal:
            break

        action = rng.choice(legal)
        atype = action.action_type

        if atype == WingspanActionType.GAIN_FOOD.value:
            moves.append({
                "player_id": player_id,
                "action_type": "gain_food",
                "food_type": action.food_choice or "seed",
            })
        elif atype == WingspanActionType.LAY_EGGS.value:
            moves.append({"player_id": player_id, "action_type": "lay_eggs"})
        elif atype == WingspanActionType.DRAW_CARDS.value:
            moves.append({
                "player_id": player_id,
                "action_type": "draw_cards",
                "draw_source": action.draw_source or "deck",
            })
        elif atype == WingspanActionType.PLAY_BIRD.value:
            moves.append({
                "player_id": player_id,
                "action_type": "play_bird",
                "card_name": action.card_name or "",
                "habitat": action.target_habitat or "forest",
            })

        result = env._engine.step(env._state, action)
        env._state = result.new_state

    env.close()
    return {
        "game_id": f"synthetic_{seed}",
        "seed": seed,
        "winner": winner,
        "moves": moves,
    }
