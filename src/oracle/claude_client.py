from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048
_CACHE_VERSION = "v1"


class ClaudeClient:
    """Anthropic API wrapper with disk-based response caching and retry logic.

    Cache key: sha256(model + serialised messages + temperature).
    Cache location: CACHE_DIR env var (default ./data/cache/).
    This makes iterating on prompts during development free — repeated calls
    with identical inputs never hit the API.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.use_cache = use_cache

        cache_root = cache_dir or os.environ.get("CACHE_DIR", "./data/cache")
        self._cache_dir = Path(cache_root)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_prompt_cache: bool = True,
    ) -> str:
        """Send a chat completion request and return the text response.

        Args:
            messages: List of {role, content} dicts.
            system: Optional system prompt.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens in the response.
            use_prompt_cache: If True, adds cache_control to the system
                prompt so Anthropic caches it server-side between calls.

        Returns:
            The assistant text response.
        """
        cache_key = self._make_cache_key(messages, system, temperature, max_tokens)

        if self.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit for key %s", cache_key[:8])
                return cached

        response_text = self._call_api(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            use_prompt_cache=use_prompt_cache,
        )

        if self.use_cache:
            self._save_to_cache(cache_key, response_text)

        return response_text

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        """Same as complete() but parses the response as JSON.

        Args:
            messages: List of {role, content} dicts.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Parsed JSON dict from the model response.

        Raises:
            ValueError: If the response is not valid JSON.
        """
        text = self.complete(messages, system=system, temperature=temperature, max_tokens=max_tokens)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Try to extract JSON block if the model wrapped it in markdown
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Model response is not valid JSON:\n{text}") from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float,
        max_tokens: int,
        use_prompt_cache: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            if use_prompt_cache:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system

        logger.debug("Calling Claude API (model=%s, msgs=%d)", self.model, len(messages))
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _make_cache_key(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "version": _CACHE_VERSION,
                "model": self.model,
                "system": system,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data["response"]
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Corrupt cache file %s: %s", path, exc)
                path.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, key: str, response: str) -> None:
        path = self._cache_path(key)
        path.write_text(
            json.dumps({"response": response}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Cached response to %s", path.name)
