"""RAG retrieval over ChromaDB rule chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A rulebook chunk returned by a retrieval query."""

    text: str
    page: int
    section: str
    game: str
    chunk_type: str
    score: float
    chunk_id: str = ""


class RuleRetriever:
    """Retrieves relevant rule chunks from ChromaDB using semantic search."""

    def __init__(self, chroma_persist_dir: str | Path) -> None:
        self._chroma_dir = Path(chroma_persist_dir)
        self._collections: dict[str, object] = {}

    def query(self, question: str, game: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the k most relevant chunks for a question about game rules.

        Args:
            question: Natural language question.
            game: Game identifier for collection selection.
            k: Number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by relevance (descending).
        """
        raise NotImplementedError("S1.2 — implement in Sprint 1")

    def query_with_filter(
        self, question: str, game: str, chunk_type: str, k: int = 3
    ) -> list[RetrievedChunk]:
        """Like query() but filtered to a specific chunk_type."""
        raise NotImplementedError("S1.2 — implement in Sprint 1")
