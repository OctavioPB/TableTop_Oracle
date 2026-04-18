"""PDF rulebook ingestion pipeline: PDF → chunks → ChromaDB.

Pipeline:
1. pymupdf extracts text per page
2. Section detection via regex over headings
3. Chunking: 400 tokens with 80-token overlap, section context preserved
4. Metadata per chunk: {page, section, game, chunk_type}
5. Embed with all-MiniLM-L6-v2
6. Persist in ChromaDB, collection: rules_{game_name}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ChunkType = Literal["rule", "example", "exception", "card_power"]

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class RuleChunk:
    """A single chunk from the rulebook with its metadata."""

    text: str
    page: int
    section: str
    game: str
    chunk_type: ChunkType
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


class RulebookIngester:
    """Ingests a PDF rulebook into ChromaDB."""

    def __init__(self, chroma_persist_dir: str | Path) -> None:
        self._chroma_dir = Path(chroma_persist_dir)
        self._collection: object | None = None

    def ingest(self, pdf_path: str | Path, game: str) -> int:
        """Ingest a PDF rulebook for the given game.

        Args:
            pdf_path: Path to the PDF file.
            game: Game identifier used as collection name suffix.

        Returns:
            Number of chunks ingested.
        """
        raise NotImplementedError("S1.1 — implement in Sprint 1")

    def _extract_text_by_page(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Return list of (page_number, text) tuples."""
        raise NotImplementedError

    def _chunk_text(self, pages: list[tuple[int, str]], game: str) -> list[RuleChunk]:
        """Split pages into overlapping chunks with section metadata."""
        raise NotImplementedError

    def _detect_section(self, text: str) -> str:
        """Heuristic section heading detection via regex."""
        raise NotImplementedError
