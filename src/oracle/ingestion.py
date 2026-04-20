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

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ChunkType = Literal["rule", "example", "exception", "card_power"]

# 1 token ≈ 0.75 words — rough approximation, avoids adding tiktoken dependency
_WORDS_PER_TOKEN = 0.75
CHUNK_SIZE_TOKENS = 80
CHUNK_OVERLAP_TOKENS = 20
CHUNK_SIZE_WORDS: int = max(1, int(CHUNK_SIZE_TOKENS * _WORDS_PER_TOKEN))   # ~60
CHUNK_OVERLAP_WORDS: int = max(1, int(CHUNK_OVERLAP_TOKENS * _WORDS_PER_TOKEN))  # ~15
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Heading patterns: ALL CAPS (≥2 words or ≥8 chars), or Title Case line alone
_HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s\-/:]{7,}|[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,6})\s*$"
)
# Card power trigger keywords
_POWER_TRIGGERS = re.compile(
    r"\b(when played|when activated|once between turns|end of round|"
    r"all players|pink power|brown power|teal power|white power)\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(r"\b(for example|example:|e\.g\.)\b", re.IGNORECASE)
_EXCEPTION_RE = re.compile(
    # No trailing \b — "exception:" ends with a non-word char so \b would fail
    r"\b(exception:|unless|however if|does not apply|overrides)",
    re.IGNORECASE,
)


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


# Module-level lazy singleton — avoids reloading the 80 MB model on every call
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s …", EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


class RulebookIngester:
    """Ingests a PDF rulebook into ChromaDB."""

    def __init__(self, chroma_persist_dir: str | Path) -> None:
        self._chroma_dir = Path(chroma_persist_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, pdf_path: str | Path, game: str) -> int:
        """Ingest a PDF rulebook for the given game.

        Args:
            pdf_path: Path to the PDF file.
            game: Game identifier (used as ChromaDB collection suffix).

        Returns:
            Number of chunks ingested.
        """
        import chromadb

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Rulebook not found: {pdf_path}")

        logger.info("Extracting text from %s …", pdf_path.name)
        pages = self._extract_text_by_page(pdf_path)
        logger.info("Extracted %d pages", len(pages))

        chunks = self._chunk_text(pages, game)
        logger.info("Created %d chunks", len(chunks))

        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        collection_name = f"rules_{game}"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear existing data for this game so re-ingestion is idempotent
        existing = collection.count()
        if existing > 0:
            logger.warning(
                "Collection %s already has %d chunks — dropping before re-ingest",
                collection_name,
                existing,
            )
            client.delete_collection(collection_name)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        model = _get_embed_model()
        texts = [c.text for c in chunks]

        logger.info("Embedding %d chunks …", len(chunks))
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "page": c.page,
                    "section": c.section,
                    "game": c.game,
                    "chunk_type": c.chunk_type,
                }
                for c in chunks
            ],
        )

        logger.info("Ingested %d chunks into collection '%s'", len(chunks), collection_name)
        return len(chunks)

    def ingest_extra(self, doc_path: str | Path, game: str) -> int:
        """Append chunks from an extra document (PDF or TXT) to an existing collection.

        Unlike ingest(), this method does NOT drop the collection first — it adds
        new chunks alongside existing ones. Use this to supplement a rulebook with
        FAQ or quickstart documents.

        Args:
            doc_path: Path to a .pdf or .txt file.
            game: Game identifier (must match an existing collection).

        Returns:
            Number of new chunks added.
        """
        import chromadb

        doc_path = Path(doc_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        if doc_path.suffix.lower() == ".txt":
            pages = self._extract_text_from_txt(doc_path)
        else:
            pages = self._extract_text_by_page(doc_path)

        chunks = self._chunk_text(pages, game)
        # Prefix chunk IDs with source filename to avoid collisions
        src = doc_path.stem
        for c in chunks:
            c.chunk_id = hashlib.sha256(
                f"{src}:{c.chunk_id}".encode()
            ).hexdigest()[:20]

        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        collection_name = f"rules_{game}"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        model = _get_embed_model()
        texts = [c.text for c in chunks]
        logger.info("Embedding %d extra chunks from %s …", len(chunks), doc_path.name)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "page": c.page,
                    "section": c.section,
                    "game": c.game,
                    "chunk_type": c.chunk_type,
                    "source": doc_path.name,
                }
                for c in chunks
            ],
        )
        logger.info("Added %d chunks from %s to collection '%s'", len(chunks), doc_path.name, collection_name)
        return len(chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text_from_txt(self, txt_path: Path) -> list[tuple[int, str]]:
        """Return text from a plain-text file as a single pseudo-page."""
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        return [(1, text)]

    def _extract_text_by_page(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Return list of (1-indexed page_number, text) tuples."""
        import fitz  # pymupdf

        pages = []
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text))
        doc.close()
        return pages

    def _chunk_text(self, pages: list[tuple[int, str]], game: str) -> list[RuleChunk]:
        """Split pages into overlapping chunks preserving section context.

        Each word carries its (page, section) origin so that chunk metadata
        always reflects the first word of the chunk, not the last.
        """
        segments = self._extract_segments(pages)

        # Parallel arrays: words[i] belongs to pages_meta[i] = (page, section)
        words: list[str] = []
        pages_meta: list[tuple[int, str]] = []

        for page_num, section, text in segments:
            new_words = text.split()
            words.extend(new_words)
            pages_meta.extend([(page_num, section)] * len(new_words))

        chunks: list[RuleChunk] = []
        chunk_idx = 0
        pos = 0

        while pos + CHUNK_SIZE_WORDS <= len(words):
            end = pos + CHUNK_SIZE_WORDS
            chunk_words = words[pos:end]
            chunk_text = " ".join(chunk_words)
            start_page, start_section = pages_meta[pos]

            chunk_id = hashlib.sha256(
                f"{game}:{chunk_idx}:{chunk_text[:60]}".encode()
            ).hexdigest()[:20]

            chunks.append(
                RuleChunk(
                    text=chunk_text,
                    page=start_page,
                    section=start_section,
                    game=game,
                    chunk_type=self._detect_chunk_type(chunk_text),
                    chunk_id=chunk_id,
                )
            )
            chunk_idx += 1
            pos += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

        # Flush remainder (only if non-trivial)
        if pos < len(words) and (len(words) - pos) > 20:
            chunk_text = " ".join(words[pos:])
            start_page, start_section = pages_meta[pos]
            chunk_id = hashlib.sha256(
                f"{game}:{chunk_idx}:{chunk_text[:60]}".encode()
            ).hexdigest()[:20]
            chunks.append(
                RuleChunk(
                    text=chunk_text,
                    page=start_page,
                    section=start_section,
                    game=game,
                    chunk_type=self._detect_chunk_type(chunk_text),
                    chunk_id=chunk_id,
                )
            )

        return chunks

    def _extract_segments(
        self, pages: list[tuple[int, str]]
    ) -> list[tuple[int, str, str]]:
        """Return (page, section, paragraph_text) triples from all pages."""
        segments: list[tuple[int, str, str]] = []
        current_section = "Introduction"

        for page_num, text in pages:
            paragraphs = re.split(r"\n{2,}", text)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                detected = self._detect_section(para)
                if detected:
                    current_section = detected
                    # Include the heading text as part of the next segment context
                segments.append((page_num, current_section, para))

        return segments

    def _detect_section(self, text: str) -> str:
        """Return the section name if text is a section heading, else empty string."""
        line = text.strip()
        # Only consider short single-line blocks as headings
        if "\n" in line or len(line) > 80:
            return ""
        if _HEADING_RE.match(line):
            return line.title()
        return ""

    @staticmethod
    def _detect_chunk_type(text: str) -> ChunkType:
        """Classify a chunk as rule, example, exception, or card_power."""
        if _POWER_TRIGGERS.search(text):
            return "card_power"
        if _EXAMPLE_RE.search(text):
            return "example"
        if _EXCEPTION_RE.search(text):
            return "exception"
        return "rule"
