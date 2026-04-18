"""RAG retrieval over ChromaDB rule chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Re-use the same lazy-loaded model from ingestion to avoid loading it twice
from src.oracle.ingestion import EMBEDDING_MODEL, _get_embed_model


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, game: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the k most relevant chunks for a question.

        Args:
            question: Natural language question.
            game: Game identifier for collection selection.
            k: Number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by relevance score (descending).
        """
        collection = self._get_collection(game)
        embedding = _get_embed_model().encode(question).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(results, game)

    def query_with_filter(
        self, question: str, game: str, chunk_type: str, k: int = 3
    ) -> list[RetrievedChunk]:
        """Like query() but restricted to a specific chunk_type.

        Args:
            question: Natural language question.
            game: Game identifier.
            chunk_type: One of "rule", "example", "exception", "card_power".
            k: Number of chunks to return.

        Returns:
            List of RetrievedChunk filtered by chunk_type.
        """
        collection = self._get_collection(game)
        embedding = _get_embed_model().encode(question).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()),
            where={"chunk_type": {"$eq": chunk_type}},
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(results, game)

    def collection_exists(self, game: str) -> bool:
        """Return True if a ChromaDB collection exists for this game."""
        import chromadb

        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        existing = [c.name for c in client.list_collections()]
        return f"rules_{game}" in existing

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_collection(self, game: str):
        """Return (and cache) the ChromaDB collection for a game."""
        if game not in self._collections:
            import chromadb

            client = chromadb.PersistentClient(path=str(self._chroma_dir))
            collection_name = f"rules_{game}"
            try:
                self._collections[game] = client.get_collection(collection_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Collection '{collection_name}' not found. "
                    f"Run: python scripts/ingest_rulebook.py --game {game}"
                ) from exc
        return self._collections[game]

    @staticmethod
    def _parse_results(results: dict, game: str) -> list[RetrievedChunk]:
        """Convert raw ChromaDB query output to RetrievedChunk objects."""
        chunks: list[RetrievedChunk] = []

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB returns cosine distance; convert to similarity score
            score = 1.0 - dist
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    page=int(meta.get("page", 0)),
                    section=str(meta.get("section", "")),
                    game=str(meta.get("game", game)),
                    chunk_type=str(meta.get("chunk_type", "rule")),
                    score=score,
                    chunk_id=str(meta.get("chunk_id", "")),
                )
            )

        return chunks
