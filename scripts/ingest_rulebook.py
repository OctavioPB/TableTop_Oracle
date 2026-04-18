"""CLI: ingest a PDF rulebook into ChromaDB.

Usage:
    python scripts/ingest_rulebook.py --game wingspan
    python scripts/ingest_rulebook.py --game wingspan --pdf data/rulebooks/wingspan_rulebook.pdf
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a rulebook PDF into ChromaDB")
    parser.add_argument("--game", required=True, help="Game identifier, e.g. wingspan")
    parser.add_argument("--pdf", help="Path to PDF (defaults to data/rulebooks/{game}_rulebook.pdf)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else Path(f"data/rulebooks/{args.game}_rulebook.pdf")
    chroma_dir = Path(os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_db"))

    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        raise SystemExit(1)

    from src.oracle.ingestion import RulebookIngester

    ingester = RulebookIngester(chroma_persist_dir=chroma_dir)
    n_chunks = ingester.ingest(pdf_path=pdf_path, game=args.game)
    logger.info("Ingested %d chunks for game '%s'", n_chunks, args.game)


if __name__ == "__main__":
    main()
