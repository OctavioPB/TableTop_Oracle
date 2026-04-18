"""CLI: evaluate Rule Oracle accuracy against the golden Q&A dataset.

Usage:
    python scripts/eval_rule_oracle.py --game wingspan
    python scripts/eval_rule_oracle.py --game wingspan --golden data/golden_rules/wingspan_rules_qa.json

Target: accuracy >= 0.80 to proceed to Sprint 2.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.80


def _score_answer(predicted: str, expected_keywords: list[str]) -> bool:
    """Keyword-based scoring: answer is correct if it contains enough keywords.

    This is a fast proxy metric for accuracy. For rigorous evaluation, replace
    with an embedding-based or LLM-as-judge scoring function.
    """
    pred_lower = predicted.lower()
    hit_count = sum(1 for kw in expected_keywords if kw.lower() in pred_lower)
    return hit_count >= max(1, len(expected_keywords) // 2)


def _run_evaluation(args: argparse.Namespace) -> None:
    from src.oracle.claude_client import ClaudeClient
    from src.oracle.retriever import RuleRetriever
    from src.oracle.rule_oracle import RuleOracle

    chroma_dir = Path(os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_db"))
    golden_path = Path(args.golden)

    if not golden_path.exists():
        logger.error("Golden dataset not found: %s", golden_path)
        sys.exit(1)

    retriever = RuleRetriever(chroma_persist_dir=chroma_dir)
    if not retriever.collection_exists(args.game):
        logger.error(
            "ChromaDB collection 'rules_%s' not found. "
            "Run: python scripts/ingest_rulebook.py --game %s",
            args.game,
            args.game,
        )
        sys.exit(1)

    client = ClaudeClient()
    oracle = RuleOracle(client=client, retriever=retriever)

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    qa_pairs = golden["qa_pairs"]
    if args.category:
        qa_pairs = [q for q in qa_pairs if q["category"] == args.category]
        logger.info("Filtered to category '%s': %d questions", args.category, len(qa_pairs))

    results: list[dict] = []
    correct = 0

    for i, qa in enumerate(qa_pairs, 1):
        logger.info("[%d/%d] %s: %s", i, len(qa_pairs), qa["id"], qa["question"][:60])

        try:
            answer = oracle.answer_rule_question(qa["question"], args.game)
            hit = _score_answer(answer.answer, qa.get("keywords", []))
            correct += int(hit)

            results.append({
                "id": qa["id"],
                "category": qa["category"],
                "question": qa["question"],
                "predicted": answer.answer,
                "expected_keywords": qa.get("keywords", []),
                "hit": hit,
                "confidence": answer.confidence,
                "sources": answer.sources,
            })

            status = "✓" if hit else "✗"
            logger.info("  %s  conf=%.2f  predicted=%s", status, answer.confidence, answer.answer[:80])

        except Exception as exc:
            logger.error("  ERROR for %s: %s", qa["id"], exc)
            results.append({
                "id": qa["id"],
                "category": qa["category"],
                "question": qa["question"],
                "predicted": "",
                "hit": False,
                "error": str(exc),
            })

    n = len(qa_pairs)
    accuracy = correct / n if n > 0 else 0.0

    # Per-category breakdown
    categories: dict[str, list[bool]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["hit"])

    print("\n" + "=" * 60)
    print(f"Rule Oracle Evaluation — {args.game}")
    print("=" * 60)
    for cat, hits in sorted(categories.items()):
        cat_acc = sum(hits) / len(hits)
        print(f"  {cat:<20s}  {sum(hits):>2d}/{len(hits):>2d}  ({cat_acc:.0%})")
    print("-" * 60)
    print(f"  {'TOTAL':<20s}  {correct:>2d}/{n:>2d}  ({accuracy:.0%})")
    print("=" * 60)

    if accuracy >= ACCURACY_THRESHOLD:
        print(f"\n✓ Accuracy {accuracy:.0%} >= threshold {ACCURACY_THRESHOLD:.0%}  →  Sprint 2 cleared")
    else:
        print(
            f"\n✗ Accuracy {accuracy:.0%} < threshold {ACCURACY_THRESHOLD:.0%}  "
            f"→  Improve chunking or retrieval before Sprint 2"
        )

    # Save detailed results
    output_path = Path(f"data/golden_rules/{args.game}_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": accuracy, "correct": correct, "total": n, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Detailed results saved to %s", output_path)

    if accuracy < ACCURACY_THRESHOLD:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Rule Oracle against the golden Q&A dataset"
    )
    parser.add_argument("--game", default="wingspan", help="Game identifier")
    parser.add_argument(
        "--golden",
        default=None,
        help="Path to golden dataset JSON (default: data/golden_rules/{game}_rules_qa.json)",
    )
    parser.add_argument(
        "--category",
        default=None,
        choices=["basic_turn", "bird_power", "end_of_round", "edge_case", "exception"],
        help="Evaluate only this category",
    )
    args = parser.parse_args()

    if args.golden is None:
        args.golden = f"data/golden_rules/{args.game}_rules_qa.json"

    _run_evaluation(args)


if __name__ == "__main__":
    main()
