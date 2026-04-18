"""CLI: evaluate a trained checkpoint — Sprint 4.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/ppo_wingspan_best.zip \\
        --n-games 500
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-games", type=int, default=500)
    args = parser.parse_args()

    raise NotImplementedError("S4 — implement in Sprint 4")


if __name__ == "__main__":
    main()
