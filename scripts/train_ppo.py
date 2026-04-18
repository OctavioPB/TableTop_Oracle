"""CLI: train MaskablePPO on a Wingspan environment — Sprint 4.

Usage:
    python scripts/train_ppo.py \\
        --game wingspan \\
        --total-timesteps 1_000_000 \\
        --n-envs 8 \\
        --reward-mode dense \\
        --seed 42
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
    parser.add_argument("--game", default="wingspan")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--reward-mode", default="dense", choices=["terminal", "dense", "shaped"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raise NotImplementedError("S4 — implement in Sprint 4")


if __name__ == "__main__":
    main()
