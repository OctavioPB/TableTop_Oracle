# TabletopOracle

Agent that learns to play complex board games (Wingspan, Terraforming Mars) by reading the rulebook in natural language, combining LLM-based rule interpretation (RAG) with reinforcement learning (MaskablePPO).

**Research target:** AAAI 2026 Workshop on Games & AI / NeurIPS 2025 Games Workshop.

## Setup

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
pip install -e ".[dev]"
pytest tests/test_smoke.py -v  # verify baseline
```

See `PLAN.md` for full architecture and sprint roadmap.
