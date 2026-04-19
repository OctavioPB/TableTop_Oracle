# TabletopOracle

**AI agents that learn to play modern board games from natural-language rulebooks.**

TabletopOracle combines Retrieval-Augmented Generation (RAG) for rule interpretation
with Masked Proximal Policy Optimization (PPO) for strategy learning.  No hardcoded
game logic — the system reads the manual and builds a Python rule engine via
LLM-assisted code generation.

Current games: **Wingspan** · **7 Wonders Duel**

**Research target:** AAAI 2026 Workshop on Games & Agents / NeurIPS 2025 Games Workshop.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  RULE LAYER                                                      │
│  PDF Rulebook → Chunker → ChromaDB ← Rule Oracle (Claude RAG)   │
├──────────────────────────────────────────────────────────────────┤
│  ENVIRONMENT LAYER                                               │
│  GameState (Pydantic v2) + LegalMoveValidator + gym.Env         │
├──────────────────────────────────────────────────────────────────┤
│  AGENT LAYER                                                     │
│  BC pre-train → MaskablePPO (stable-baselines3 + sb3-contrib)   │
└──────────────────────────────────────────────────────────────────┘
```

Key design decision: **LLMs are used only at development time** (rule engine synthesis
and edge-case validation), never inside the RL training loop.  This reduces API cost
from ~$3,000 to ~$15 for a 1M-step training run.

---

## Installation

### Requirements

- Python 3.11+
- An Anthropic API key (for Rule Oracle; not needed for training)

### Quick start

```bash
# Clone and set up
git clone https://github.com/your-org/TableTop_Oracle.git
cd TableTop_Oracle
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Verify installation
python -c "from src.oracle.claude_client import ClaudeClient; print('OK')"

# Run tests
pytest tests/ -v --tb=short
```

### Using Docker

```bash
docker build -t tabletop-oracle .
docker run --env-file .env tabletop-oracle pytest tests/ -v
```

---

## Usage

### Train a PPO agent on Wingspan

```bash
python scripts/train_ppo.py \
    --game wingspan \
    --total-timesteps 1_000_000 \
    --n-envs 4 \
    --reward-mode dense \
    --seed 42
```

### BC pre-training + PPO fine-tuning

```bash
python scripts/train_bc.py \
    --n-demo-games 200 \
    --bc-epochs 50 \
    --ppo-steps 500_000 \
    --seed 42
```

### Ablation study (4 conditions × 3 seeds)

```bash
python scripts/ablation_study.py \
    --total-timesteps 1_000_000 \
    --seeds 42 123 7 \
    --n-envs 4 \
    --experiment-name ablation_s6
```

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ppo_wingspan_best.zip \
    --n-games 500 \
    --also-vs-greedy
```

### Run the Rule Oracle

```bash
# First, ingest the rulebook (requires PDF in data/rulebooks/)
python scripts/ingest_rulebook.py --game wingspan

# Evaluate accuracy on golden dataset
python scripts/eval_rule_oracle.py \
    --game wingspan \
    --golden data/golden_rules/wingspan_rules_qa.json
```

---

## Project structure

```
TabletopOracle/
├── src/
│   ├── games/
│   │   ├── base/               # ABCs: GameEngine, GameState, Action
│   │   ├── wingspan/           # Wingspan engine, cards, rules, rewards
│   │   └── seven_wonders_duel/ # 7 Wonders Duel engine, cards, rules, rewards
│   ├── envs/
│   │   ├── wingspan_env.py
│   │   └── seven_wonders_duel_env.py
│   ├── agents/
│   │   ├── baselines.py        # RandomAgent, GreedyAgent
│   │   ├── ppo_agent.py        # build_maskable_ppo(), WinRateCallback
│   │   └── bc_agent.py         # BehavioralCloningTrainer
│   ├── imitation/
│   │   ├── demo_buffer.py      # DemonstrationBuffer, SyntheticDemoGenerator
│   │   ├── bga_parser.py       # BGALogParser
│   │   └── tts_parser.py       # TTSLogParser
│   ├── eval/
│   │   ├── game_runner.py      # run_game() helper
│   │   ├── metrics.py          # win_rate, avg_score, score_distribution, etc.
│   │   ├── tournament.py       # EloTable, Tournament (round-robin Elo)
│   │   └── llm_judge.py        # LLMJudge, build_game_transcript
│   └── oracle/
│       ├── claude_client.py    # ClaudeClient with disk cache
│       ├── ingestion.py        # PDF → ChromaDB
│       ├── retriever.py        # RAG retrieval
│       └── rule_oracle.py      # RuleOracle (answers rule questions)
├── scripts/
│   ├── train_ppo.py
│   ├── train_bc.py
│   ├── ablation_study.py
│   ├── evaluate.py
│   ├── ingest_rulebook.py
│   └── eval_rule_oracle.py
├── tests/                      # pytest suite (369 tests)
├── data/
│   ├── card_catalogs/          # wingspan_birds.csv, seven_wonders_duel_cards.json
│   ├── rulebooks/              # PDF rulebooks (not committed)
│   └── golden_rules/           # Rule Oracle evaluation dataset
├── experiments/                # Auto-created per training run
├── checkpoints/                # Saved model checkpoints
└── paper/
    └── paper_outline.md        # AAAI 2026 workshop paper draft
```

---

## Environment variables

Create `.env` from the example:

```bash
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_DIR=./data/chroma_db
RULEBOOK_DIR=./data/rulebooks
CARD_CATALOG_DIR=./data/card_catalogs
GAME_LOGS_DIR=./data/game_logs
EXPERIMENTS_DIR=./experiments
CHECKPOINTS_DIR=./checkpoints
LOG_LEVEL=INFO
CACHE_DIR=./data/cache
```

---

## Reproducing experiments

All results are reproducible.  Each run creates a unique numbered directory under
`experiments/` with `config.json`, `results.json`, and `training_curves.png`.
Seeds are fixed at all levels: `torch.manual_seed`, `np.random.seed`, `random.seed`,
and `env.reset(seed=...)`.

---

## Adding a new game

1. Create `src/games/<game>/` — `cards.py`, `state.py`, `actions.py`, `rules.py`,
   `engine.py`, `rewards.py`.
2. Create `src/envs/<game>_env.py` — `gym.Env` subclass with `action_masks()`.
3. Verify with `check_env(YourEnv())`.
4. Reuse `build_maskable_ppo()`, `BehavioralCloningTrainer`, `Tournament`, and
   `LLMJudge` unchanged.

See [src/games/seven_wonders_duel/](src/games/seven_wonders_duel/) for a worked example.

---

## Citation

```bibtex
@inproceedings{perezbravo2026tabletop,
  title     = {TabletopOracle: Combining Rule Grounding and Reinforcement Learning
               for Modern Board Games},
  author    = {P\'{e}rez Bravo, Octavio},
  booktitle = {AAAI 2026 Workshop on Games and Agents},
  year      = {2026},
}
```

---

## License

Code: MIT.  Card catalogs: derived from publicly available game data; cite the
original publishers.  Rulebook PDFs: not distributed.

---

*OPB AI Mastery Lab · From pipeline to decision.*
