# TabletopOracle

**AI agents that learn to play modern board games from natural-language rulebooks — without hardcoded game logic.**

TabletopOracle is a research system that combines Retrieval-Augmented Generation (RAG) for rule interpretation with Masked Proximal Policy Optimization (MaskablePPO) for strategy learning. Given only a PDF rulebook and a card catalog, the system produces a fully playable game environment and a trained agent capable of beating greedy baselines.

**Current games:** Wingspan · 7 Wonders Duel

**Research target:** AAAI 2026 Workshop on Games & Agents / NeurIPS 2025 Games Workshop

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/TableTop_Oracle.git
cd TableTop_Oracle
pip install -e .

# 2. Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Place your Wingspan rulebook PDF at data/rulebooks/wingspan.pdf
#    then ingest it
python scripts/ingest_rulebook.py --game wingspan --pdf data/rulebooks/wingspan.pdf

# 4. Verify everything works
pytest tests/ -v --tb=short

# 5. Train an agent (smoke test — 10k steps, ~1 min)
python scripts/train_ppo.py --total-timesteps 10000 --n-envs 2 --seed 42 --no-tensorboard

# 6. Run the full ablation study (~4 hours)
python scripts/ablation_study.py \
    --total-timesteps 1000000 --seeds 42 123 7 \
    --n-envs 4 --experiment-name ablation_s6
```

That's it. Results land in `experiments/exp_NNN_ablation_s6/results.json`.

---

## Why this matters

Teaching an AI to play board games is a longstanding benchmark for general reasoning. Prior work (AlphaGo, MuZero, OpenAI Five) either relies on perfect simulators hand-coded by domain experts, or on games with simple, fixed rule sets. Modern hobby board games present a harder challenge:

- **Rules are written in natural language**, with edge cases and exceptions that are ambiguous without context
- **Cards have unique powers** that interact with each other in ways no fixed rule set can enumerate
- **The action space is dynamic** — what moves are legal depends on the board state in complex ways
- **Multiple games exist**, each with entirely different mechanics — a general system should not be re-engineered from scratch for each one

TabletopOracle addresses all four challenges. The Rule Oracle interprets natural-language rules via RAG. The LegalMoveValidator enforces legality in Python without hardcoded game knowledge. MaskablePPO ensures the agent only explores legal actions, learning strategy rather than spending gradient steps on illegal moves. The architecture is game-agnostic: adding a new game required only ~18% new code when generalizing from Wingspan to 7 Wonders Duel.

**The key cost insight:** naively calling an LLM at every step of 1M-step RL training would cost ~$3,000 USD. TabletopOracle's design keeps the total API cost of the entire project at ~$15 by using LLMs only at development time, never inside the training loop.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  RULE LAYER                                                          │
│                                                                      │
│  PDF Rulebook ──► PyMuPDF chunker ──► ChromaDB (all-MiniLM-L6-v2)  │
│                                            ▲                         │
│                              Rule Oracle (Claude RAG)               │
│                              ↳ answers rule questions at dev time   │
│                              ↳ validates edge cases during engine    │
│                                synthesis — never during training     │
├──────────────────────────────────────────────────────────────────────┤
│  ENVIRONMENT LAYER                                                   │
│                                                                      │
│  GameState (Pydantic v2, immutable)                                  │
│     ↓                                                                │
│  LegalMoveValidator ──► get_legal_actions() — pure Python, no LLM  │
│     ↓                                                                │
│  GameEngine.step() ──► new GameState                                │
│     ↓                                                                │
│  gym.Env wrapper ──► action_masks() ──► MaskablePPO                │
├──────────────────────────────────────────────────────────────────────┤
│  AGENT LAYER                                                         │
│                                                                      │
│  SyntheticDemoGenerator (GreedyAgent expert)                         │
│     ↓                                                                │
│  BehavioralCloningTrainer ──► warm-start policy weights             │
│     ↓                                                                │
│  MaskablePPO fine-tuning ──► WinRateCallback ──► checkpoints        │
└──────────────────────────────────────────────────────────────────────┘
```

### Architectural decisions

**LLM not in training loop.**
The Rule Oracle is used during development to synthesize and validate the Python rule engine. Once the engine is complete, training runs in pure Python with no API calls. This is the single most important cost decision in the project.

**MaskablePPO over standard PPO.**
Without action masking, the agent wastes gradient steps learning to avoid illegal moves rather than learning strategy. `sb3_contrib.MaskablePPO` reads `action_masks()` from the environment at each step, zeroing out the logits of illegal actions before the policy update. This is not just an optimization — using standard PPO in this setting is a design flaw.

**Pydantic v2 for GameState.**
All game state is modeled as immutable Pydantic v2 objects. `model_copy(update={...})` produces new states without mutating the original. This eliminates aliasing bugs (a documented source of subtle errors in RL environments) and gives free JSON serialization for logging and replay.

**BC pre-training before PPO.**
Behavioral Cloning on synthetic demos (from a GreedyAgent expert) gives the policy a warm start. The agent enters PPO with a reasonable prior rather than exploring from random initialization. This reduces sample complexity and produces faster learning curves — directly measurable in the ablation study.

**ChromaDB with all-MiniLM-L6-v2.**
Chosen for consistency with existing infrastructure (OPB AI Mastery Lab stack). One collection per game (`rules_wingspan`, `rules_seven_wonders_duel`). Chunks are ~300 tokens with 50-token overlap to preserve context across section boundaries.

**Elo-based tournament evaluation.**
Win rate against a single baseline is insufficient to characterize agent quality. The tournament module runs round-robin matches between all agents (RandomAgent, GreedyAgent, BC-only, PPO-baseline, BC+PPO) and maintains an Elo rating. Player roles alternate every game to cancel first-mover advantage.

---

## Installation

### Requirements

- Python 3.11+
- pip
- An Anthropic API key (required for Rule Oracle; not needed to run training)
- ~2 GB disk space for ChromaDB and model checkpoints

### Step 1 — Clone and install

```bash
git clone https://github.com/your-org/TableTop_Oracle.git
cd TableTop_Oracle
pip install -e .
```

The `-e` flag installs in editable mode so changes to `src/` are reflected immediately without reinstalling.

### Step 2 — Create the `.env` file

```bash
cp .env.example .env   # if the example exists, otherwise create it manually
```

Edit `.env` and set your API key:

```
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
CHROMA_PERSIST_DIR=./data/chroma_db
RULEBOOK_DIR=./data/rulebooks
CARD_CATALOG_DIR=./data/card_catalogs
GAME_LOGS_DIR=./data/game_logs
EXPERIMENTS_DIR=./experiments
CHECKPOINTS_DIR=./checkpoints
LOG_LEVEL=INFO
CACHE_DIR=./data/cache
```

The `.env` file is in `.gitignore` and will never be committed.

### Step 3 — Verify the installation

```bash
python -c "from src.oracle.claude_client import ClaudeClient; print('Oracle: OK')"
python -c "from src.envs.wingspan_env import WingspanEnv; print('Wingspan env: OK')"
python -c "from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv; print('7WD env: OK')"
```

### Step 4 — Run the test suite

```bash
pytest tests/ -v --tb=short
```

All tests should pass. The suite covers 374 test cases across the full stack.

### Step 5 — Ingest a rulebook (required for Rule Oracle)

Place the rulebook PDF in `data/rulebooks/`. Then:

```bash
python scripts/ingest_rulebook.py --game wingspan --pdf data/rulebooks/wingspan.pdf
```

This chunks the PDF, embeds each chunk with `all-MiniLM-L6-v2`, and stores the vectors in ChromaDB at `data/chroma_db/`. You only need to do this once per game.

### Using Docker

```bash
docker build -t tabletop-oracle .
docker run --env-file .env tabletop-oracle pytest tests/ -v
```

---

## Usage

### Train a PPO agent from scratch

```bash
python scripts/train_ppo.py \
    --game wingspan \
    --total-timesteps 1_000_000 \
    --n-envs 4 \
    --reward-mode dense \
    --seed 42
```

The script creates `experiments/exp_NNN_ppo_wingspan_dense_seed42/` with `config.json`, `results.json`, and a final checkpoint in `checkpoints/`.

### BC pre-training followed by PPO fine-tuning

```bash
python scripts/train_bc.py \
    --n-demo-games 200 \
    --bc-epochs 50 \
    --ppo-steps 500_000 \
    --seed 42
```

BC trains on 200 synthetic games generated by the GreedyAgent, then the warm-started policy is fine-tuned with MaskablePPO.

### Ablation study (the core scientific contribution)

```bash
python scripts/ablation_study.py \
    --total-timesteps 1_000_000 \
    --seeds 42 123 7 \
    --n-envs 4 \
    --n-demo-games 200 \
    --bc-epochs 50 \
    --reward-mode dense \
    --experiment-name ablation_s6
```

Runs 4 conditions × 3 seeds:

| Variant | Description |
|---------|-------------|
| `baseline` | PPO from scratch, no BC, no RAG |
| `rag` | PPO + RAG oracle on edge cases (requires `--include-rag`) |
| `bc` | BC pre-train → PPO fine-tune, no RAG |
| `full` | BC pre-train → PPO + RAG (complete system) |

Results are written to `experiments/exp_NNN_ablation_s6/results.json`.

### Evaluate a trained checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ppo_wingspan_best.zip \
    --n-games 500 \
    --also-vs-greedy
```

Reports win rate vs. RandomAgent, win rate vs. GreedyAgent, average score (mean ± std), and rule violation rate.

### Evaluate the Rule Oracle

```bash
python scripts/eval_rule_oracle.py \
    --game wingspan \
    --golden data/golden_rules/wingspan_rules_qa.json
```

Measures accuracy on the golden Q&A dataset. Target: ≥ 0.80.

### Run a tournament

```python
from src.eval.tournament import Tournament
from src.agents.baselines import RandomAgent, GreedyAgent
from src.games.wingspan.engine import WingspanEngine

engine = WingspanEngine(seed=0)
agents = {"random": RandomAgent(seed=0), "greedy": GreedyAgent()}
t = Tournament(engine)
elo_table = t.run(agents, n_games_per_pair=200)
for name, rating in elo_table.standings():
    print(f"{name:20s}  Elo={rating:.1f}")
```

---

## Project structure

```
TabletopOracle/
├── src/
│   ├── games/
│   │   ├── base/                   # ABCs: GameEngine, GameState, Action
│   │   ├── wingspan/               # Full Wingspan implementation
│   │   │   ├── cards.py            # Bird card catalog loader
│   │   │   ├── state.py            # WingspanState, WingspanPlayerBoard
│   │   │   ├── actions.py          # WingspanAction, WingspanActionType
│   │   │   ├── rules.py            # LegalMoveValidator
│   │   │   ├── engine.py           # WingspanEngine (step, reset, score)
│   │   │   └── rewards.py          # sparse / dense reward modes
│   │   └── seven_wonders_duel/     # Full 7 Wonders Duel implementation
│   │       ├── cards.py            # Age/Wonder/Progress token catalog
│   │       ├── state.py            # SWDState, SWDPlayerBoard, pyramid layouts
│   │       ├── actions.py          # SWDAction, SWDActionType
│   │       ├── rules.py            # SWDLegalMoveValidator (dynamic trading)
│   │       ├── engine.py           # SWDEngine (3 win conditions)
│   │       └── rewards.py          # sparse / dense reward modes
│   ├── envs/
│   │   ├── wingspan_env.py         # WingspanEnv (gymnasium.Env)
│   │   ├── seven_wonders_duel_env.py  # SevenWondersDuelEnv (gymnasium.Env)
│   │   └── wrappers.py             # NormaliseObsWrapper
│   ├── agents/
│   │   ├── baselines.py            # RandomAgent, GreedyAgent
│   │   ├── ppo_agent.py            # build_maskable_ppo(), WinRateCallback
│   │   └── bc_agent.py             # BehavioralCloningTrainer, load_bc_weights_into_ppo
│   ├── imitation/
│   │   ├── demo_buffer.py          # DemonstrationBuffer, SyntheticDemoGenerator
│   │   ├── bga_parser.py           # BoardGameArena log parser
│   │   └── tts_parser.py           # Tabletop Simulator log parser
│   ├── eval/
│   │   ├── game_runner.py          # run_game() — shared helper
│   │   ├── metrics.py              # win_rate, avg_score, rule_violation_rate, etc.
│   │   ├── tournament.py           # EloTable, Tournament (round-robin)
│   │   └── llm_judge.py            # LLMJudge, build_game_transcript
│   └── oracle/
│       ├── claude_client.py        # ClaudeClient with disk cache + prompt caching
│       ├── ingestion.py            # PDF → ChromaDB pipeline
│       ├── retriever.py            # Semantic search over rule chunks
│       └── rule_oracle.py          # RuleOracle — answers rule questions via RAG
├── scripts/
│   ├── train_ppo.py                # Full PPO training run
│   ├── train_bc.py                 # BC pre-train + PPO fine-tune
│   ├── ablation_study.py           # 4-condition ablation (S6)
│   ├── evaluate.py                 # Evaluate a saved checkpoint
│   ├── ingest_rulebook.py          # PDF → ChromaDB ingestion
│   └── eval_rule_oracle.py         # Rule Oracle accuracy on golden dataset
├── tests/                          # 374 pytest tests
│   ├── test_wingspan_engine.py
│   ├── test_gym_env.py
│   ├── test_imitation_parser.py
│   ├── test_eval.py
│   ├── test_seven_wonders_duel.py
│   └── ...
├── data/
│   ├── card_catalogs/
│   │   ├── wingspan_birds.csv           # 214 bird cards
│   │   └── seven_wonders_duel_cards.json  # 69 age cards + 12 wonders + 10 tokens
│   ├── rulebooks/                  # PDF rulebooks — not committed to git
│   ├── golden_rules/               # Rule Oracle evaluation Q&A datasets
│   └── chroma_db/                  # ChromaDB vector store (auto-created)
├── experiments/                    # Auto-created per training run, never overwritten
├── checkpoints/                    # Saved model checkpoints
├── paper/
│   └── paper_outline.md            # AAAI 2026 workshop paper draft
├── Dockerfile
└── pyproject.toml
```

---

## Reproducing experiments

Reproducibility is a first-class requirement for published research. Every training run:

- Fixes seeds at all levels: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`, `env.reset(seed=seed)`
- Writes `config.json` with all hyperparameters before training starts
- Creates a new numbered directory (`exp_001_`, `exp_002_`, ...) — existing experiments are never overwritten
- Writes `results.json` with final metrics after training completes

To reproduce the main ablation:

```bash
python scripts/ablation_study.py \
    --total-timesteps 1_000_000 \
    --seeds 42 123 7 \
    --n-envs 4 \
    --experiment-name ablation_s6
```

---

## Adding a new game

The framework is designed for this. The game-specific code for 7 Wonders Duel reused ~82% of the existing infrastructure unchanged.

**Steps:**

1. **Implement the game module** under `src/games/<game>/`:
   - `cards.py` — data classes and catalog loader
   - `state.py` — Pydantic v2 `GameState` (immutable) and `PlayerBoard`
   - `actions.py` — `Action` subclass with all fields needed to describe a move
   - `rules.py` — `LegalMoveValidator` with `get_legal_actions(state)`
   - `engine.py` — `GameEngine` subclass: `reset()`, `step()`, `is_terminal()`, `compute_score()`
   - `rewards.py` — `compute_reward(state_before, state_after, player_id, done, winner, mode)`

2. **Wrap it as a gym environment** in `src/envs/<game>_env.py`:
   - Subclass `gymnasium.Env`
   - Define `observation_space` and `action_space`
   - Implement `action_masks()` — must be 100% consistent with `engine.get_legal_actions()`
   - Run `gymnasium.utils.env_checker.check_env(YourEnv())` — must pass with zero warnings

3. **Verify the full pipeline works unchanged:**
   ```python
   from src.agents.ppo_agent import build_maskable_ppo
   from src.agents.bc_agent import BehavioralCloningTrainer
   from src.eval.tournament import Tournament
   # All of these work with any game that follows the interface
   ```

4. **Ingest the rulebook:**
   ```bash
   python scripts/ingest_rulebook.py --game <game> --pdf data/rulebooks/<game>.pdf
   ```

See [src/games/seven_wonders_duel/](src/games/seven_wonders_duel/) for a complete worked example.

---

## Stack

| Component | Library | Version |
|-----------|---------|---------|
| RL | stable-baselines3 + sb3-contrib (MaskablePPO) | ≥ 2.3.0 |
| Game environments | gymnasium | ≥ 0.29.0 |
| Game state | pydantic | v2 ≥ 2.0 |
| LLM | anthropic (Claude) | ≥ 0.40.0 |
| Vector store | chromadb | ≥ 1.5.5 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | ≥ 3.0.0 |
| PDF parsing | pymupdf | ≥ 1.24.0 |
| Deep learning | torch | ≥ 2.2.0 |
| Testing | pytest + hypothesis | ≥ 8.0.0 / ≥ 6.0.0 |

**Default Claude model:** `claude-sonnet-4-6`

---

## FAQ

**Do I need an Anthropic API key to train agents?**
No. The API key is only required for the Rule Oracle (`ingest_rulebook.py`, `eval_rule_oracle.py`). Training (`train_ppo.py`, `ablation_study.py`) runs entirely in Python with no API calls. You can train without a key.

**How long does training take?**
On a modern CPU with 4 parallel envs, 1M steps takes roughly 4–5 hours for Wingspan and 3–4 hours for 7 Wonders Duel. A GPU does not significantly help here because the bottleneck is environment simulation, not neural network forward passes.

**Why MaskablePPO and not standard PPO?**
Without action masking, the agent wastes gradient steps learning to avoid illegal moves rather than learning strategy. In a game like Wingspan with ~200 possible actions per step but only 4–6 legal at any given state, standard PPO would spend the majority of training discovering which actions are illegal. This is a design flaw, not a tuning problem.

**What is the Rule Oracle accuracy?**
90% on the 50-question golden dataset (100% basic turn, 93% bird power, 90% edge case, 90% end-of-round, 60% exception). The knowledge base comprises 4 documents: the rulebook PDF, the official Stonemaier FAQ, and two targeted clarification files covering edge cases and bird power terminology. The 60% on exception questions is the practical ceiling — those rules are implicit board game conventions not written in any official document. This does not affect training since the Python rule engine runs independently.

**Why BC+PPO helps more in 7WD than in Wingspan?**
In Wingspan, both conditions converge at ~200k steps — BC reduces variance but not the performance ceiling. In 7 Wonders Duel (larger action space, 3 win conditions), BC+PPO reaches 0.80 WR vs 0.67 for PPO baseline at 1M steps. The GreedyAgent prior eliminates unproductive exploration that PPO from scratch spends significant budget on in more complex games.

**Can I add a new game?**
Yes. See [Adding a new game](#adding-a-new-game). The 7 Wonders Duel implementation required ~18% new code — everything else (agent, training, evaluation, tournament) was reused unchanged.

**The training crashed and left a partial experiment directory. How do I re-run?**
Delete the partial directory and re-run the script. Each script creates the experiment directory at startup with `exist_ok=False`, so if the directory already exists it will fail. Delete it manually:
```bash
rm -rf experiments/ppo_7wd_dense_seed42
```

**How do I reproduce the paper results exactly?**
```bash
python scripts/ablation_study.py \
    --total-timesteps 1000000 --seeds 42 123 7 \
    --n-envs 4 --n-demo-games 200 --bc-epochs 50 \
    --reward-mode dense --experiment-name ablation_s6
```
All seeds are fixed at every level. Results are deterministic given the same hardware and library versions (see `pyproject.toml` for pinned versions).

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

Code: MIT. Card catalogs: derived from publicly available game data; cite the original publishers. Rulebook PDFs are not distributed — provide your own copy.

---

*OPB AI Mastery Lab · From pipeline to decision. — Octavio Pérez Bravo*
