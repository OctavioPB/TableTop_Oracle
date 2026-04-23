# TabletopOracle

**AI agents that learn to play modern board games from natural-language rulebooks — without hardcoded game logic.**

TabletopOracle is a research system that combines Retrieval-Augmented Generation (RAG) for rule interpretation with Masked Proximal Policy Optimization (MaskablePPO) for strategy learning. Given only a PDF rulebook and a card catalog, the system produces a fully playable game environment and a trained agent capable of beating greedy baselines.

**Current games:** Wingspan · 7 Wonders Duel · Splendor

**Research target:** AAAI 2026 Workshop on Games & Agents / NeurIPS 2025 Games Workshop

---

## Results at a glance

| Game | Action space | PPO baseline WR | BC+PPO WR | Steps to WR ≥ 0.9 |
|---|---|---|---|---|
| Wingspan | ~150 | 0.927 ± 0.025 | **0.973 ± 0.009** | 200k |
| 7 Wonders Duel | ~120 | 0.667 ± 0.058 | **0.800 ± 0.082** | >1M |
| Splendor | 60 | 0.950 ± 0.041 | **0.967 ± 0.024** | 50k (baseline) |

All results: mean ± std over 3 seeds, 1M training steps, agent vs. random opponent.  
Total API cost for the complete project: **$7.56 USD**.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/TableTop_Oracle.git
cd TableTop_Oracle
pip install -e .

# 2. Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Verify everything works
pytest tests/ -v --tb=short

# 4. Train a Splendor agent (fastest convergence, ~1 hour on CPU)
python scripts/train_ppo.py \
    --game splendor --total-timesteps 1000000 \
    --n-envs 4 --seed 42

# 5. Or train with BC pre-training across 3 seeds
python scripts/train_bc_ppo.py \
    --game splendor --seeds 42 123 7 \
    --n-demo-games 200 --bc-epochs 50

# 6. Run the full Wingspan ablation (~4 hours)
python scripts/ablation_study.py \
    --total-timesteps 1000000 --seeds 42 123 7 \
    --n-envs 4 --experiment-name ablation_wingspan
```

Results land in `experiments/<exp_name>/results.json`.

---

## Why this matters

Teaching an AI to play board games is a longstanding benchmark for general reasoning. Prior work (AlphaGo, MuZero) either relies on perfect simulators hand-coded by domain experts, or on games with simple, fixed rule sets. Modern hobby board games present a harder challenge:

- **Rules are written in natural language**, with edge cases and exceptions that are ambiguous without context
- **Cards have unique powers** that interact with each other in ways no fixed rule set can enumerate
- **The action space is dynamic** — legal moves depend on the board state in complex ways
- **Multiple games exist** with entirely different mechanics — a general system should not be re-engineered from scratch for each one

TabletopOracle addresses all four challenges. The Rule Oracle interprets natural-language rules via RAG. The `LegalMoveValidator` enforces legality in Python without hardcoded game knowledge. `MaskablePPO` ensures the agent only explores legal actions. The architecture is game-agnostic: adding a new game required only ~15–18% new code in each case.

**The key cost insight:** naively calling an LLM at every step of 1M-step RL training would cost ~$3,000 USD. TabletopOracle uses LLMs only at development time, never inside the training loop, keeping the total project API cost at $7.56 USD.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  RULE LAYER                                         (offline, ~$15)  │
│                                                                      │
│  PDF Rulebook ──► PyMuPDF chunker ──► ChromaDB (all-MiniLM-L6-v2)  │
│                                            ▲                         │
│                              Rule Oracle (Claude Sonnet 4.6 + RAG)  │
│                              ↳ answers rule questions at dev time   │
│                              ↳ NEVER called during RL training       │
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
│  AGENT LAYER                                  (game-agnostic)        │
│                                                                      │
│  SyntheticDemoGenerator (game-specific greedy expert)                │
│     ↓                                                                │
│  BehavioralCloningTrainer ──► warm-start policy weights             │
│     ↓                                                                │
│  MaskablePPO fine-tuning ──► WinRateCallback ──► checkpoints        │
└──────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

**LLM not in the training loop.**
The Rule Oracle is used during development to synthesise and validate the Python rule engine. Once the engine is complete, training runs in pure Python with zero API calls. This is the single most important cost decision in the project: $7.56 vs. ~$3,000.

**MaskablePPO over standard PPO.**
Without action masking, the agent wastes gradient steps learning to avoid illegal moves rather than learning strategy. `sb3_contrib.MaskablePPO` reads `action_masks()` from the environment at each step, zeroing out illegal action logits before the policy update. Preliminary experiments confirmed that unmasked PPO fails to surpass random performance after 200k steps in these environments. This is a design requirement, not an optimisation.

**Pydantic v2 for immutable GameState.**
All game state is modelled as immutable Pydantic v2 objects. `model_copy(update={...})` produces new states without mutating the original, eliminating aliasing bugs and providing free JSON serialisation for logging and replay.

**BC pre-training before PPO.**
A `SyntheticDemoGenerator` rolls out a game-specific `GreedyAgent` to produce expert demonstrations without any human data. `BehavioralCloningTrainer` warms up the policy before PPO begins. This reduces cross-seed variance by 71% in Wingspan (std 0.009 vs. 0.025). Note: BC benefit depends on demo quality — a greedy heuristic with ~34% validation accuracy (Splendor) delays convergence relative to a strong one at ~72% (Wingspan). Calibrate before committing.

**Elo-based tournament evaluation.**
Win rate against a single baseline understates agent quality. The tournament module runs round-robin matches between all agents and maintains an Elo rating. Player roles alternate every game to cancel first-mover advantage.

---

## Installation

### Requirements

- Python 3.11+
- An Anthropic API key (required for Rule Oracle only; not needed for training)
- ~2 GB disk space for ChromaDB and model checkpoints

### Step 1 — Clone and install

```bash
git clone https://github.com/your-org/TableTop_Oracle.git
cd TableTop_Oracle
pip install -e .
```

### Step 2 — Create the `.env` file

```
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
CHROMA_PERSIST_DIR=./data/chroma_db
RULEBOOK_DIR=./data/rulebooks
CARD_CATALOG_DIR=./data/card_catalogs
EXPERIMENTS_DIR=./experiments
CHECKPOINTS_DIR=./checkpoints
LOG_LEVEL=INFO
CACHE_DIR=./data/cache
```

The `.env` file is in `.gitignore` and will never be committed.

### Step 3 — Verify the installation

```bash
python -c "from src.envs.wingspan_env import WingspanEnv; print('Wingspan: OK')"
python -c "from src.envs.seven_wonders_duel_env import SevenWondersDuelEnv; print('7WD: OK')"
python -c "from src.envs.splendor_env import SplendorEnv; print('Splendor: OK')"
```

### Step 4 — Run the test suite

```bash
pytest tests/ -v --tb=short
```

### Step 5 — Ingest a rulebook (required for Rule Oracle only)

```bash
python scripts/ingest_rulebook.py --game wingspan --pdf data/rulebooks/wingspan.pdf
```

---

## Usage

### Train a PPO agent

```bash
# Splendor (fastest: converges at ~50k steps)
python scripts/train_ppo.py --game splendor --total-timesteps 1000000 --n-envs 4 --seed 42

# 7 Wonders Duel
python scripts/train_ppo.py --game seven_wonders_duel --total-timesteps 1000000 --n-envs 4 --seed 42

# Wingspan
python scripts/train_ppo.py --game wingspan --total-timesteps 1000000 --n-envs 4 --seed 42
```

Each run creates `experiments/exp_NNN_ppo_<game>_dense_seed<N>/` with `config.json`, `results.json`, and a final checkpoint.

### BC pre-training + PPO fine-tuning (3 seeds)

```bash
python scripts/train_bc_ppo.py \
    --game wingspan \
    --seeds 42 123 7 \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --n-demo-games 200 \
    --bc-epochs 50 \
    --experiment-name bc_ppo_wingspan
```

Supported for all three games. Results are written to `experiments/<experiment-name>/results.json`.

### Wingspan ablation study (4 conditions × 3 seeds)

```bash
python scripts/ablation_study.py \
    --total-timesteps 1000000 \
    --seeds 42 123 7 \
    --n-envs 4 \
    --n-demo-games 200 \
    --bc-epochs 50 \
    --reward-mode dense \
    --experiment-name ablation_wingspan
```

| Variant | Description |
|---|---|
| `baseline` | MaskablePPO from random initialisation, no BC, no RAG shaping |
| `rag` | MaskablePPO with oracle-confidence reward bonuses |
| `bc_ppo` | BC warm-start → MaskablePPO fine-tune, no oracle shaping |
| `full` | BC warm-start → MaskablePPO with oracle shaping |

### Evaluate a saved checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ppo_splendor_final.zip \
    --game splendor \
    --n-games 500
```

### Evaluate the Rule Oracle

```bash
python scripts/eval_rule_oracle.py \
    --game wingspan \
    --golden data/golden_rules/wingspan_rules_qa.json
```

Target accuracy: ≥ 0.80. Current Wingspan accuracy: **90%** (50-question golden dataset).

---

## Project structure

```
TabletopOracle/
├── src/
│   ├── games/
│   │   ├── base/                   # ABCs: GameEngine, GameState, Action
│   │   ├── wingspan/               # Wingspan (74 birds, 4 rounds)
│   │   │   ├── cards.py            # Bird card catalog loader
│   │   │   ├── state.py            # WingspanState, WingspanPlayerBoard
│   │   │   ├── actions.py          # WingspanAction, WingspanActionType
│   │   │   ├── rules.py            # LegalMoveValidator
│   │   │   ├── engine.py           # WingspanEngine
│   │   │   └── rewards.py          # sparse / dense reward modes
│   │   ├── seven_wonders_duel/     # 7 Wonders Duel (69 cards, 3 win conditions)
│   │   │   ├── cards.py
│   │   │   ├── state.py            # SWDState, pyramid layout
│   │   │   ├── actions.py
│   │   │   ├── rules.py            # SWDLegalMoveValidator (dynamic trading)
│   │   │   ├── engine.py           # SWDEngine
│   │   │   └── rewards.py
│   │   └── splendor/               # Splendor (45 cards, nobles, gem economy)
│   │       ├── cards.py            # Development cards + noble tiles
│   │       ├── state.py            # SplendorState, SplendorPlayerBoard
│   │       ├── actions.py          # Discrete(60) action space
│   │       ├── engine.py           # SplendorEngine (rules + noble auto-claim)
│   │       └── rewards.py
│   ├── envs/
│   │   ├── wingspan_env.py         # WingspanEnv (gymnasium.Env)
│   │   ├── seven_wonders_duel_env.py
│   │   ├── splendor_env.py         # SplendorEnv (gymnasium.Env)
│   │   └── wrappers.py
│   ├── agents/
│   │   ├── baselines.py            # RandomAgent, GreedyAgent
│   │   ├── ppo_agent.py            # build_maskable_ppo(), WinRateCallback
│   │   ├── bc_agent.py             # BehavioralCloningTrainer
│   │   ├── encoders.py             # WingspanFeaturesExtractor, SWDFeaturesExtractor
│   │   └── splendor_extractor.py   # SplendorFeaturesExtractor
│   ├── imitation/
│   │   ├── demo_buffer.py          # DemonstrationBuffer, SyntheticDemoGenerator
│   │   ├── bga_parser.py           # BoardGameArena log parser
│   │   └── tts_parser.py           # Tabletop Simulator log parser
│   ├── eval/
│   │   ├── metrics.py              # win_rate, avg_score, rule_violation_rate
│   │   ├── tournament.py           # EloTable, Tournament (round-robin)
│   │   └── llm_judge.py            # LLMJudge, build_game_transcript
│   └── oracle/
│       ├── claude_client.py        # ClaudeClient with disk cache
│       ├── ingestion.py            # PDF → ChromaDB pipeline
│       ├── retriever.py            # Semantic search over rule chunks
│       └── rule_oracle.py          # RuleOracle — answers rule questions via RAG
├── scripts/
│   ├── train_ppo.py                # PPO training (all 3 games)
│   ├── train_bc_ppo.py             # BC pre-train + PPO fine-tune (all 3 games)
│   ├── ablation_study.py           # 4-condition ablation (Wingspan)
│   ├── evaluate.py                 # Evaluate a saved checkpoint
│   ├── ingest_rulebook.py          # PDF → ChromaDB ingestion
│   ├── eval_rule_oracle.py         # Rule Oracle accuracy on golden dataset
│   └── plot_ablation.py            # Generate ablation figures
├── tests/
│   ├── test_wingspan_engine.py
│   ├── test_gym_env.py
│   ├── test_seven_wonders_duel.py
│   ├── test_splendor.py            # 16 tests covering engine + env
│   └── ...
├── data/
│   ├── rulebooks/                  # PDF rulebooks — not committed to git
│   ├── golden_rules/               # Rule Oracle evaluation Q&A datasets
│   └── chroma_db/                  # ChromaDB vector store (auto-created)
├── experiments/                    # Auto-created per run, never overwritten
├── checkpoints/                    # Saved model checkpoints
├── paper/
│   ├── paper_draft.md              # Full workshop paper draft
│   └── figures/                    # All 6 paper figures (PNG)
└── pyproject.toml

```

---

## Adding a new game

The framework is designed for this. Each of the two target games required ~15–18% new code; everything else transferred unchanged.

1. **Implement the game module** under `src/games/<game>/`:
   - `cards.py` — data classes and catalog
   - `state.py` — Pydantic v2 `GameState` (immutable) and `PlayerBoard`
   - `actions.py` — `Action` subclass + flat integer index mapping
   - `engine.py` — `GameEngine`: `reset()`, `step()`, `is_terminal()`, `get_legal_actions()`
   - `rewards.py` — `compute_reward(prev_state, result, mode)`

2. **Wrap as a gym environment** in `src/envs/<game>_env.py`:
   - Subclass `gymnasium.Env`; define `observation_space` and `action_space`
   - Implement `action_masks()` — must be 100% consistent with `engine.get_legal_actions()`
   - Maintain `state.player_id == 0` at every call to `step()` and `action_masks()`
   - Pass `gymnasium.utils.env_checker.check_env(YourEnv())` with zero warnings

3. **Add a game-specific greedy heuristic** in `SyntheticDemoGenerator` for BC pre-training.
   Measure BC validation accuracy before training: if it falls below ~50%, the prior may delay convergence rather than help it.

4. **Register in `build_maskable_ppo()`** and **`train_bc_ppo.py`** with a new `game=` branch.

5. **Write tests** covering: reset state validity, legal actions non-empty, random game completes, `check_env` passes, obs shape and range, action mask consistency, and `player_id` invariant.

See [src/games/splendor/](src/games/splendor/) and [src/games/seven_wonders_duel/](src/games/seven_wonders_duel/) for complete worked examples.

---

## Reproducing experiments

Every training run fixes seeds at all levels — `torch.manual_seed`, `np.random.seed`, `random.seed`, and `env.reset(seed=...)` — and writes `config.json` with all hyperparameters before training starts. Experiment directories are never overwritten.

```bash
# Replicate Wingspan ablation (paper Table 1)
python scripts/ablation_study.py \
    --total-timesteps 1000000 --seeds 42 123 7 \
    --n-envs 4 --n-demo-games 200 --bc-epochs 50 \
    --reward-mode dense --experiment-name ablation_wingspan

# Replicate Splendor baseline (paper Table 4)
for seed in 42 123 7; do
  python scripts/train_ppo.py \
      --game splendor --total-timesteps 1000000 \
      --n-envs 4 --seed $seed
done

# Replicate Splendor BC+PPO (paper Table 4)
python scripts/train_bc_ppo.py \
    --game splendor --seeds 42 123 7 \
    --total-timesteps 1000000 --n-envs 4 \
    --n-demo-games 200 --bc-epochs 50 \
    --experiment-name bc_ppo_splendor
```

---

## Stack

| Component | Library | Version |
|---|---|---|
| RL | stable-baselines3 + sb3-contrib (MaskablePPO) | ≥ 2.3.0 |
| Game environments | gymnasium | ≥ 0.29.0 |
| Game state | pydantic v2 | ≥ 2.0 |
| LLM | anthropic (Claude Sonnet 4.6) | ≥ 0.40.0 |
| Vector store | chromadb | ≥ 1.5.5 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | ≥ 3.0.0 |
| PDF parsing | pymupdf | ≥ 1.24.0 |
| Deep learning | torch | ≥ 2.2.0 |
| Testing | pytest + hypothesis | ≥ 8.0.0 / ≥ 6.0.0 |

---

## FAQ

**Do I need an Anthropic API key to train agents?**
No. The key is only needed for the Rule Oracle (`ingest_rulebook.py`, `eval_rule_oracle.py`). All training scripts run entirely in Python with zero API calls.

**Which game should I start with?**
Splendor. It has the smallest action space (Discrete(60)), converges fastest (~50k steps to WR ≥ 0.9), and has no external data dependencies. It is the fastest way to verify a working installation end-to-end.

**How long does training take?**
On a modern CPU with 4 parallel envs and 1M steps: Splendor ~1–2 hours, Wingspan ~4–5 hours, 7 Wonders Duel ~3–4 hours. The bottleneck is environment simulation, not neural network forward passes; a GPU does not significantly reduce wall-clock time.

**Why MaskablePPO and not standard PPO?**
In Wingspan, only 4–6 of ~150 actions are legal at any given state (3–22% legal fraction). Without masking, PPO spends the majority of gradient steps learning to avoid illegal moves rather than learning strategy. Preliminary experiments confirmed unmasked PPO fails to surpass random performance after 200k steps. This is a design requirement, not a tuning choice.

**Why does BC+PPO help more in 7WD than in Wingspan?**
The GreedyAgent expert for Wingspan achieves ~72% validation accuracy, providing a useful but imperfect prior. In 7WD, the expert achieves ~99% accuracy (the pyramid structure leaves fewer equivalent choices), giving BC a cleaner signal and producing a larger performance gap (+13.3 pp vs. +4.6 pp at 1M steps). In Splendor, the greedy heuristic achieves only ~34% accuracy — low enough that the prior introduces biases the PPO must partially unlearn, delaying convergence from 50k to 200k steps despite a slightly higher final WR.

**Why does oracle reward shaping hurt performance?**
The Rule Oracle assigned higher confidence to `gain_food` than to `play_bird` because food mechanics are more extensively documented. However, `play_bird` yields higher marginal return per step in Wingspan's actual reward structure. Confidence-weighted bonuses inverted the marginal-value ordering of the two most important actions, pushing the agent toward suboptimal food accumulation. LLM confidence is an epistemic measure of documentation quality, not of strategic value; using it as a reward signal requires careful calibration against the environment's empirical reward distribution.

**What is the Rule Oracle accuracy?**
90% on the 50-question Wingspan golden dataset (100% basic turn, 93% bird power, 90% edge case and end-of-round, 60% exception). The 60% on exception questions reflects implicit board game meta-conventions not written in any official document — a practical ceiling that does not affect training since the Python rule engine runs independently.

**The training crashed and left a partial experiment directory.**
Delete it manually and re-run. Scripts use `exist_ok=False` at startup, so a pre-existing directory will cause an immediate failure rather than a silent overwrite.

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
