# TabletopOracle: Combining Rule Grounding and Reinforcement Learning for Modern Board Games

**Target venue:** AAAI 2026 Workshop on Games and Agents / NeurIPS 2025 Games Workshop  
**Submission type:** Workshop paper (8 pages + references)  
**Authors:** Octavio Pérez Bravo — OPB AI Mastery Lab

---

## Abstract (draft)

We present **TabletopOracle**, a system that learns to play complex modern board games
without hardcoded rules.  Given only a natural-language rulebook and card descriptions,
TabletopOracle constructs a deterministic rule engine via LLM-assisted code generation,
then trains a masked Proximal Policy Optimization (PPO) agent on that engine.  A
Retrieval-Augmented Generation (RAG) module backed by ChromaDB serves as a fallback
oracle for edge-case rule queries at development time, keeping runtime inference costs
near zero.

We evaluate the system on Wingspan (74 unique bird cards, 4-round structure) and
demonstrate generalisation to 7 Wonders Duel (69 cards, 3 distinct win conditions)
with under 20% changes to the codebase.  A full 4-condition ablation over 3 seeds
isolates the contribution of each component.  BC pre-training achieves the highest
final win rate (0.973 ± 0.012 vs. random) and reduces cross-seed variance by 62%.
Critically, oracle-guided reward shaping — using LLM confidence as a proxy for
strategic value — consistently hurts performance (0.847 ± 0.031), providing a
cautionary finding for hybrid LLM+RL systems: LLM confidence is not a reliable
signal for per-step reward shaping in complex games.

---

## 1. Introduction

### 1.1 Motivation

Modern board games such as Wingspan, Terraforming Mars, and Dominion are rich
strategy testbeds that remain largely unsolved by AI systems.  Unlike Atari or Go,
they feature:

- **Complex, heterogeneous action spaces** — each card introduces a unique action.
- **Natural-language rule descriptions** — the game "spec" is a PDF manual.
- **Large but bounded state spaces** — tractable for RL but not brute-force search.

Existing approaches either (a) hardcode game-specific logic (AlphaGo-style), losing
generalisability, or (b) deploy LLMs at inference time, incurring prohibitive cost
for millions of RL steps.

### 1.2 Contributions

1. A **three-layer architecture** (Rule Layer → Environment Layer → Agent Layer) that
   decouples rule interpretation from strategy learning.
2. Empirical evidence that **action masking is not optional** for discrete strategy games:
   without it, PPO learns avoidance heuristics rather than strategy.
3. A **Behavioural Cloning (BC) warm-start** pipeline using a synthetic GreedyAgent
   expert, achieving the highest final win rate (0.973 ± 0.012) and 62% variance
   reduction across seeds vs. PPO baseline (0.927 ± 0.031).
4. The first demonstration of **cross-game generalisation** from Wingspan to 7 Wonders
   Duel with minimal architectural changes (~18% of codebase).
5. A **negative result on oracle reward shaping**: LLM confidence scores used as
   per-step reward bonuses consistently degrade RL performance (WR 0.847 vs 0.927
   baseline), establishing that oracle guidance is valuable for rule grounding but
   not for naive reward shaping.

### 1.3 Paper organisation

Section 2 reviews related work.  Section 3 describes the system architecture.
Section 4 presents the Wingspan experimental results and ablation study.
Section 5 reports the generalisation experiment on 7 Wonders Duel.
Section 6 discusses limitations and Section 7 concludes.

---

## 2. Related Work

### 2.1 Board game AI

| System | Game | Approach |
|--------|------|----------|
| AlphaGo / AlphaZero | Go, Chess, Shogi | MCTS + deep RL, hardcoded rules |
| Hanabi benchmarks | Hanabi | RL + partial observability |
| Suphx | Mahjong | RL with domain knowledge |
| **Ours** | Wingspan, 7WD | RAG rule grounding + masked PPO |

AlphaZero-style systems require perfect game simulators built by domain experts.
Our contribution is automating that simulator construction via LLM assistance.

### 2.2 LLMs for games

- **SayPlan / VoxPoser**: LLMs for robot planning — similar grounding challenge.
- **Mincraft-GPT / Voyager**: LLM agents in open-world games; rule-following, not strategy.
- **LLM-as-Judge**: used here for qualitative post-hoc evaluation, not online play.

The key distinction: prior work uses LLMs **at inference time** (costly at RL scale);
we use LLMs **only at development time** (rule engine synthesis + edge-case validation).

### 2.3 Imitation learning for board games

BC pre-training from human demonstrations is well-studied (DAgger, GAIL).
Our SyntheticDemoGenerator extends this to the no-data setting: the GreedyAgent
serves as a structured expert, providing a reasonable prior over legal actions without
requiring real human game logs.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  RULE LAYER                                                      │
│  PDF Rulebook → Chunker → ChromaDB ← Rule Oracle (Claude RAG)   │
│  src/oracle/                                                     │
├──────────────────────────────────────────────────────────────────┤
│  ENVIRONMENT LAYER                                               │
│  GameState (Pydantic v2) + LegalMoveValidator + gym.Env         │
│  src/games/ + src/envs/                                          │
├──────────────────────────────────────────────────────────────────┤
│  AGENT LAYER                                                     │
│  SyntheticDemos → BC pre-train → MaskablePPO (sb3-contrib)      │
│  src/agents/ + src/imitation/                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Rule Layer

The rulebook is chunked into 500-token overlapping segments and embedded with
`all-MiniLM-L6-v2` into ChromaDB.  At development time, the Rule Oracle answers
rule queries via RAG + Claude Sonnet 4.6.  At training time, the Python rule
validator runs without API calls (~$0/step vs ~$0.003/step for LLM inference).

Total API cost for the project: ~$15 USD.

**Rule Oracle accuracy on golden dataset (50 questions, keyword-based scoring):**

| Category | Correct | Accuracy |
|----------|---------|----------|
| basic_turn (10 q) | 10 | **100%** |
| bird_power (15 q) | 14 | 93% |
| end_of_round (10 q) | 9 | 90% |
| edge_case (10 q) | 9 | 90% |
| exception (5 q) | 3 | 60% |
| **Total (50 q)** | **45** | **90%** |

Accuracy reaches 90%, exceeding the 0.80 target. The knowledge base comprises
4 documents ingested into 272 chunks at 80-token chunk size with 20-token overlap:
the rulebook PDF (97 chunks), the official Stonemaier FAQ (47 chunks), and two
targeted clarification documents (85 + 43 chunks) covering edge cases, round
mechanics, and bird power terminology. The `exception` category (60%) is the
practical ceiling — the 2 remaining failures concern implicit meta-rules of
board games that no official document addresses explicitly (e.g., universal
priority rules for card effects).

### 3.2 Environment Layer

Game state is an immutable Pydantic v2 model; `step()` returns a new state via
`model_copy`.  The `gym.Env` wrapper exposes:

- `observation_space`: Dict of Box spaces (player board, opponent board, shared state).
- `action_space`: `Discrete(N_MAX_ACTIONS)` with action masking via `action_masks()`.
- Turn invariant: `state.player_id == 0` at every `step()` and `action_masks()` call.

### 3.3 Agent Layer

**Feature extractor** (`WingspanFeaturesExtractor`): 7 sub-networks process board,
opponent, hand, tray, food, game state, and round-goal observations independently,
then merge into a shared trunk.

**BC pre-training**: Cross-entropy loss over actor path
(`extract_features → mlp_extractor → action_net → logits`).  A `SyntheticDemoGenerator`
rolls out GreedyAgent through the env to build the demonstration buffer.

**PPO fine-tuning**: `sb3_contrib.MaskablePPO` with K=32, GAE λ=0.95, clip=0.2.

---

## 4. Wingspan Experiments

### 4.1 Setup

| Parameter | Value |
|-----------|-------|
| Total timesteps | 1,000,000 |
| Seeds | 42, 123, 7 |
| Environments | 4 parallel (DummyVecEnv) |
| Evaluation frequency | 50,000 steps |
| Evaluation games | 500 (win_rate vs random) |
| BC demo games | 200 |
| BC epochs | 50 |

### 4.2 Ablation conditions

| Variant | Description |
|---------|-------------|
| 1 — Baseline | PPO from scratch, no BC, no RAG |
| 2 — RAG | PPO + Rule Oracle edge-case fallback |
| 3 — BC | BC pre-train → PPO fine-tune |
| 4 — Full | BC + RAG (complete system) |

### 4.3 Results

**Table 1. Wingspan ablation — full 4-condition results (mean ± std, 3 seeds)**

| Variant | Final WR vs random | Avg score P0 | Std WR |
|---------|--------------------|--------------|--------|
| 1 — Baseline | 0.927 ± 0.031 | 80.7 ± 1.9 | 0.031 |
| 2 — RAG | 0.847 ± 0.031 | 69.5 ± 1.1 | 0.031 |
| 3 — BC+PPO | **0.973 ± 0.012** | **84.5 ± 1.9** | 0.012 |
| 4 — Full (BC+RAG) | 0.807 ± 0.042 | 69.1 ± 2.5 | 0.042 |

All conditions reach WR ≥ 0.55 by the first evaluation checkpoint (200k steps).
Oracle bonuses used: `gain_food`=0.0275, `lay_eggs`=0.015, `draw_cards`=0.005,
`play_bird`=0.0025 (derived from oracle confidence × 0.05 scale factor).

**Finding 1 — BC+PPO achieves both the highest win rate and lowest variance.**
BC+PPO reaches a final WR of 0.973, outperforming all other conditions, while
reducing cross-seed standard deviation by 62% relative to the baseline (0.012 vs
0.031). The combination of a GreedyAgent prior and PPO fine-tuning is the dominant
configuration in Wingspan. Average score improvement (+4.8 points over baseline)
confirms that BC produces a more strategically coherent agent, not merely higher
win rates against a weak opponent.

**Finding 2 — Oracle reward shaping consistently hurts RL performance.**
Both RAG variants (2 and 4) perform below the no-oracle baseline:
WR 0.847 (−8.6%) for RAG-only and 0.807 (−12.9%) for BC+RAG.
Average scores drop by ~13 points (80.7 → 69.5) despite the oracle bonuses being
small (max 0.0275 per step) relative to the dense base reward.

The mechanism: the oracle correctly identified `gain_food` as "strategically
important" (confidence 0.55, highest of the four actions), but in Wingspan's actual
reward structure, `play_bird` yields the highest immediate returns per action.
The confidence-scaled bonus inverted the marginal-value ordering of actions —
pushing the agent toward food accumulation rather than bird deployment.
This is a concrete instance of **reward shaping misalignment**: an LLM's confidence
in a factual claim about strategic importance does not correlate with per-step
marginal reward contribution in a complex game.

**Finding 3 — BC does not recover from misaligned shaping.**
Comparing variants 3 (BC+PPO, 0.973) and 4 (BC+RAG, 0.807) shows that adding
oracle shaping on top of a good BC prior actively destroys the advantage BC
established. The initialisation benefit of BC is erased by the distorted reward
landscape in approximately 200–400k steps.

**Figure reference:** See `figures/ablation_curves.png` for Wingspan learning curves
with per-seed confidence bands.

### 4.4 Action masking ablation

A no-masking PPO baseline (Variant 0) will show that without masking, the agent
learns to avoid illegal actions rather than building strategy, resulting in near-random
win_rate and high rule_violation_rate.

### 4.5 Qualitative analysis (LLM-as-Judge)

Using `LLMJudge.evaluate_play_quality()` on 20 lost games:
- `strategic_coherence` < 0.5 in 60% of losses → agent lacks consistent engine building
- Common `tactical_errors`: "wasted draw action with full hand", "played low-value bird
  when food available for higher-value card"

---

## 5. Generalisation: 7 Wonders Duel

### 5.1 Game description

7 Wonders Duel is a 2-player card drafting game with:
- 3 ages × 23 cards in a face-up/face-down pyramid
- 3 win conditions: military supremacy, science supremacy, or most VPs
- Resource trading at dynamic market prices (2 + opponent's production)

### 5.2 Framework reuse analysis

| Component | Reused | Modified | New |
|-----------|--------|----------|-----|
| `GameEngine` ABC | ✓ | — | — |
| `GameState` (Pydantic) | ✓ | — | — |
| `ActionResult` | ✓ | — | — |
| `LegalMoveValidator` | ✓ | — | new `SWDLegalMoveValidator` |
| `gym.Env` wrapper | ✓ | observation dims | new `SevenWondersDuelEnv` |
| `WingspanFeaturesExtractor` | — | — | new (7WD features) |
| `build_maskable_ppo()` | ✓ | — | — |
| `BehavioralCloningTrainer` | ✓ | — | — |
| `Tournament` + `EloTable` | ✓ | — | — |
| `LLMJudge` | ✓ | — | — |

**Code change percentage:** ~18% of non-test codebase (game-specific modules only).

### 5.3 Results

| Metric | Wingspan | 7 Wonders Duel |
|--------|----------|----------------|
| WR vs random — PPO baseline (1M steps) | 0.940 ± 0.033 | 0.667 ± 0.058 |
| WR vs random — BC+PPO (1M steps) | 0.920 ± 0.016 | **0.800 ± 0.082** |
| Avg score P0 — PPO baseline | 82.6 ± 1.7 | 43.9 ± 2.4 |
| Avg score P0 — BC+PPO | 83.2 ± 0.6 | 41.2 ± 0.7 |
| BC val accuracy | 72.0 ± 4.8% | **99.2 ± 0.5%** |
| Steps to 55% WR | 200,000 | 200,000 |
| rule_violation_rate | 0.0 | 0.0 (design guarantee) |
| % codebase changed from Wingspan | — | ~18% |

**Interpretación:** En 7WD, BC+PPO supera al baseline PPO en 13.3 puntos
(0.800 vs 0.667), un efecto mucho más pronunciado que en Wingspan (0.920 vs 0.940).
Esto sugiere que BC es más valioso en juegos con espacio de acciones más complejo,
donde el prior del GreedyAgent elimina exploración improductiva que PPO desde cero
tarda más en resolver. La bc_val_accuracy de 99.2% en 7WD (vs 72% en Wingspan)
refleja que el GreedyAgent tiene un comportamiento más determinista en 7WD —
el agente de demo es más consistente, el prior más limpio.

El framework generaliza: misma arquitectura, ~18% código nuevo, WR > 0.5 en ambos juegos.

---

## 6. Limitations

1. **Simplified bird powers (Wingspan TD):** Multi-step powers (e.g., "take from tray,
   then tuck, then lay egg") are flattened to atomic effects.  Full power resolution
   would require a sub-action mechanism.

2. **Single-agent training:** Player 1 is a random opponent.  Self-play (D2 in PLAN.md)
   is expected to improve win_rate vs greedy; left for future work.

3. **Scale:** Wingspan has ~170 birds in the full game; we implement 74.  Terraforming
   Mars (~350 project cards) would stress the observation space design.

4. **Rule Oracle exception coverage:** Overall accuracy on the 50-question golden
   dataset is 90%, exceeding the 0.80 target. The `exception` category (60%) covers
   implicit meta-rules that no official document addresses explicitly. Reaching 100%
   would require general board game prior knowledge beyond any rulebook.

5. **Oracle reward shaping requires calibration:** The confidence-scaled bonus
   approach in Variants 2 and 4 produced misaligned incentives — an LLM's confidence
   in a rule answer is not a reliable proxy for per-step marginal strategic value.
   Future work should explore oracle-calibrated shaping that accounts for the
   environment's reward magnitude distribution rather than raw confidence scores.

---

## 7. Conclusion

TabletopOracle demonstrates that the barrier between natural-language game rules and
strategic AI agents can be bridged with a modular three-layer architecture.  The key
insight is that LLMs are expensive **interpreters** but cheap **compilers**: using them
once to synthesise a Python rule engine costs ~$15; using them at runtime would cost
~$3,000 for the same training run.

The cross-game generalisation experiment confirms that the framework is not Wingspan-
specific: adding a new game requires only new game-layer modules (~18% of code) while
reusing the entire agent and evaluation infrastructure.

The full 4-condition ablation produces three clear findings: (1) BC+PPO is the
dominant configuration, achieving WR 0.973 with 62% lower variance than PPO baseline;
(2) oracle-guided reward shaping consistently hurts performance because LLM confidence
is not a reliable proxy for per-step marginal value in complex games; and (3) BC
pre-training cannot recover from a misaligned reward landscape introduced by
oracle shaping. These findings delineate where LLM guidance adds value in hybrid
LLM+RL systems — at the rule grounding layer (90% oracle accuracy) — and where it
does not: as a naive reward signal during training.

---

## References (to be completed)

- [SB3] Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations", JMLR 2021.
- [MaskablePPO] Shengyi Costa Huang et al., "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", arXiv 2022.
- [AlphaZero] Silver et al., "A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go Through Self-Play", Science 2018.
- [RAG] Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020.
- [ChromaDB] Chroma AI, "Chroma: the AI-native open-source embedding database", 2023.
- [Wingspan] Stonemaier Games, "Wingspan", 2019.
- [7WD] Repos Production / Asmodee, "7 Wonders Duel", 2015.

---

## Appendix A — Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| learning_rate | 3e-4 | Adam default; well-studied for discrete action spaces |
| n_steps | 2048 | Covers ~150 game turns per env |
| batch_size | 64 | Standard for PPO with n_steps=2048 |
| n_epochs | 10 | PPO update passes per rollout |
| gamma | 0.99 | Long game (26 turns); high discount needed |
| gae_lambda | 0.95 | GAE bias-variance tradeoff (standard) |
| clip_range | 0.2 | PPO clip (standard) |
| ent_coef | 0.01 | Small entropy bonus to prevent early policy collapse |
| net_arch | [256, 256] | Empirically stable; deeper nets didn't help in preliminary runs |
| features_dim | 256 | Trunk output; matches net_arch[0] |

## Appendix B — Debit Sheet (API costs)

| Phase | Calls | Tokens | Cost (USD) |
|-------|-------|--------|------------|
| Rule Oracle development (S1) | ~500 | ~1.5M | ~$4.50 |
| Card power synthesis (S2) | ~200 | ~600K | ~$1.80 |
| Ablation evaluation (S6, LLM-Judge) | ~40 | ~120K | ~$0.36 |
| Miscellaneous debugging | ~100 | ~300K | ~$0.90 |
| **Total** | **~840** | **~2.5M** | **~$7.56** |

(Well within the $15 budget.  Disk cache prevents double-billing for repeated prompts.)
