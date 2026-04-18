# TabletopOracle — PLAN.md
> Agente que aprende a jugar juegos de mesa complejos leyendo el manual
> en lenguaje natural y combinando LLM para reglas + RL para estrategia.

**Maintainer:** Octavio Pérez Bravo · OPB AI Mastery Lab  
**Target games:** Wingspan (MVP) → Terraforming Mars → Spirit Island  
**Stack:** Claude API · ChromaDB · Python gym · stable-baselines3 · FastAPI  

---

## Visión de arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        RULE LAYER                               │
│  PDF Rulebook → Chunker → ChromaDB ← Rule Oracle (Claude)      │
│                                ↑ RAG lookup on demand           │
├─────────────────────────────────────────────────────────────────┤
│                     ENVIRONMENT LAYER                           │
│  GameState (Pydantic) + ActionSpace + LegalMoveValidator        │
│  → gym.Env wrapper con action masking                           │
├─────────────────────────────────────────────────────────────────┤
│                       AGENT LAYER                               │
│  BC pre-train (BGA/TTS logs) → MaskablePPO (sb3-contrib)       │
│  State encoder: Dict obs → MLP trunk                            │
├─────────────────────────────────────────────────────────────────┤
│                      EVAL LAYER                                 │
│  Win rate / Rule violation rate / LLM-as-Judge / Elo proxy      │
└─────────────────────────────────────────────────────────────────┘
```

**Decisión de diseño crítica:** el LLM NO juega en tiempo real. Genera el
validador de reglas (código Python) asistido por prompting durante el
desarrollo. En inferencia, solo se consulta para edge cases no cubiertos
por el código generado. Esto hace el entrenamiento de RL viable sin
costos de API por step.

---

## Estructura del repositorio

```
tabletop-oracle/
├── PLAN.md                         ← este archivo
├── README.md
├── pyproject.toml
├── .env.example
│
├── data/
│   ├── rulebooks/                  ← PDFs originales
│   │   └── wingspan_rulebook.pdf
│   ├── card_catalogs/              ← CSVs de cartas por juego
│   │   └── wingspan_birds.csv
│   ├── game_logs/                  ← logs de BGA / TTS
│   │   ├── bga/
│   │   └── tts/
│   └── golden_rules/               ← golden dataset de Q&A de reglas
│       └── wingspan_rules_qa.json
│
├── src/
│   ├── oracle/                     ← RULE LAYER
│   │   ├── __init__.py
│   │   ├── ingestion.py            ← PDF → chunks → ChromaDB
│   │   ├── retriever.py            ← RAG retrieval
│   │   ├── rule_oracle.py          ← LLM + RAG → validación de reglas
│   │   └── prompts/
│   │       ├── rule_validator.txt
│   │       └── edge_case_resolver.txt
│   │
│   ├── games/                      ← ENVIRONMENT LAYER
│   │   ├── base/
│   │   │   ├── game_state.py       ← schema Pydantic abstracto
│   │   │   ├── action_space.py     ← ActionType enum + Action dataclass
│   │   │   └── engine.py           ← GameEngine ABC
│   │   └── wingspan/
│   │       ├── state.py            ← WingspanState : GameState
│   │       ├── actions.py          ← WingspanAction variants
│   │       ├── engine.py           ← WingspanEngine : GameEngine
│   │       ├── cards.py            ← BirdCard dataclass + catalog loader
│   │       ├── rules.py            ← validador de movimientos legales
│   │       └── rewards.py          ← reward shaping
│   │
│   ├── envs/                       ← GYM WRAPPER
│   │   ├── wingspan_env.py         ← gym.Env para Wingspan
│   │   └── wrappers.py             ← normalización, frame stack, etc.
│   │
│   ├── agents/                     ← AGENT LAYER
│   │   ├── ppo_agent.py            ← MaskablePPO wrapper
│   │   ├── bc_agent.py             ← Behavioral Cloning
│   │   ├── encoders.py             ← state → tensor
│   │   └── baselines.py            ← Random, Greedy agents
│   │
│   ├── imitation/                  ← IMITATION LEARNING
│   │   ├── bga_parser.py           ← parser de logs BGA
│   │   ├── tts_parser.py           ← parser de logs TTS
│   │   └── demo_buffer.py          ← DemonstrationBuffer
│   │
│   └── eval/                       ← EVAL LAYER
│       ├── metrics.py
│       ├── tournament.py
│       └── llm_judge.py
│
├── scripts/
│   ├── ingest_rulebook.py
│   ├── train_ppo.py
│   ├── train_bc.py
│   ├── evaluate.py
│   └── play_interactive.py
│
├── tests/
│   ├── test_rule_oracle.py
│   ├── test_wingspan_engine.py
│   ├── test_gym_env.py
│   └── test_imitation_parser.py
│
└── notebooks/
    ├── 00_rulebook_exploration.ipynb
    ├── 01_rule_oracle_dev.ipynb
    ├── 02_wingspan_engine_debug.ipynb
    └── 03_training_curves.ipynb
```

---

## Sprint 0 — Foundation
**Duración estimada:** 3–4 días  
**Objetivo:** Repositorio funcionando, dependencias instaladas, modelos de datos base.

### Tareas

- [ ] **S0.1** Inicializar repositorio con `pyproject.toml` (uv/poetry)
  ```toml
  # dependencias core
  anthropic>=0.40.0
  chromadb>=1.5.5
  sentence-transformers>=3.0.0
  pydantic>=2.0
  gymnasium>=0.29.0
  stable-baselines3>=2.3.0
  sb3-contrib>=2.3.0          # para MaskablePPO
  torch>=2.2.0
  pymupdf>=1.24.0              # lectura de PDFs
  pandas>=2.0.0
  pytest>=8.0.0
  ```

- [ ] **S0.2** Crear `.env.example` con variables requeridas:
  ```
  ANTHROPIC_API_KEY=
  CHROMA_PERSIST_DIR=./data/chroma_db
  RULEBOOK_PATH=./data/rulebooks/
  CARD_CATALOG_PATH=./data/card_catalogs/
  ```

- [ ] **S0.3** Implementar `src/games/base/game_state.py`
  - `GameState` (Pydantic BaseModel abstracto): `player_id`, `turn`, `phase`, `players`, `shared_board`
  - `PlayerState`: `hand`, `score`, `resources`
  - `Action` dataclass: `action_type: str`, `params: dict`, `player_id: int`
  - `ActionResult` dataclass: `success: bool`, `new_state: GameState`, `events: list[str]`

- [ ] **S0.4** Implementar `src/games/base/engine.py`
  ```python
  class GameEngine(ABC):
      @abstractmethod
      def reset(self) -> GameState: ...
      @abstractmethod
      def step(self, state: GameState, action: Action) -> ActionResult: ...
      @abstractmethod
      def get_legal_actions(self, state: GameState) -> list[Action]: ...
      @abstractmethod
      def is_terminal(self, state: GameState) -> bool: ...
      @abstractmethod
      def get_winner(self, state: GameState) -> int | None: ...
  ```

- [ ] **S0.5** Claude API wrapper con retry + caching en disco:
  ```python
  # src/oracle/claude_client.py
  # Cache: hashlib.sha256(prompt+model) → JSON en ./data/cache/
  # Retry: tenacity con 3 intentos, backoff exponencial
  # Siempre usar claude-sonnet-4-6 por defecto
  ```

- [ ] **S0.6** Tests de smoke: importar todos los módulos sin error.

### Deliverables
- Repo con estructura completa de carpetas
- `GameState`, `Action`, `GameEngine` base implementados y con tests
- Claude client con cache funcionando
- `pytest` en verde (0 tests reales, solo imports)

---

## Sprint 1 — Rule Interpreter
**Duración estimada:** 5–7 días  
**Objetivo:** Pipeline RAG sobre el manual de Wingspan funcionando. Rule Oracle que responde preguntas de reglas con referencias textuales y valida si un movimiento es legal dado un estado.

### Contexto técnico
El chunking de manuales de juegos de mesa no es trivial. Los manuales tienen estructura
mixta: reglas generales, excepciones en sidebar, efectos de cartas individuales, ejemplos.
Estrategia: chunking jerárquico con contexto de sección padre en cada chunk.

### Tareas

- [ ] **S1.1** `src/oracle/ingestion.py` — pipeline de ingesta
  ```python
  # Pasos:
  # 1. pymupdf → texto por página
  # 2. Detectar secciones (regex sobre headings o LLM para estructura)
  # 3. Chunking: 400 tokens con overlap 80, preservar contexto de sección
  # 4. Metadata por chunk: {page, section, game, chunk_type: rule|example|exception}
  # 5. Embed con all-MiniLM-L6-v2
  # 6. Persistir en ChromaDB, collection: "rules_{game_name}"
  ```

- [ ] **S1.2** `src/oracle/retriever.py` — RAG retrieval
  ```python
  class RuleRetriever:
      def query(self, question: str, game: str, k: int = 5) -> list[RuleChunk]
      def query_with_filter(self, question: str, chunk_type: str, k: int = 3) -> list[RuleChunk]
  ```

- [ ] **S1.3** `data/golden_rules/wingspan_rules_qa.json` — golden dataset de reglas
  - Mínimo 50 pares Q&A anotados manualmente
  - Categorías: `basic_turn`, `bird_power`, `end_of_round`, `edge_case`, `exception`
  - Cada caso: `{question, expected_answer, source_page, category}`

- [ ] **S1.4** `src/oracle/rule_oracle.py` — Rule Oracle principal
  ```python
  class RuleOracle:
      def answer_rule_question(self, question: str, game: str) -> RuleAnswer
      # → {answer: str, confidence: float, sources: list[str], verbatim_quotes: list[str]}
      
      def validate_action(self, state: GameState, action: Action, game: str) -> ValidationResult
      # → {is_legal: bool, reason: str, rule_references: list[str]}
      
      def resolve_conflict(self, rules_a: str, rules_b: str) -> str
      # Para reglas que parecen contradecirse
  ```

- [ ] **S1.5** `src/oracle/prompts/rule_validator.txt` — prompt de validación
  ```
  # Estructura del prompt:
  # 1. Rol: "Eres el árbitro oficial de {game}."
  # 2. Instrucción: "Dado el estado del juego y la acción propuesta,
  #    determina si es legal según las reglas. Cita textualmente."
  # 3. Contexto RAG: <rules>{retrieved_chunks}</rules>
  # 4. Estado: <state>{game_state_json}</state>
  # 5. Acción: <action>{action_json}</action>
  # 6. Output format: JSON {is_legal, reason, rule_quote, confidence}
  ```

- [ ] **S1.6** Evaluación del Rule Oracle sobre golden dataset
  ```python
  # scripts/eval_rule_oracle.py
  # Métricas: accuracy (answer relevancy), faithfulness (vs source page)
  # Umbral mínimo para continuar: accuracy >= 0.80
  ```

- [ ] **S1.7** Notebook `01_rule_oracle_dev.ipynb` con exploración cualitativa

### Deliverables
- ChromaDB poblado con reglas de Wingspan chunkeadas
- Rule Oracle con accuracy ≥ 0.80 sobre golden dataset de reglas
- Test suite `tests/test_rule_oracle.py` con 20+ casos

### Riesgo crítico
El chunking de manuales de juegos de mesa es más difícil de lo que parece. Los efectos
de cartas individuales ("cuando juegues esta ave, puedes...") son reglas distribuidas
en cientos de líneas. Puede requerir chunking por carta además de chunking por sección.
Si el accuracy del RAG < 0.75, escalar a chunking híbrido antes de avanzar.

---

## Sprint 2 — Game Engine: Wingspan
**Duración estimada:** 7–10 días  
**Objetivo:** Simulador completo de Wingspan en Python. Este sprint es el más laborioso y el cuello de botella técnico del proyecto.

### Reglas de Wingspan a implementar (alcance MVP)
```
Incluir:
  - 4 habitats (bosque/pradera/humedal) con 5 slots cada uno
  - 4 acciones por turno: Ganar comida, Poner huevo, Obtener carta, Jugar carta
  - Mecánica de poder de aves: {acumular, productivo, flocking, predador, otros}
  - Fin de ronda: objetivos de fin de ronda (4 tarjetas por partida)
  - Puntuación final: aves + huevos + comida en alas + cartas debajo + objetivos + bonos de fin

Excluir del MVP:
  - Bandeja de Audubon (expansión)
  - Cartas de Nekton (expansión europea/oceánica)
  - Modo solitario (Automa)
  - Partidas de 3-5 jugadores (empezar con 2 jugadores)
```

### Tareas

- [ ] **S2.1** `data/card_catalogs/wingspan_birds.csv`
  - Fuente: Wingspan Wiki o extracción manual del manual
  - Campos: `name, habitat, cost_food, nest_type, egg_limit, points,
    power_timing, power_type, power_text`
  - Mínimo 170 aves para el set base

- [ ] **S2.2** `src/games/wingspan/state.py` — WingspanState
  ```python
  class WingspanState(GameState):
      # Tablero personal (por jugador)
      habitats: dict[Habitat, list[BirdSlot | None]]  # 5 slots × 3 habitats
      
      # Recursos personales
      food_supply: dict[FoodType, int]   # semillas, fruta, invertebrado, roedor, pez
      hand: list[BirdCard]
      
      # Bandeja de huevos por hábitat
      eggs_per_habitat: dict[Habitat, int]
      
      # Estado global
      bird_feeder: dict[FoodType, int]   # dados de comida en el comedero
      bird_tray: list[BirdCard]          # 3 cartas disponibles en la bandeja
      draw_deck_count: int
      
      # Ronda actual y turno
      round: int          # 1-4
      turn_in_round: int  # decrements (empieza en 8, 5, 5, 3 por ronda)
      current_player: int
      
      # Objetivos de fin de ronda
      round_end_goals: list[RoundGoal]
  ```

- [ ] **S2.3** `src/games/wingspan/actions.py` — space de acciones
  ```python
  class WingspanActionType(Enum):
      GAIN_FOOD = "gain_food"
      LAY_EGGS = "lay_eggs"
      DRAW_CARDS = "draw_cards"
      PLAY_BIRD = "play_bird"
      # Poderes de aves (se generan dinámicamente)
      ACTIVATE_POWER = "activate_power"
  
  @dataclass
  class WingspanAction(Action):
      action_type: WingspanActionType
      # Para PLAY_BIRD:
      card_index: int | None = None
      target_habitat: Habitat | None = None
      # Para ACTIVATE_POWER:
      bird_slot: tuple[Habitat, int] | None = None
      power_choices: dict | None = None  # decisiones dentro del poder
  ```

- [ ] **S2.4** `src/games/wingspan/engine.py` — WingspanEngine
  - Implementar `step()` para las 4 acciones base
  - Implementar `get_legal_actions()` correctamente (crítico para RL)
  - Implementar poderes de aves de menor complejidad primero (productivos, de acumulación)
  - Los poderes complejos (predadores con interacción) pueden quedar como stub para S2 y completarse iterativamente

- [ ] **S2.5** `src/games/wingspan/rules.py` — LegalMoveValidator
  ```python
  # Validación determinista en código Python (NO via LLM en tiempo de inferencia)
  # El LLM se usa en S1 para entender las reglas; aquí traducimos eso a código
  # La Rule Oracle se consulta solo cuando un edge case no está cubierto
  
  class LegalMoveValidator:
      def get_legal_play_bird_actions(self, state) -> list[WingspanAction]
      def get_legal_gain_food_actions(self, state) -> list[WingspanAction]
      def get_legal_lay_eggs_actions(self, state) -> list[WingspanAction]
      def get_legal_draw_cards_actions(self, state) -> list[WingspanAction]
      def validate_action(self, state, action) -> tuple[bool, str]
  ```

- [ ] **S2.6** `src/games/wingspan/rewards.py` — reward shaping
  ```python
  # Reward design es problema de investigación — decisiones documentadas aquí:
  #
  # Opción A: Solo terminal (+1 victoria, -1 derrota) → más puro, más lento
  # Opción B: Reward denso (puntos por acción) → aprende más rápido, sesgo
  # Opción C: Potential-based shaping → teóricamente correcto
  #
  # Implementar las tres. Hiperparámetro: reward_mode = "terminal" | "dense" | "shaped"
  
  def compute_reward(state_before, action, state_after, done, reward_mode) -> float
  ```

- [ ] **S2.7** Tests exhaustivos del motor de juego
  - `tests/test_wingspan_engine.py`: ≥ 50 casos de test
  - Partida completa aleatoria sin crash (100 juegos simulados)
  - Verificar que puntuación final es consistente
  - Verificar que `get_legal_actions()` nunca devuelve lista vacía en estado no terminal
  - Property-based testing con Hypothesis para invariantes del estado

### Deliverables
- Motor de Wingspan funcionando con las 4 acciones base
- ≥ 50 aves implementadas con sus poderes
- 100 partidas completas simuladas sin crash
- `tests/test_wingspan_engine.py` en verde

### Riesgo crítico
Los poderes de las cartas de Wingspan son texto libre en lenguaje natural.
Implementarlos todos manualmente sería semanas de trabajo. Estrategia:
usar LLM para generar el código Python del poder dado el texto de la carta,
revisarlo manualmente, y solo entonces incluirlo. Esto es parte del
research problem: ¿qué fracción de poderes puede el LLM traducir
correctamente sin intervención humana?

---

## Sprint 3 — Gym Environment
**Duración estimada:** 3–4 días  
**Objetivo:** WingspanEnv como gym.Env válido, con observation space y action masking funcionales.

### Tareas

- [ ] **S3.1** `src/envs/wingspan_env.py` — gym.Env
  ```python
  class WingspanEnv(gym.Env):
      metadata = {"render_modes": ["text", "ansi"]}
      
      def __init__(self, num_players=2, reward_mode="dense", render_mode=None):
          # observation_space: gymnasium.spaces.Dict con subspaces por componente
          # action_space: Discrete(N_MAX_ACTIONS) — varía por estado → necesita masking
          ...
      
      def reset(self, seed=None, options=None) -> tuple[ObsDict, InfoDict]
      def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]
      def action_masks(self) -> np.ndarray  # requerido por MaskablePPO
      def render(self) -> str | None
  ```

- [ ] **S3.2** Diseño del observation space — documentar decisiones
  ```python
  # Estrategia de encoding del estado:
  # Un estado de Wingspan tiene ~500 variables discretas
  # No caben en imagen (no es Atari) → Dict observation space
  #
  # Componentes:
  # - birds_on_board: array (3 habitats × 5 slots × max_bird_features)
  # - food_supply: array de 5 ints (tipos de comida)
  # - eggs_count: array de 3 ints (por habitat)
  # - hand_cards: array (max_hand_size × card_features) — orden ignorado
  # - bird_tray: array (3 × card_features)
  # - round_goals: array (4 × goal_features)
  # - round, turn_in_round, current_player: escalares
  #
  # card_features: one-hot habitat + one-hot nest + food_cost + points + 
  #                power_timing_embedding (LLM → vector pre-computado)
  ```

- [ ] **S3.3** Action masking — crítico para convergencia RL
  ```python
  # En juegos de mesa, el espacio de acciones ilegales >> acciones legales
  # Sin masking, el agente aprende a evitar acciones ilegales en lugar de
  # aprender estrategia. MaskablePPO (sb3-contrib) soporta masking nativo.
  #
  # action_masks() → np.ndarray bool de tamaño N_MAX_ACTIONS
  # Las acciones ilegales tienen probabilidad 0 durante sampling
  
  N_MAX_ACTIONS = 256  # cota superior del espacio de acciones por turno
  ```

- [ ] **S3.4** `gymnasium.utils.env_checker` — validación oficial
  ```python
  from gymnasium.utils.env_checker import check_env
  check_env(WingspanEnv())  # debe pasar sin warnings
  ```

- [ ] **S3.5** Vectorized envs para training paralelo
  ```python
  # make_vec_env con N_ENVS=8 para acelerar recolección de experiencia
  ```

- [ ] **S3.6** Tests `tests/test_gym_env.py`
  - 1000 steps con política aleatoria sin crash
  - Verificar shape de observaciones
  - Verificar que action_masks() es consistente con WingspanEngine.get_legal_actions()

### Deliverables
- `WingspanEnv` válido según `check_env()`
- Action masking funcionando
- 1000 steps con política random sin crash

---

## Sprint 4 — RL Agent
**Duración estimada:** 5–7 días  
**Objetivo:** MaskablePPO entrenando correctamente. Baseline de referencia (random + greedy). Primeras curvas de aprendizaje.

### Tareas

- [ ] **S4.1** `src/agents/encoders.py` — state encoder
  ```python
  # El Dict obs space necesita un feature extractor personalizado
  # Arquitectura: CombinedExtractor que concatena:
  # - MLP para recursos (food, eggs, round info)
  # - Flatten + Linear para birds_on_board
  # - Flatten + Linear para hand y tray
  # - Concatenación → MLP trunk compartido
  
  class WingspanFeaturesExtractor(BaseFeaturesExtractor):
      # Integra con stable-baselines3 policy_kwargs={"features_extractor_class": ...}
  ```

- [ ] **S4.2** `src/agents/baselines.py` — agentes de referencia
  ```python
  class RandomAgent:
      # Selecciona uniformemente de acciones legales
  
  class GreedyAgent:
      # Heurística simple: maximiza puntos inmediatos por acción
      # Play bird con más puntos disponible → si no, lay eggs → si no, gain food
  
  # Estos son el baseline mínimo que PPO debe superar
  ```

- [ ] **S4.3** `src/agents/ppo_agent.py` — configuración MaskablePPO
  ```python
  from sb3_contrib import MaskablePPO
  
  model = MaskablePPO(
      "MultiInputPolicy",
      env=make_vec_env(WingspanEnv, n_envs=8),
      policy_kwargs={"features_extractor_class": WingspanFeaturesExtractor,
                     "net_arch": [256, 256]},
      learning_rate=3e-4,
      n_steps=2048,
      batch_size=64,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      # ... documentar todos los hiperparámetros con justificación
  )
  ```

- [ ] **S4.4** `scripts/train_ppo.py` — training loop con callbacks
  ```python
  # Callbacks:
  # - EvalCallback: evalúa contra random agent cada 50K steps
  # - StopTrainingOnNoModelImprovement
  # - TensorBoard
  # - Checkpoint cada 100K steps
  
  # Métricas a trackear:
  # - win_rate_vs_random: % victorias contra RandomAgent
  # - win_rate_vs_greedy: % victorias contra GreedyAgent
  # - avg_score: puntuación promedio por partida
  # - rule_violation_rate: acciones ilegales intentadas (debería ser 0 con masking)
  # - episode_length: turnos por partida
  ```

- [ ] **S4.5** Experimento baseline
  - Entrenar 1M steps contra RandomAgent self-play
  - Registrar win rate cada 50K steps
  - Resultado esperado: win_rate_vs_random > 70% al final
  - Si no converge → diagnosticar reward shaping o observation space

- [ ] **S4.6** Hiperparámetro sweep inicial (manual, 5 configuraciones)
  - Variables: learning_rate × net_arch × reward_mode
  - Documentar en `experiments/exp_001_ppo_baseline.json`

### Deliverables
- PPO entrenando sin error
- `win_rate_vs_random` > 50% en el primer millón de steps
- Curvas de aprendizaje en TensorBoard
- `experiments/` con primer resultado documentado

---

## Sprint 5 — Imitation Learning
**Duración estimada:** 5–7 días  
**Objetivo:** Parser de logs de BGA/TTS. Behavioral Cloning como pre-entrenamiento. Verificar que BC + PPO supera PPO solo.

### Fuentes de datos de demostración
```
BoardGameArena (BGA):
  - Wingspan tiene cientos de partidas registradas
  - Formato: JSON de eventos por turno
  - Acceso: scraping o API si disponible
  - Requiere: account de BGA (gratuito)

Tabletop Simulator (TTS):
  - Logs en formato propio (.json con history de estados)
  - Comunidad activa en Steam Workshop
  - Más fácil de parsear que BGA
```

### Tareas

- [ ] **S5.1** `src/imitation/bga_parser.py`
  ```python
  class BGALogParser:
      def parse_game_log(self, raw_log: dict) -> list[Transition]
      # Transition: {state: WingspanState, action: WingspanAction, next_state, reward, done}
      
      # Desafío: mapear eventos BGA a WingspanAction del engine propio
      # Requiere un mapping manual de ~50 tipos de evento
  ```

- [ ] **S5.2** `src/imitation/demo_buffer.py`
  ```python
  class DemonstrationBuffer:
      def add_game(self, transitions: list[Transition]) -> None
      def sample(self, batch_size: int) -> tuple[obs, actions, ...]
      def filter_by_winner(self) -> "DemonstrationBuffer"  # solo partidas ganadas
      def __len__(self) -> int
  ```

- [ ] **S5.3** `src/agents/bc_agent.py` — Behavioral Cloning
  ```python
  # BC: supervisado, minimiza cross-entropy entre acción del experto y predicción
  # Pre-entrena el actor de PPO antes del fine-tuning con RL
  
  class BehavioralCloningTrainer:
      def train(self, demo_buffer, model, n_epochs=50) -> TrainingMetrics
      # Métricas: BC accuracy (% acciones correctas del experto reproducidas)
  ```

- [ ] **S5.4** Pipeline completo: BC → PPO fine-tuning
  ```python
  # scripts/train_bc_then_ppo.py
  # 1. BC pre-train (50 épocas sobre demos)
  # 2. Guardar checkpoint
  # 3. Cargar checkpoint en MaskablePPO
  # 4. Fine-tune con PPO (misma config de S4)
  ```

- [ ] **S5.5** Experimento comparativo (research question primaria)
  ```
  Condición A: PPO from scratch (baseline de S4)
  Condición B: BC → PPO
  Condición C: BC solo (sin RL)
  
  Métricas: win_rate_vs_random, win_rate_vs_greedy, avg_score
  Hipótesis: B > A en sample efficiency (misma calidad final con menos steps)
  ```

- [ ] **S5.6** Tests `tests/test_imitation_parser.py`
  - Parser produce Transitions válidas (sin NaN, shapes correctos)
  - BC accuracy > 0.60 sobre heldout de demos (sonido de alarma si < 0.5)

### Deliverables
- Parser de BGA funcionando con ≥ 50 partidas parseadas
- DemonstrationBuffer con las partidas
- BC accuracy > 0.60 sobre heldout
- Experimento A/B/C documentado

---

## Sprint 6 — Evaluación y Ablación
**Duración estimada:** 4–5 días  
**Objetivo:** Framework de evaluación riguroso. Ablation study que justifique cada componente del sistema.

### Tareas

- [ ] **S6.1** `src/eval/metrics.py` — métricas de evaluación
  ```python
  # Métricas de juego:
  # - win_rate(agent_a, agent_b, n_games=500): float
  # - avg_score(agent, n_games=200): (mean, std)
  # - score_distribution(agent): histograma
  
  # Métricas de adherencia a reglas (investigación):
  # - rule_violation_rate: acciones ilegales sobre total de acciones intentadas
  #   (con masking debería ser 0 para PPO; relevante para LLM-only baseline)
  
  # Métricas de eficiencia:
  # - steps_to_target_winrate: sample efficiency
  ```

- [ ] **S6.2** `src/eval/tournament.py` — torneo round-robin
  ```python
  class Tournament:
      def run(self, agents: dict[str, BaseAgent], n_games_per_pair=200) -> EloTable
      # Calcula Elo aproximado con resultados de partida
  ```

- [ ] **S6.3** `src/eval/llm_judge.py` — LLM-as-Judge para calidad de juego
  ```python
  # Dado un transcript de partida:
  # - ¿Las decisiones son estratégicamente coherentes?
  # - ¿Se aprovechan las sinergias de cartas?
  # - ¿Hay errores tácticos evidentes?
  #
  # Esto es para evaluación qualitativa, no para training
  
  class LLMJudge:
      def evaluate_play_quality(self, game_transcript: str) -> PlayQualityReport
  ```

- [ ] **S6.4** Ablation study formal
  ```
  Objetivo: aislar la contribución de cada componente
  
  Variante 1: Baseline PPO (sin RAG, sin BC)
  Variante 2: PPO + RAG (Rule Oracle disponible durante training)
  Variante 3: BC → PPO (sin RAG)
  Variante 4: BC → PPO + RAG (sistema completo)
  
  Protocolo: 3 semillas × 1M steps × 500 partidas de evaluación
  
  Resultado esperado:
    Variante 4 > Variante 3 ≈ Variante 2 > Variante 1
    (BC aporta más que RAG para win_rate, pero RAG reduce rule violations)
  ```

- [ ] **S6.5** Análisis de errores cualitativos
  - Seleccionar 20 partidas perdidas contra greedy
  - Para cada una: ¿en qué turno se tomó la decisión sub-óptima clave?
  - Clasificar: `positional_blunder | resource_mismanagement | power_unused | other`

### Deliverables
- Framework de evaluación completo
- Tabla de Elo entre todos los agentes
- Ablation study con 3 semillas
- Análisis de errores cualitativos en `experiments/ablation_study.md`

---

## Sprint 7 — Generalización y Publicación
**Duración estimada:** variable (investigación abierta)  
**Objetivo:** Demostrar que el sistema se generaliza a un segundo juego sin cambiar la arquitectura. Preparar estructura del paper.

### Tareas

- [ ] **S7.1** Selección del segundo juego
  ```
  Candidatos (ordenados por complejidad de implementación):
  
  1. Terraforming Mars — mayor, más cartas (~200), mecánicas de proyecto más complejas
  2. 7 Wonders (versión 2P Duel) — más compacto, más accesible para validación rápida
  3. Spirit Island — asimétrico, requiere representar poderes únicos por espíritu
  
  Recomendación para S7 MVP: 7 Wonders Duel
  (estado de juego más pequeño, verificación de generalización más rápida)
  ```

- [ ] **S7.2** Port del framework al segundo juego
  - Nuevo `src/games/second_game/` siguiendo el mismo ABC
  - Nuevo rulebook ingestado en ChromaDB en collection separada
  - Reutilizar Rule Oracle sin cambios
  - Reutilizar WingspanEnv pattern con adaptaciones mínimas

- [ ] **S7.3** Entrenamiento del agente en segundo juego
  - Misma pipeline: BC (si hay logs) → PPO
  - Reportar si la muestra eficiencia es comparable a Wingspan

- [ ] **S7.4** Estructura del paper
  ```
  Venue objetivo: AAAI 2026 Workshop on Games and AI / NeurIPS 2025 Games Workshop
  
  Secciones:
  1. Introduction — el problema de grounding de LLMs en juegos
  2. Related Work — boardgame AI (AlphaGo, Hanabi), LLM grounding, RL con reglas
  3. System Architecture — Rule Layer + Env Layer + Agent Layer
  4. Experiments — Wingspan benchmark + ablation study
  5. Generalization — segundo juego
  6. Limitations — poderes no implementados, escala a más jugadores
  7. Conclusion
  
  Contribución principal: primer sistema que combina RAG sobre manuales
  con RL para juegos de mesa modernos no-Atari
  ```

- [ ] **S7.5** Open source release checklist
  - README con instrucciones de instalación y uso
  - Dockerfile para reproducibilidad
  - Script de descarga de rulebooks públicamente disponibles
  - Licencia: MIT (código) + citar paper para los datos

### Deliverables
- Agente funcionando en segundo juego
- Draft del paper con resultados de S6

---

## Decisiones de diseño abiertas (documentar antes de implementar)

Estas son preguntas técnicas sin respuesta única correcta. Cada una debe
decidirse y documentarse antes de escribir el código correspondiente.

```
D1: ¿Cómo representar poderes de cartas para el observation space?
    Opción A: embedding pre-computado del texto (LLM → vector 128d)
    Opción B: features discretas manuales (timing, type, magnitude)
    Opción C: combinación A+B
    → Impacto: generalización vs interpretabilidad

D2: ¿Self-play o fixed adversary para entrenamiento?
    Opción A: self-play (aprende a jugar contra sí mismo)
    Opción B: fixed baselines (random, greedy)
    Opción C: self-play con periodical checkpoints (AlphaGo style)
    → Impacto: calidad del aprendizaje vs complejidad de implementación

D3: ¿Cómo manejar poderes con decisiones multi-paso?
    Ejemplo: "Pone 1 huevo en cada ave adyacente. Si hay ≥3 aves, gana 1 punto."
    Opción A: sub-actions (el poder genera sub-acciones para resolver)
    Opción B: flatten a acciones atómicas (estado aumentado)
    → Impacto: complejidad del action space vs fidelidad del juego

D4: ¿Cuándo consultar la Rule Oracle durante training?
    Opción A: nunca (solo código determinista)
    Opción B: solo para edge cases no cubiertos (fallback)
    Opción C: en cada step (muy costoso)
    → Impacto: costo de API vs cobertura de reglas
```

---

## Métricas de éxito del proyecto

```
MVP (Sprint 0–4 completados):
  ✓ Motor de Wingspan funcionando con ≥50 aves
  ✓ PPO con win_rate_vs_random > 60%
  ✓ Rule Oracle con accuracy > 80% en golden dataset

Research-grade (Sprint 0–6):
  ✓ win_rate_vs_greedy > 60%
  ✓ Ablation study con resultados estadísticamente significativos (p<0.05)
  ✓ LLM-assisted power implementation: ≥70% de poderes generados sin corrección manual
  ✓ Rule violation rate: 0% con action masking (verifica que el engine es correcto)

Publicable (Sprint 0–7):
  ✓ Generalización demostrada en segundo juego con <20% cambios en el framework
  ✓ Reproducible con Dockerfile
  ✓ Paper draft completo con resultados de ablación
```

---

## Estimación de costo de API (Claude Sonnet 4.6)

```
Sprint 1 — Rule Oracle development:
  ~500 calls × ~3K tokens avg = 1.5M tokens
  → ~$4.50 (con cache en system prompt)

Sprints 2–3 — LLM-assisted code gen para poderes de cartas:
  ~200 cartas × ~2K tokens = 400K tokens
  → ~$1.20

Sprint 5 — BC training (no usa API):
  → $0

Sprint 6 — LLM-as-Judge evaluation:
  ~200 transcripts × ~4K tokens = 800K tokens
  → ~$2.40

Total estimado: ~$10–15 USD para el proyecto completo
(el costo principal es tiempo de cómputo para RL training, no API)
```

---

## Comandos de desarrollo frecuentes

```bash
# Setup
git clone <repo>; cd tabletop-oracle
uv sync  # o: pip install -e ".[dev]"
cp .env.example .env  # configurar ANTHROPIC_API_KEY

# Ingesta del manual
python scripts/ingest_rulebook.py --game wingspan --pdf data/rulebooks/wingspan_rulebook.pdf

# Evaluar Rule Oracle
python scripts/eval_rule_oracle.py --game wingspan --golden data/golden_rules/wingspan_rules_qa.json

# Entrenamiento
python scripts/train_ppo.py --game wingspan --total-timesteps 1000000 --n-envs 8 --reward-mode dense

# Evaluación
python scripts/evaluate.py --checkpoint checkpoints/ppo_wingspan_1M.zip --n-games 500

# Tests
pytest tests/ -v
pytest tests/test_wingspan_engine.py -v --tb=short

# Interactive play
python scripts/play_interactive.py --agent ppo --checkpoint checkpoints/ppo_wingspan_best.zip
```

---

*OPB AI Mastery Lab · From pipeline to decision. — Octavio Pérez Bravo*
