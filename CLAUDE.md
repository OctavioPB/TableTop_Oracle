# CLAUDE.md — TabletopOracle
> Instrucciones de sesión para Claude Code.
> Lee este archivo completo antes de escribir una sola línea de código.

---

## Quién soy y cómo trabajo

**Octavio Pérez Bravo** — Data Engineer en Teradata (Informatica PowerCenter,
Apache Kafka), estudiante de maestría en Big Data y Ciencias de Datos,
y Data & AI Strategy Architect en transición hacia liderazgo técnico.

Tengo experiencia sólida en Python, pipelines de datos, arquitecturas de IA,
y uso diario del ecosistema Claude. No soy nuevo en esto. No necesito que me
expliques qué es un DataFrame, un embeddings model, o un gym.Env.

**Cómo espero que trabajes conmigo:**

- Responde en **español** siempre, salvo que el contexto sea explícitamente
  técnico (nombres de clases, funciones, errores de Python) — esos en inglés.
- **No me adules.** "¡Excelente pregunta!" o "¡Gran idea!" son ruido.
  Si algo está mal o es subóptimo, dímelo con justificación.
- **Arquitectura antes que código.** Si hay una decisión de diseño no trivial,
  descríbela y justifícala antes de implementarla. No me des código sin saber
  por qué tomaste esa estructura.
- **Profundidad sobre brevedad.** Prefiero la explicación completa a la
  versión simplificada. Si algo tiene un matiz importante, inclúyelo.
- **Una variable a la vez.** Si cambias el código y algo falla, cambia
  exactamente una cosa antes de decir "prueba esto". No hagas 5 cambios
  simultáneos y esperes que yo identifique cuál funcionó.
- Si no sabes algo, dilo. No inventes APIs, parámetros ni comportamientos
  de bibliotecas. Prefiero "no lo sé, verifica la documentación de sb3"
  a una respuesta confiada y equivocada.

---

## Qué es este proyecto

**TabletopOracle** — agente que aprende a jugar juegos de mesa complejos
(Wingspan, Terraforming Mars) sin reglas hardcodeadas. El agente lee el
manual en lenguaje natural y combina:

- **LLM** (Claude) como intérprete de reglas vía RAG
- **RL** (PPO con action masking) para aprender estrategia

**Esto es investigación publicable.** El target es un workshop de AAAI 2026
o NeurIPS 2025 en Games & AI. Cada decisión de diseño debe poder justificarse
académicamente, no solo pragmáticamente.

El documento de referencia completo del proyecto está en `PLAN.md`.
Si necesitas saber el alcance de un sprint, la estructura del repo,
o una decisión de diseño pendiente, léelo ahí.

---

## Arquitectura — tres capas

```
┌─────────────────────────────────────────────────────────────────┐
│  RULE LAYER                                                     │
│  PDF Rulebook → Chunker → ChromaDB ← Rule Oracle (Claude)      │
│  src/oracle/                                                    │
├─────────────────────────────────────────────────────────────────┤
│  ENVIRONMENT LAYER                                              │
│  GameState (Pydantic) + LegalMoveValidator + gym.Env           │
│  src/games/ + src/envs/                                         │
├─────────────────────────────────────────────────────────────────┤
│  AGENT LAYER                                                    │
│  BC pre-train → MaskablePPO (sb3-contrib)                       │
│  src/agents/ + src/imitation/                                   │
└─────────────────────────────────────────────────────────────────┘
```

**La decisión de diseño más importante del proyecto:**
El LLM **no juega en tiempo real**. Durante el desarrollo (S1–S2) se usa
para generar y validar el código Python de reglas y poderes de cartas.
Durante el entrenamiento de RL, el validador de movimientos legales corre
en Python puro. El LLM solo se consulta para edge cases no cubiertos en
el código — esto es un fallback, no el camino principal.

Razón: 1M steps de RL × 1 llamada API por step = ~$3,000 USD.
Con el diseño actual, el costo total de API del proyecto es ~$15 USD.

---

## Stack técnico — versiones exactas

```toml
# pyproject.toml — dependencias core
anthropic = ">=0.40.0"
chromadb = ">=1.5.5"
sentence-transformers = ">=3.0.0"   # modelo: all-MiniLM-L6-v2
pydantic = ">=2.0"
gymnasium = ">=0.29.0"
stable-baselines3 = ">=2.3.0"
sb3-contrib = ">=2.3.0"             # MaskablePPO
torch = ">=2.2.0"
pymupdf = ">=1.24.0"                # ingesta de PDFs
pandas = ">=2.0.0"
tenacity = ">=8.0.0"                # retry logic para API calls
pytest = ">=8.0.0"
hypothesis = ">=6.0.0"             # property-based testing
```

**Modelo Claude por defecto:** `claude-sonnet-4-6`
Usar Haiku solo para compresión de historial o tareas de clasificación simple.
Usar Opus solo si Sonnet falla consistentemente en una tarea específica.

**Embeddings:** `all-MiniLM-L6-v2` (no cambiar sin discutir — es el modelo
que ya está en el stack ChromaDB del proyecto OPB AI Mastery Lab).

**ChromaDB:** `PersistentClient` en `./data/chroma_db/`. Un collection por
juego: `rules_wingspan`, `rules_terraforming_mars`, etc.

**RL:** `sb3_contrib.MaskablePPO` únicamente. No usar PPO estándar de SB3
— sin action masking, el agente aprende a evitar acciones ilegales en lugar
de aprender estrategia. Esta es una falla de diseño, no solo suboptimalidad.

---

## Convenciones de código

### Python
- **Type hints obligatorios** en todas las funciones públicas.
  Sin `def foo(x, y)` — siempre `def foo(x: int, y: str) -> bool`.
- **Pydantic v2** para todos los modelos de datos (GameState, Action,
  RuleChunk, etc.). No usar dataclasses para modelos de dominio.
- **No usar `Any`** en type hints sin un comentario que justifique por qué.
- **Docstrings en inglés**, una línea de resumen + Args/Returns si la función
  es pública o no obvia.
- **Importaciones absolutas** siempre: `from src.oracle.retriever import ...`
  No importaciones relativas (`from .retriever import ...`).

### Naming
```python
# Clases: PascalCase
class WingspanEngine(GameEngine): ...

# Funciones y variables: snake_case
def get_legal_actions(state: WingspanState) -> list[Action]: ...
legal_actions = engine.get_legal_actions(current_state)

# Constantes: UPPER_SNAKE_CASE
N_MAX_ACTIONS = 256
DEFAULT_MODEL = "claude-sonnet-4-6"

# Archivos: snake_case
# wingspan_engine.py, rule_oracle.py, demo_buffer.py

# Collections de ChromaDB: prefijo rules_ + nombre del juego
COLLECTION_WINGSPAN = "rules_wingspan"
```

### Estructura de archivos de módulo
```python
# Orden estándar dentro de cada archivo:
# 1. Imports stdlib
# 2. Imports third-party
# 3. Imports internos (src.*)
# 4. Constantes del módulo
# 5. Clases (ABCs primero, implementaciones después)
# 6. Funciones standalone
# 7. if __name__ == "__main__": (solo en scripts, no en módulos)
```

### Manejo de errores
```python
# No usar bare except. Siempre capturar excepciones específicas.
# Malas:
try:
    result = engine.step(state, action)
except:
    pass

# Buenas:
try:
    result = engine.step(state, action)
except InvalidActionError as e:
    logger.warning(f"Invalid action attempted: {e}. State: {state.turn}")
    return ActionResult(success=False, reason=str(e))
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Usar niveles correctamente:
# DEBUG: estado interno, valores intermedios
# INFO: flujo normal del programa (partida iniciada, step completado)
# WARNING: algo inesperado pero recuperable
# ERROR: fallo que impide completar la operación
# No usar print() en código de producción
```

---

## Testing — filosofía y estándares

**Regla de oro:** antes de declarar que algo funciona, escribe el test que
lo demuestra. No me muestres código sin tests correspondientes para los
caminos críticos.

```python
# Estructura de tests
tests/
├── test_rule_oracle.py       # S1 — accuracy en golden dataset
├── test_wingspan_engine.py   # S2 — motor de juego, invariantes
├── test_gym_env.py           # S3 — gym.Env válido, masking
└── test_imitation_parser.py  # S5 — parser de logs BGA/TTS

# Convención de nombres
def test_<unit>_<behavior>_<condition>():
    # test_wingspan_engine_legal_actions_empty_board
    # test_rule_oracle_returns_source_reference
    # test_gym_env_action_mask_consistent_with_engine
```

**Tests obligatorios por componente:**

- `WingspanEngine`: partida completa aleatoria sin crash (100 juegos)
- `get_legal_actions()`: nunca devuelve lista vacía en estado no terminal
- `WingspanEnv`: `check_env(WingspanEnv())` sin warnings
- `RuleOracle`: accuracy ≥ 0.80 sobre `data/golden_rules/wingspan_rules_qa.json`
- `action_masks()`: 100% consistencia con `engine.get_legal_actions()`

**Property-based testing con Hypothesis** para invariantes del GameState:
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=3))
def test_wingspan_score_nonnegative(round_num):
    # El score nunca puede ser negativo en Wingspan
    ...
```

---

## Decisiones ya tomadas — no reabrir

Estas decisiones están en el PLAN.md con justificación. No las sugieran
como alternativas ni pidan confirmación para cada línea de código:

| Decisión | Elección | Por qué |
|----------|----------|---------|
| LLM en inferencia | No (solo en desarrollo) | Costo: $3K vs $15 |
| Action masking | MaskablePPO obligatorio | Sin masking, el RL aprende lo incorrecto |
| Modelo base | claude-sonnet-4-6 | Balance calidad/costo para este stack |
| Embeddings | all-MiniLM-L6-v2 | Consistencia con el stack OPB existente |
| Primer juego | Wingspan | Documentación pública, comunidad activa |
| Estado de juego | Pydantic v2 | Validación automática, serialización JSON |
| Framework RL | stable-baselines3 + sb3-contrib | Maduro, bien documentado, MaskablePPO |

---

## Decisiones abiertas — discutir antes de implementar

Estas requieren análisis antes de codificar. Si llegas a un punto donde
alguna de estas decisiones bloquea el progreso, para y discútela.

```
D1 · Representación de poderes de cartas en el observation space
     Afecta: generalización del agente, interpretabilidad
     Opciones: embedding LLM pre-computado | features discretas | híbrido
     Estado: PENDIENTE (relevante para Sprint 3)

D2 · Self-play vs fixed adversary durante entrenamiento
     Afecta: calidad del aprendizaje, complejidad de implementación
     Opciones: self-play | baselines fijos | self-play con checkpoints (AlphaGo)
     Estado: PENDIENTE (relevante para Sprint 4)

D3 · Poderes de cartas multi-paso
     Ejemplo: "Pon 1 huevo en cada ave adyacente. Si hay ≥3, gana 1 punto."
     Opciones: sub-acciones | flatten a atómicas | estado aumentado
     Estado: PENDIENTE (relevante para Sprint 2)

D4 · Condiciones de consulta al Rule Oracle durante training
     Opciones: nunca | fallback para edge cases | siempre (descartado por costo)
     Estado: PENDIENTE (relevante para Sprint 2)
```

---

## Gestión del costo de API

Este proyecto tiene un presupuesto estimado de ~$15 USD total en API.
Cada llamada a Claude cuenta. Reglas para no desperdiciar tokens:

```python
# 1. Cache en disco para todas las llamadas de desarrollo
# Implementado en: src/oracle/claude_client.py
# Cache key: sha256(model + messages + temperature)
# Esto significa que iterar sobre el mismo prompt no cuesta nada

# 2. Prompt caching de Anthropic para el system prompt del Rule Oracle
# El system prompt del Rule Oracle es estático (~800 tokens)
# → Marcar con cache_control: ephemeral
# → Ahorro: 90% en tokens de input del system prompt

# 3. Nunca llamar a la API en el training loop de RL
# Si encuentras código que hace esto, es un bug de diseño, no de implementación

# 4. Usar Haiku para tareas de compresión/clasificación simple
# Ejemplo: comprimir historial conversacional, clasificar tipo de power

# 5. Batch las llamadas cuando sea posible
# Para evaluar el golden dataset, no llames una por una — usa asyncio.gather()
```

---

## Contexto de investigación

Este proyecto es investigación original con intención de publicación.
Esto afecta cómo debe escribirse el código:

**Reproducibilidad es obligatoria:**
- Fijar seeds en todo: `torch.manual_seed(42)`, `np.random.seed(42)`, 
  `random.seed(42)`, y `seed` en `env.reset(seed=42)`
- Guardar todos los hiperparámetros usados en cada experimento en JSON
- No hardcodear hiperparámetros en el código de training — usar config files

**Registrar todo:**
- Cada experimento tiene su propio directorio en `experiments/exp_NNN_descripcion/`
- Estructura mínima: `config.json`, `results.json`, `training_curves.png`
- El directorio de un experimento nunca se sobreescribe — se crea uno nuevo

**Ablación antes que optimización:**
- No optimices hiperparámetros antes de tener los experimentos de ablación
  de S6. La ablación es la contribución científica, no el mejor win_rate.

**Claim vs evidencia:**
- Si vas a decir "el agente aprendió X", necesitas los números que lo
  demuestren. No hacer afirmaciones cualitativas sobre el comportamiento
  del agente sin métricas de soporte.

---

## Lo que NO debes hacer

```
❌ No simplificar la lógica de Wingspan "por ahora" sin documentarlo
   como deuda técnica explícita con una issue o TODO

❌ No usar print() para debugging — usar logging correctamente

❌ No cambiar más de una variable entre experimentos

❌ No declarar una función "lista" sin tests correspondientes

❌ No inventar parámetros de sb3 o gymnasium — verifica la documentación

❌ No usar PPO estándar sin action masking — es un error de diseño

❌ No llamar a la API de Claude dentro del training loop de RL

❌ No hardcodear rutas absolutas — usar pathlib.Path y variables de entorno

❌ No mezclar español e inglés dentro de un mismo docstring o comentario
   (inglés para código y docstrings, español para respuestas a mí)

❌ No usar Any en type hints sin justificación en comentario adyacente

❌ No usar bare except — siempre capturar excepciones específicas
```

---

## Variables de entorno requeridas

```bash
# .env (nunca commitear al repo)
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_DIR=./data/chroma_db
RULEBOOK_DIR=./data/rulebooks
CARD_CATALOG_DIR=./data/card_catalogs
GAME_LOGS_DIR=./data/game_logs
EXPERIMENTS_DIR=./experiments
CHECKPOINTS_DIR=./checkpoints
LOG_LEVEL=INFO                      # DEBUG durante desarrollo activo
CACHE_DIR=./data/cache              # cache de respuestas Claude
```

---

## Comandos frecuentes

```bash
# Verificar setup
python -c "from src.oracle.claude_client import ClaudeClient; print('OK')"

# Ingestar manual
python scripts/ingest_rulebook.py --game wingspan

# Evaluar Rule Oracle (S1)
python scripts/eval_rule_oracle.py --game wingspan

# Verificar gym env (S3)
python -c "
from gymnasium.utils.env_checker import check_env
from src.envs.wingspan_env import WingspanEnv
check_env(WingspanEnv())
print('Env check passed')
"

# Training (S4)
python scripts/train_ppo.py \
  --game wingspan \
  --total-timesteps 1_000_000 \
  --n-envs 8 \
  --reward-mode dense \
  --seed 42

# Evaluación
python scripts/evaluate.py \
  --checkpoint checkpoints/ppo_wingspan_best.zip \
  --n-games 500

# Tests
pytest tests/ -v --tb=short
pytest tests/test_wingspan_engine.py -v -k "legal_actions"

# Play interactivo
python scripts/play_interactive.py --checkpoint checkpoints/ppo_wingspan_best.zip
```

---

## Estado actual del proyecto

```
Sprint 0 — Foundation          [x] Completado (2026-04-17)
Sprint 1 — Rule Interpreter    [x] Completado (2026-04-18)
Sprint 2 — Wingspan Engine     [x] Completado (2026-04-17)
Sprint 3 — Gym Environment     [x] Completado (2026-04-18)
Sprint 4 — RL Agent            [x] Completado (2026-04-18)
Sprint 5 — Imitation Learning  [x] Completado (2026-04-18)
Sprint 6 — Evaluación          [x] Completado (2026-04-19)
Sprint 7 — Generalización      [x] Completado (2026-04-19)
```

Actualiza esta sección al final de cada sesión de trabajo.

---

## Cómo empezar una sesión de trabajo

1. Lee `PLAN.md` → identifica el sprint activo y las tareas pendientes.
2. Lee este `CLAUDE.md` → ya lo estás haciendo.
3. Corre `pytest tests/ -v` → verifica que el estado del repo es limpio.
4. Identifica la tarea específica del sprint actual.
5. Si la tarea involucra una decisión de diseño abierta (D1–D4), discútela
   conmigo antes de escribir código.
6. Implementa, escribe tests, verifica que pasan.
7. Actualiza el estado del sprint en este archivo.

---

## Lecciones aprendidas (Sprints 0–5)

Bugs reales encontrados durante el desarrollo — registrados para no repetirlos.

### S2 — Motor de juego

**`TURNS_PER_ROUND` en el import incorrecto**
`engine.py` lo importaba desde `cards.py` pero vive en `state.py`.
Regla: antes de importar una constante, verificar en qué módulo está definida,
no asumir por el nombre del archivo.

**Aliasing en `WingspanPlayerBoard.from_dict()`**
`food_supply=d.get("food_supply", {})` devolvía referencia al dict interno del
estado serializado. Mutar el board mutaba el estado original.
Regla: cualquier `from_dict()` o constructor que recibe colecciones mutables
debe hacer copia defensiva: `dict(...)`, `list(...)`.

### S3 — Entorno gym

**`check_env()` falla si los valores de observación salen de [0, 1]**
Varios campos (food_supply, eggs) podían superar 1.0 sin normalización explícita.
Regla: todos los valores del observation space van por `np.clip(v, 0.0, 1.0)`
antes de devolverlos. No confiar en que los rangos del juego sean acotados.

**Invariante `player_id == 0` requiere manejo explícito en cada salida de `step()`**
Al terminar una ronda, `_end_round()` restablece `player_id`. En los turnos
normales (no fin de ronda), hay que forzar `model_copy(update={"player_id": 0})`
manualmente. Sin esto, `action_masks()` calcularía legal actions para P1.
Regla: en single-agent mode, documentar el invariante y verificarlo con test.

### S4 — Agente RL

**`DummyVecEnv` llama `env.action_masks()` vía `env_method`**
No se necesita el wrapper `ActionMasker` de sb3-contrib cuando el env tiene
`action_masks()` como método público. `MaskablePPO` lo detecta automáticamente.
Regla: no agregar wrappers de compatibilidad que ya están cubiertos por el framework.

**`build_maskable_ppo` debe separar `features_dim` y `net_arch` del dict `**hp`**
`MaskablePPO.__init__` no acepta `features_dim` ni `net_arch` como kwargs directos;
van dentro de `policy_kwargs`. Si se pasan en `**hp`, el constructor falla con
`unexpected keyword argument`.
Regla: leer la firma del constructor antes de usar `**kwargs` para pasar config.

### S5 — Imitation learning

**`Transition` debe almacenar observaciones gym (dict de numpy), no estados del motor**
El diseño inicial tenía `state: WingspanState` en `Transition`. BC necesita obs
tensorizables, no objetos Pydantic. Cambiar a `obs: dict[str, np.ndarray]` eliminó
la dependencia entre el buffer y el motor de juego.
Regla: los componentes de RL (buffer, trainer) operan sobre el espacio de obs gym,
no sobre los objetos de dominio del motor.

**`BehavioralCloningTrainer.evaluate()` debe llamar `policy.to(device)` explícitamente**
`MaskablePPO` detecta CUDA y pone la policy en GPU por default. Si `evaluate()`
se llama sin mover previamente la policy al device del trainer, los tensores de
obs (CPU) y los pesos del modelo (CUDA) chocan en runtime.
Regla: toda función que hace inferencia debe llamar `policy.to(self._device)` antes
del primer forward pass, no solo en `train()`.

**`BGALogParser` requiere temporalmente cambiar `player_id` en el estado para P1**
`get_legal_actions()` filtra por `state.player_id`. Si el parser está reproduciendo
el turno de P1 pero el estado tiene `player_id=0`, las acciones legales devueltas
son incorrectas. Hay que hacer `model_copy(update={"player_id": 1})` antes de
consultar legal actions para P1, y restaurar después.
Regla: cualquier código que llama `get_legal_actions()` fuera de `step()` debe
asegurarse que `state.player_id` corresponde al jugador correcto.

### Patrones de diseño que funcionaron

- **Smoke test como primer test**: un test que solo hace `import` y una inicialización
  básica atrapa errores de instalación y de firma de constructores antes de escribir
  los tests reales. Vale la pena siempre.

- **`model_copy(update={...})` de Pydantic v2 para transiciones de estado**:
  la inmutabilidad del estado hace que los bugs de aliasing sean raros. El único
  lugar donde persisten es en `from_dict()` con colecciones anidadas.

- **Separar `_action_to_idx` y `_idx_to_action` como métodos del env, no del agente**:
  la conversión entre el espacio de acciones del motor y los índices enteros del gym
  pertenece al env, no al agente. El agente solo ve índices; el motor solo ve acciones.

- **`SyntheticDemoGenerator` como alternativa a datos externos**:
  para BC, usar el `GreedyAgent` como experto sintético permite desarrollar y testear
  el pipeline completo sin necesidad de logs de BGA. Los datos reales mejoran la
  calidad del prior, pero no son necesarios para verificar que el código funciona.

---

*OPB AI Mastery Lab · From pipeline to decision. — Octavio Pérez Bravo*
