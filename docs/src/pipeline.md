# Pipeline Module (`src/pipeline.py`)

## Purpose

The pipeline module is the **central orchestrator** of the RAG Financial Analyst. It wires together routing, retrieval, code generation, validation, and guardrails into a single `run_query()` call with a 4-stage cascading fallback. Guardrail violations are a hard stop — only validation failures trigger retries.

## Cascading Fallback

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Step 1: Route              │  Llama-3-70B classifies query
│  [GPU 0-1, port 8000]       │  → simple | threshold | multi_hop
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 2: Retrieve           │  ChromaDB similarity search
│  [all-MiniLM-L6-v2]         │  k=3 (simple) or k=5 (multi_hop)
└─────────────────────────────┘
    │
    ▼
┌───────────── FALLBACK CASCADE ──────────────┐
│                                              │
│  Try 1: Qwen → Guardrails + Llama Validator  │
│    ├── ✅ Passed → return (stage 1)          │
│    ├── ❌ Guardrail → HARD STOP (stage 4)    │
│    └── ❌ Validation → continue              │
│                                              │
│  Try 2: Qwen + feedback → Guardrails only    │
│    ├── ✅ Passed → return (stage 2)          │
│    └── ❌ Failed → continue                  │
│                                              │
│  Try 3: Llama generates → Guardrails only    │
│    ├── ✅ Passed → return (stage 3)          │
│    └── ❌ Failed → continue                  │
│                                              │
│  Try 4: Return raw docs + plan (stage 4)     │
│    └── Graceful degradation                  │
│                                              │
└──────────────────────────────────────────────┘
```

## Key Design Decision: Guardrail Hard Stop

```
Guardrail failure (os, subprocess, open, etc.)
    → IMMEDIATE BLOCK — no retry
    → model_used = "blocked"
    → Prevents LLM from circumventing safety on retry

Validation failure (wrong column, logic bug)
    → Cascade to Try 2 with Llama's feedback
```

**Why?** If the LLM generates `os.system('rm -rf /')` on Try 1, retrying with feedback might produce a subtler exploit (e.g., `df.to_csv('/tmp/stolen.csv')`). Guardrail violations are treated as intentional and blocked permanently.

## Classes

### `PipelineResult`

Extended result container with execution metadata:

| Field | Type | Description |
|---|---|---|
| `code_output` | `CodeOutput` | Validated code, explanation, confidence, columns |
| `fallback_stage` | `int` | Which try succeeded (1–4) |
| `context` | `str` | Retrieved domain knowledge |
| `plan_steps` | `list` | Router plan steps |
| `model_used` | `str` | `"qwen"`, `"llama"`, `"blocked"`, or `"none"` |
| `query_type` | `str` | `"simple"`, `"threshold"`, or `"multi_hop"` |
| `latency_route` | `float` | Routing time (seconds) |
| `latency_retrieve` | `float` | Retrieval time (seconds) |
| `latency_generate` | `float` | Total generation time across all tries |
| `latency_validate` | `float` | Total validation time across all tries |
| `total_latency` | `float` | End-to-end wall time |

### `Pipeline`

Main orchestrator class.

#### `__init__(self, config_path=None)`

1. Loads YAML config from `configs/config.yaml` (or custom path)
2. Connects to ChromaDB vectorstore via `load_vectorstore()`
3. Initializes `tracker = None` (MLflow disabled by default)

#### `enable_tracking(project, run_name) -> None`

Creates a `QueryTracker` for MLflow logging. Must be called explicitly before `run_query()` to enable tracking.

#### `_validate_code(code, user_query) -> tuple[bool, list, str]`

Two-layer validation:

1. **Guardrails** (fast, deterministic) — AST/regex pattern matching via `check_code_safety()`
2. **Llama cross-validation** (semantic) — Llama reviews code against query via `validate_code()`

Returns 3-tuple: `(is_valid, issues, fail_type)` where `fail_type` is:
- `"guardrail"` — blocked dangerous pattern (hard stop)
- `"validation"` — Llama found a logic/correctness issue (cascade continues)
- `"passed"` — all checks passed

#### `_track(...)` 

Helper to log per-query metrics to MLflow if tracking is enabled. Called at each return point with current latencies and metadata.

#### `run_query(user_query, k=3) -> PipelineResult`

Main entry point. Executes the full cascade:

| Try | Model | Validation | On Fail |
|---|---|---|---|
| 1 | Qwen (temp=0) | Guardrails + Llama | Guardrail → block; Validation → Try 2 |
| 2 | Qwen (temp=0.1) + feedback | Guardrails only | → Try 3 |
| 3 | Llama (temp=0) | Guardrails only | → Try 4 |
| 4 | None | None | Return raw docs + plan |

## Latency Tracking

Latencies are tracked per-stage and accumulate across retries:

```
t_route     = time for route_query()           [measured once]
t_retrieve  = time for retrieve_context()      [measured once]
t_generate += time for each call_llm()          [accumulates across tries]
t_validate += time for each validation step     [accumulates across tries]
total       = wall clock from start to return
```

## Retrieval Strategy

| Query Complexity | Retrieval `k` | Rationale |
|---|---|---|
| `simple` | 3 | Minimal context needed |
| `threshold` | 3 | Need rules/thresholds from docs |
| `multi_hop` | 5 (`k+2`) | More context per step for multi-step plan |

## Usage

```python
from pipeline import Pipeline

pipe = Pipeline()

# Without tracking
result = pipe.run_query("Show all fraudulent transactions")
print(result.code_output.code)
print(f"Stage: {result.fallback_stage}, Model: {result.model_used}")

# With MLflow tracking
pipe.enable_tracking(project="rag-financial-analyst", run_name="eval-run-1")
result = pipe.run_query("Identify mule accounts")
pipe.tracker.log_summary()
pipe.tracker.finish()
```

## Dependencies

- `src/router.py` — `route_query()`
- `src/chain.py` — `retrieve_context()`, `build_prompt()`
- `src/models.py` — `load_config()`, `call_llm()`
- `src/guardrails.py` — `check_code_safety()`
- `src/validator.py` — `validate_code()`
- `src/validation.py` — `CodeOutput`, `parse_code_output_with_fallback()`
- `src/tracking.py` — `QueryTracker`
- `src/ingest.py` — `load_vectorstore()`
