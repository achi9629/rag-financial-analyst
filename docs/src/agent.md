# Agent Module (`src/agent.py`)

## Purpose

The agent module implements an **agentic self-correction loop** that upgrades the static cascading fallback (pipeline.py) into an intelligent retry system. When generated code fails execution, Llama-3-70B **diagnoses** the root cause and chooses the optimal recovery action — regenerate, retrieve more context, or patch the bug in-place.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  route_query() [Llama-3-70B]│  → RouterOutput (complexity, steps, retrieval_queries)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  retrieve_context()         │  → Initial ChromaDB retrieval
│  [all-MiniLM-L6-v2]         │
└─────────────────────────────┘
    │
    ▼
┌──────────────── AGENT LOOP (max 3 retries) ──────────────────┐
│                                                              │
│  ┌──────────────────────────┐                                │
│  │ Generate / Fix Code      │  ← Qwen-Coder-32B (temp ↑)     │
│  │ [GPU 2, port 8001]       │    0.0 → 0.2 → 0.4             │
│  └──────────────────────────┘                                │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────┐                                │
│  │ Guardrails Check         │  ❌ → HARD STOP (no retry)    │
│  └──────────────────────────┘                                │
│      │ ✅                                                   │
│      ▼                                                       │
│  ┌──────────────────────────┐                                │
│  │ Sandbox Execution        │  → ✅ STOP_SUCCESS            │
│  └──────────────────────────┘                                │
│      │ ❌ (error)                                           │
│      ▼                                                       │
│  ┌──────────────────────────┐                                │
│  │ _diagnose() [Llama-3-70B]│  → Parse error → choose action │
│  │ [GPU 0-1, port 8000]     │                                │
│  └──────────────────────────┘                                │
│      │                                                       │
│      ├── REGENERATE ──→ loop back (fresh code with feedback) │
│      ├── RETRIEVE_MORE → ChromaDB query → loop back          │
│      └── FIX_CODE ─────→ patch code → loop back              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  AgentResult (code, output, trace, metadata)
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Guardrail = hard stop | Security violations are intentional, never retry |
| Llama diagnoses errors | 70B model reasons better about root cause than heuristics |
| Increasing temperature | 0.0 → 0.2 → 0.4 adds diversity on retries to escape local minima |
| Dynamic retrieval | If error is missing domain knowledge, fetch more docs before retrying |
| Separate from pipeline.py | Agent is opt-in (Agent tab), pipeline stays unchanged |

## Classes

### `Action` (Enum)

Terminal and non-terminal actions the agent can take:

| Action | Type | Meaning |
|---|---|---|
| `REGENERATE` | Non-terminal | Discard code, generate fresh with error feedback |
| `RETRIEVE_MORE` | Non-terminal | Fetch additional ChromaDB docs, then regenerate |
| `FIX_CODE` | Non-terminal | Patch the specific bug (Llama provides fix or Qwen rewrites) |
| `STOP_BLOCKED` | Terminal | Guardrail violation — immediate block |
| `STOP_SUCCESS` | Terminal | Code executed successfully |
| `STOP_MAX_RETRIES` | Terminal | Exhausted all retry attempts |

### `AgentState`

Mutable state tracked across the loop:

- `user_query` — original NL query
- `code` — current generated code (mutated on fix/regenerate)
- `error` — last execution error message
- `attempt` — current attempt number (0-indexed)
- `context` — retrieved domain knowledge (grows with `RETRIEVE_MORE`)
- `action_history` — list of `(Action, diagnosis)` tuples for observability

### `AgentResult`

Immutable result returned to the caller:

- `code`, `explanation`, `exec_output`, `confidence`, `columns_used` — same as pipeline
- `attempts` — total attempts (1 = first-pass success)
- `action_history` — human-readable trace: `["regenerate: column X not found", ...]`
- `final_action` — terminal action string
- `query_type` — from router (simple / threshold / multi_hop)
- `error` — last error if failed, `None` if succeeded
- `total_latency` — end-to-end wall time

### `AgentPipeline`

Main class. Wraps an existing `Pipeline` instance.

#### `__init__(self, pipeline)`

Takes an initialized `Pipeline` and reuses its config and vectorstore.

#### `run(self, user_query, df, max_retries=3, k=3) -> AgentResult`

Main entry point. Executes the agent loop:

1. Route query via Llama
2. Initial ChromaDB retrieval
3. Loop: generate → guardrail → execute → diagnose → act
4. Return `AgentResult` with full trace

#### `_diagnose(self, state) -> Action`

Sends the failed code + error to Llama with `DIAGNOSIS_PROMPT`. Llama returns JSON:

```json
{"action": "regenerate|retrieve_more|fix_code", "diagnosis": "reason"}
```

For `retrieve_more`, also returns `retrieval_query`. For `fix_code`, returns `fix` (corrected code). Falls back to `REGENERATE` if Llama's response can't be parsed.

#### `_dynamic_retrieve(self, state) -> str`

Queries ChromaDB with the retrieval query suggested by Llama's diagnosis. Appends results to `state.context` so the next generation attempt has more knowledge.

#### `_fix_code(self, state) -> str`

Uses Llama's inline fix if available. Otherwise, sends the buggy code + error + diagnosis to Qwen via `FIX_CODE_PROMPT` for a targeted rewrite.

## Temperature Schedule

| Attempt | Temperature | Purpose |
|---|---|---|
| 0 | 0.0 | Deterministic first attempt |
| 1 | 0.2 | Slight variation to escape failure mode |
| 2 | 0.4 | More diversity for harder queries |
| 3+ | 0.4 | Capped at 0.4 |

## Prompts

### `DIAGNOSIS_PROMPT`

Sent to Llama after execution failure. Includes:
- User query
- Generated code
- Execution error (traceback)

Llama must respond with exactly one JSON object choosing `regenerate`, `retrieve_more`, or `fix_code`.

### `FIX_CODE_PROMPT`

Sent to Qwen when `_fix_code()` needs a targeted rewrite. Includes:
- PaySim schema reminder
- Original query
- Buggy code
- Error message
- Llama's diagnosis

## Usage

### Standalone

```python
from pipeline import Pipeline
from agent import AgentPipeline
import pandas as pd

pipe = Pipeline()
agent = AgentPipeline(pipe)

df = pd.read_csv("assets/datasets/PaySim/PS_20174392719_1491204439457_log.csv")

result = agent.run("Detect structuring — multiple txns just below 10 lakh", df)

print(f"Result: {result.final_action} | Attempts: {result.attempts}")
print(f"Code:\n{result.code}")
print(f"Trace: {result.action_history}")
```

### Via Gradio (Agent Tab)

The Agent tab in `app.py` calls `agent.run()` and displays:
- Generated code (syntax highlighted)
- Execution result
- Explanation + metadata
- **Agent Trace** — step-by-step diagnosis and actions taken

## Comparison: Pipeline vs Agent

| Aspect | Pipeline (static cascade) | Agent (self-correction) |
|---|---|---|
| Retry strategy | Fixed: Qwen → Qwen+feedback → Llama → docs | Dynamic: Llama diagnoses → chooses action |
| Context | Fixed at retrieval time | Grows with dynamic retrieval |
| Temperature | Fixed 0.0 | Increasing: 0.0 → 0.2 → 0.4 |
| Code fix | Full regeneration only | Can patch specific bugs |
| Observability | Fallback stage number | Full action trace with diagnosis |
| Max retries | 4 stages | 3 retries (configurable) |

## Dependencies

- `src/pipeline.py` — `Pipeline` class (wrapped, not replaced)
- `src/models.py` — `call_llm()`
- `src/router.py` — `route_query()`
- `src/chain.py` — `retrieve_context()`, `build_prompt()`
- `src/guardrails.py` — `check_code_safety()`
- `src/executor.py` — `execute_code_safe()`
- `src/validation.py` — `CodeOutput`, `parse_code_output_with_fallback()`
