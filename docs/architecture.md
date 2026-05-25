# RAG Financial Analyst — Architecture

## Overview

A multi-model RAG pipeline for financial fraud analysis over 6M+ synthetic PaySim transactions. The system translates natural language queries into executable pandas code using retrieved domain knowledge, with a 4-stage cascading fallback and an optional agentic self-correction loop.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Gradio UI (app.py)                                │
│                        [Pipeline Mode]  [Agent Mode]                           │
└──────────────┬──────────────────────────────────────┬──────────────────────────┘
               │                                      │
               ▼                                      ▼
┌──────────────────────────┐           ┌──────────────────────────────────┐
│   Pipeline (pipeline.py) │           │  AgentPipeline (agent.py)        │
│   4-stage static cascade │           │  Observe→Diagnose→Act loop       │
└──────────────┬───────────┘           └──────────────┬───────────────────┘
               │                                      │
               └──────────────┬───────────────────────┘
                              │ (shared stages)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  1. ROUTE    │─▶│  2. RETRIEVE      │──▶│  3. GENERATE │─▶│ 4. VALIDATE │  │
│  │  router.py   │   │  chain.py         │   │  chain.py    │   │             │  │
│  │  Llama-70B   │   │  ChromaDB         │   │  Qwen-32B    │   │ validation  │  │
│  │              │   │  + MiniLM-L6-v2   │   │  (vLLM)      │   │ guardrails  │  │
│  │  complexity  │   │  similarity search│   │  build_prompt│   │ validator   │  │
│  │  plan steps  │   │  deduplication    │   │  call_llm    │   │ executor    │  │
│  │  retrieval   │   │                   │   │              │   │             │  │
│  │  queries     │   │                   │   │              │   │             │  │
│  └──────────────┘   └───────────────────┘   └──────────────┘   └─────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  5. TRACK            │
                   │  tracking.py         │
                   │  MLflow per-query    │
                   │  + aggregate metrics │
                   └──────────────────────┘
```

---

## Multi-Model Setup

| Model | Hardware | Port | Role |
|-------|----------|------|------|
| **Llama-3-70B-Instruct** | GPU 0–1, tensor-parallel=2 | 8000 | Router, cross-validator, fallback code generator |
| **Qwen2.5-Coder-32B-Instruct** | GPU 2 | 8001 | Primary code generator |
| **all-MiniLM-L6-v2** | CPU | — | Document embeddings (384-dim) |

Both LLMs are served via **vLLM** with OpenAI-compatible APIs. The `models.py` module provides a unified `call_llm(config, prompt, model="qwen"|"llama")` interface backed by the OpenAI Python client.

---

## Pipeline Stages

### Stage 1 — Query Routing (`router.py`)

Llama-3-70B classifies the user query into a complexity tier and produces an execution plan.

**Input:** Natural language query
**Output:** `RouterOutput` (Pydantic model)

| Field | Description |
|-------|-------------|
| `complexity` | `simple` / `threshold` / `multi_hop` |
| `steps` | 1–5 execution steps describing how to answer the query |
| `retrieval_queries` | 1–3 search queries to send to the vector store |

**Classification rules:**
- **simple** — Direct column lookup, basic aggregation, single-condition filter
- **threshold** — Requires a numeric threshold or rule from domain knowledge
- **multi_hop** — Multiple computation steps, combining rules, or derived features

### Stage 2 — Context Retrieval (`chain.py → retrieve_context`)

Searches ChromaDB using the router's retrieval queries. Each query returns the top-k most similar document chunks, deduplicated by content prefix.

- Embedding model: `all-MiniLM-L6-v2` (384-dim, runs on CPU)
- Default k=3 per query; k=5 for `multi_hop` queries
- Results include source metadata (document name, section, similarity score)

**Knowledge base** (`data/docs/`): 6 markdown documents ingested into ChromaDB:

| Document | Content |
|----------|---------|
| `schema.md` | PaySim column definitions, data types, value ranges |
| `fraud_rules.md` | Fraud detection thresholds (balance mismatch, zeroed balance, large transfer rules) |
| `patterns.md` | Known fraud patterns (mule accounts, structuring, rapid movement) |
| `metrics.md` | Financial metrics and KPI definitions |
| `regulatory.md` | Compliance rules and reporting thresholds |
| `domain_glossary.md` | Financial domain terminology |

**Ingestion pipeline** (`ingest.py`):
1. Load markdown files from `data/docs/`
2. Extract section headers as metadata
3. Chunk with `RecursiveCharacterTextSplitter` (800 chars, 120 overlap)
4. Embed via `HuggingFaceEmbeddings` (all-MiniLM-L6-v2)
5. Store in ChromaDB (`data/chromadb/`)

### Stage 3 — Code Generation (`chain.py → build_prompt + generate_code`)

Assembles a structured prompt from the user query, execution plan, and retrieved context, then calls Qwen2.5-Coder-32B to generate pandas code.

**Prompt structure:**
```
## User Query
<query>

## Query Complexity
<simple|threshold|multi_hop>

## Execution Plan
1. <step 1>
2. <step 2>

## Retrieved Domain Knowledge
[Source: schema.md | Section: Columns | Score: 0.234]
<chunk content>
---
[Source: fraud_rules.md | Section: Thresholds | Score: 0.312]
<chunk content>

## Instructions
Generate executable pandas code that answers the user query.
The DataFrame `df` is already loaded with the PaySim dataset.
```

**System prompt** enforces structured JSON output via `STRUCTURED_CODEGEN_SYSTEM_PROMPT` (defined in `validation.py`):

```json
{
    "code": "<executable pandas code>",
    "explanation": "<1-3 sentence explanation>",
    "confidence": 0.95,
    "columns_used": ["isFraud", "amount"]
}
```

**Output parsing** (`validation.py`):
- `parse_code_output()` — strict JSON parse into `CodeOutput` (Pydantic)
- `parse_code_output_with_fallback()` — falls back to raw code extraction if JSON parse fails, wraps in `CodeOutput` with default metadata

### Stage 4 — Validation Stack

Four-layer validation, each progressively deeper:

#### 4a. Pydantic Schema Validation (`validation.py`)

`CodeOutput` model validates:
- `code` is non-empty
- `confidence` is between 0.0 and 1.0
- `columns_used` are checked against valid PaySim column names (warns on unknown)

#### 4b. Guardrails Safety Check (`guardrails.py`)

Static analysis via AST walking + regex fallback. A guardrail failure is a **hard stop** — no retry, to prevent the LLM from learning to circumvent safety checks.

**Blocked patterns:**

| Category | Examples |
|----------|----------|
| Dangerous imports | `os`, `subprocess`, `sys`, `shutil`, `requests`, `socket`, `pickle` |
| Dangerous calls | `exec()`, `eval()`, `compile()`, `open()`, `__import__()` |
| Dangerous attributes | `.system()`, `.popen()`, `.remove()`, `.Popen()` |
| Pandas I/O | `read_csv()`, `to_csv()`, `read_excel()`, `to_parquet()` (df is pre-loaded) |
| Dunder exploits | `__subclasses__`, `__bases__`, `__globals__`, `__builtins__` |
| Shell patterns | `rm -rf`, `curl`, `wget`, `chmod` |
| SQL injection | `DROP TABLE`, `DELETE FROM`, `UNION SELECT` |

**Allowed:** `pandas`, `numpy`, `math`, `re`, `datetime`, `collections`

#### 4c. Llama Cross-Validation (`validator.py`)

Llama-3-70B reviews the generated code against the original query:

1. **Programmatic column check** — regex extraction of column references, validated against PaySim schema (fast, no LLM call)
2. **Semantic validation** — Llama reviews for query alignment, logic errors, data assumptions, and output presence

**Output:** `ValidatorOutput` with `valid: bool` and `issues: list[str]`

#### 4d. Sandbox Execution (`executor.py`)

Executes code in a restricted `exec()` environment:

- **Restricted globals** — only `pd`, `np`, `df`, `print`, and safe builtins (`len`, `range`, `int`, `float`, `str`, `list`, `dict`, `set`, `tuple`, `round`, `abs`, `min`, `max`, `sum`, `sorted`)
- **Timeout enforcement** — `signal.SIGALRM` on Linux (main thread), `threading.Thread` fallback on Windows
- **stdout capture** — via `redirect_stdout` to `StringIO`
- **Output:** `ExecutionResult` with `success`, `output`, and `error`

### Stage 5 — Experiment Tracking (`tracking.py`)

MLflow logging via `QueryTracker`:

**Per-query metrics (logged as MLflow step metrics):**
- `latency_route`, `latency_retrieve`, `latency_generate`, `latency_validate`, `total_latency`
- `fallback_stage`, `model_used`, `query_type`, `confidence`
- `code_safe`, `context_length`, `columns_used`

**Aggregate metrics (logged at run end via `log_summary()`):**
- Code correctness rate, avg latency per stage, fallback frequency
- Fallback stage distribution, model usage distribution

---

## Cascading Fallback (Pipeline Mode)

The pipeline tries up to 4 stages before giving up. Only **validation failures** (wrong columns, logic bugs) trigger the cascade. **Guardrail failures are a hard stop** — no retry.

```
Try 1: Qwen generates → Pydantic + Guardrails + Llama validator → ✅ return
                                                                  ❌ guardrail → HARD STOP
                                                                  ❌ validation ↓

Try 2: Qwen regenerates (with Llama feedback appended to prompt) → Guardrails → ✅ return
                                                                                ❌ ↓

Try 3: Llama generates code directly (model swap) → Guardrails → ✅ return
                                                                 ❌ ↓

Try 4: Graceful degradation → return raw retrieved context + router plan (no code)
```

| Stage | Generator | Validation | Temperature |
|-------|-----------|------------|-------------|
| 1 | Qwen-32B | Full (Pydantic + Guardrails + Llama + Sandbox) | 0.0 |
| 2 | Qwen-32B (with error feedback) | Guardrails only | 0.1 |
| 3 | Llama-70B (model swap) | Guardrails only | 0.0 |
| 4 | None (degradation) | None | — |

---

## Agentic Self-Correction Loop (Agent Mode)

`AgentPipeline` wraps the base pipeline with an observe→diagnose→act loop.

```
                    ┌─────────────┐
                    │  Generate   │◀──────────────────────┐
                    └──────┬──────┘                       │
                           ▼                              │
                    ┌─────────────┐                       │
                    │  Guardrails │──▶ BLOCKED (stop)     │
                    └──────┬──────┘                       │
                           ▼                              │
                    ┌─────────────┐                       │
                    │  Execute    │──▶ SUCCESS (stop)     │
                    │  (sandbox)  │                       │
                    └──────┬──────┘                       │
                           ▼ (error)                     │
                    ┌─────────────┐                       │
                    │  Diagnose   │ (Llama)               │
                    └──────┬──────┘                       │
                     ┌─────┼──────┐                       │
                     ▼     ▼      ▼                       │
               regenerate  retrieve  fix_code             │
                  │       more │      │                    │
                  │        ▼   │      ▼                   │
                  │   ChromaDB │   patch code              │
                  │   append   │      │                    │
                  └────────┴───┴──────┘────────────────────┘
                           (max 3 retries)
```

**Actions (decided by Llama diagnosis):**

| Action | When | What happens |
|--------|------|-------------|
| `REGENERATE` | Overall approach is wrong | Fresh code generation with error feedback, increasing temperature (0.0 → 0.2 → 0.4) |
| `RETRIEVE_MORE` | Missing domain knowledge | Dynamic ChromaDB retrieval using a diagnosed search query, context grows |
| `FIX_CODE` | Minor bug (typo, wrong operator) | Llama provides inline fix, or Qwen patches the code |
| `STOP_BLOCKED` | Guardrail violation | Hard stop, no retry |
| `STOP_SUCCESS` | Code executed successfully | Return result |
| `STOP_MAX_RETRIES` | 3 retries exhausted | Return last state with error |

**State:** `AgentState` dataclass tracks `code`, `error`, `context`, `attempt`, and `action_history` across iterations.

---

## Gradio UI (`app.py`)

Two execution modes selectable via radio button:

| Mode | Backend | Characteristics |
|------|---------|-----------------|
| **Pipeline** | `Pipeline.run_query()` | Deterministic cascade, latency breakdown per stage |
| **Agent** | `AgentPipeline.run()` | Self-correcting loop, action trace visible in UI |

**Output panels:**
- Generated Code
- Execution Output (sandbox stdout)
- Explanation
- Metadata (model, stage, latency breakdown, confidence, columns)
- Action Trace (Agent mode only)
- Session Stats (aggregate metrics table + summary)

---

## Configuration (`configs/config.yaml`)

```yaml
models:
  qwen:
    base_url: "http://localhost:8001/v1"
    model_name: "assets/model/Qwen2.5-Coder-32B-Instruct"
    temperature: 0
    max_tokens: 2048
    role: "code_generator"
  llama:
    base_url: "http://localhost:8000/v1"
    model_name: "assets/model/Meta-Llama-3-70B-Instruct"
    temperature: 0
    max_tokens: 1024
    role: "router_validator"

ingestion:
  docs_dir: "data/docs"
  chroma_dir: "data/chromadb"
  collection_name: "rag_financial_analyst"
  chunk_size: 800
  chunk_overlap: 120
  embedding_model: "assets/model/all-MiniLM-L6-v2"
  embedding_device: "cpu"
```

---

## Module Map

```
src/
├── models.py       → LLM interface: load_config(), call_llm()
├── ingest.py       → Document ingestion: load → chunk → embed → ChromaDB
├── router.py       → Query routing: classify complexity, plan steps, retrieval queries
├── chain.py        → RAG chain: retrieve_context(), build_prompt(), generate_code()
├── validation.py   → Pydantic schema: CodeOutput, parse_code_output_with_fallback()
├── guardrails.py   → Code safety: AST analysis + regex for blocked patterns
├── validator.py    → Cross-validation: programmatic column check + Llama semantic review
├── executor.py     → Sandbox: restricted exec() with timeout + stdout capture
├── pipeline.py     → Orchestrator: 4-stage cascading fallback, PipelineResult
├── agent.py        → Agentic loop: diagnose→act cycle with AgentState/AgentResult
├── tracking.py     → MLflow: per-query metrics + aggregate summary
└── app.py          → Gradio UI: pipeline/agent modes, session stats

data/
├── docs/           → 6 domain knowledge documents (markdown)
├── chromadb/       → Persisted vector store (chroma.sqlite3)
└── eval/           → Expected outputs for evaluation (JSONL)

configs/
└── config.yaml     → Model endpoints, ingestion params, temperatures

assets/
├── datasets/PaySim → 6M+ transaction CSV
└── model/          → Local model weights (Llama-70B, Qwen-32B, MiniLM-L6-v2)
```

---

## GPU Allocation

| GPU | Model | VRAM |
|-----|-------|------|
| GPU 0 | Llama-3-70B (shard 1/2) | ~40 GB |
| GPU 1 | Llama-3-70B (shard 2/2) | ~40 GB |
| GPU 2 | Qwen2.5-Coder-32B | ~65 GB (FP16) |
| GPU 3 | Free | — |

---

## Data Flow Summary

```
User Query
    │
    ▼
[Router] Llama classifies → {complexity, steps, retrieval_queries}
    │
    ▼
[Retriever] ChromaDB similarity search → deduplicated context chunks
    │
    ▼
[Prompt Builder] query + plan + context → structured prompt
    │
    ▼
[Generator] Qwen generates → JSON {code, explanation, confidence, columns_used}
    │
    ▼
[Parser] Pydantic parse (with raw-code fallback) → CodeOutput
    │
    ▼
[Guardrails] AST + regex safety check → pass / HARD STOP
    │
    ▼
[Validator] Column check + Llama semantic review → pass / cascade
    │
    ▼
[Executor] Restricted sandbox exec() → stdout capture
    │
    ▼
[Tracker] MLflow logs latency, model, stage, confidence
    │
    ▼
[UI] Code + Output + Explanation + Metadata → Gradio
```
