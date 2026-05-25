# RAG Financial Analyst

Production-grade RAG pipeline that translates natural language queries into executable pandas code for financial fraud analysis over 6M+ synthetic PaySim transactions. Uses multi-model inference (Llama-3-70B + Qwen2.5-Coder-32B) with retrieval-augmented generation, a 4-stage cascading fallback, and an agentic self-correction loop.

---

## TL;DR

- **Multi-model RAG** — Llama-3-70B routes queries, Qwen2.5-Coder-32B generates code, ChromaDB retrieves domain knowledge
- **4-stage fallback cascade** — Route → Retrieve → Generate → Validate with automatic escalation
- **Agentic self-correction** — Observe→Diagnose→Act loop with up to 3 retry attempts
- **Security guardrails** — AST-based code safety checks, blocked module/call detection, sandboxed execution
- **6.3M transactions** — PaySim synthetic financial dataset with fraud labels
- **Full observability** — MLflow per-query metrics, latency breakdown, confidence scoring

---

## Motivation

Financial fraud detection queries require domain-specific knowledge (balance mismatch rules, structuring thresholds, mule account patterns) that LLMs don't have out of the box. A direct "text-to-code" approach fails because:

1. **Domain vocabulary** — "structuring", "mule account", "balance mismatch" require specific detection logic
2. **Threshold knowledge** — Rule-based detection needs exact numeric thresholds from regulatory docs
3. **Multi-step reasoning** — Composite risk scores require chaining multiple detection rules

RAG solves this by retrieving fraud rules, schema definitions, and detection patterns from a vector store before code generation — giving the LLM the context it needs to produce correct, executable code.

---

## Pipeline

```
User Query → Route (Llama-70B) → Retrieve (ChromaDB) → Generate (Qwen-32B) → Validate → Execute
```

| Stage | Component | Model/Tool | Purpose |
|-------|-----------|------------|---------|
| 1. Route | `router.py` | Llama-3-70B | Classify complexity, plan steps, generate retrieval queries |
| 2. Retrieve | `chain.py` | ChromaDB + MiniLM-L6-v2 | Fetch relevant fraud rules, schema, patterns |
| 3. Generate | `chain.py` | Qwen2.5-Coder-32B | Build prompt with context, generate pandas code |
| 4. Validate | `validator.py` + `guardrails.py` + `executor.py` | Llama-70B + AST | Cross-validate, safety check, sandbox execute |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Gradio UI (app.py)                        │
│                  [Pipeline Mode]  [Agent Mode]                  │
└────────────┬────────────────────────────────┬───────────────────┘
             │                                │
             ▼                                ▼
┌────────────────────────┐     ┌──────────────────────────────────┐
│  Pipeline (pipeline.py)│     │  AgentPipeline (agent.py)        │
│  4-stage static cascade│     │  Observe → Diagnose → Act loop   │
└────────────┬───────────┘     └──────────────┬───────────────────┘
             │                                │
             └────────────┬───────────────────┘
                          ▼
    ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐
    │  ROUTE   │→ │ RETRIEVE  │→ │ GENERATE │→ │   VALIDATE    │
    │ Llama-70B│  │ ChromaDB  │  │ Qwen-32B │  │ guardrails +  │
    │          │  │ MiniLM-L6 │  │  (vLLM)  │  │ executor      │
    └──────────┘  └───────────┘  └──────────┘  └───────────────┘
                          │
                          ▼
               ┌───────────────────┐
               │  MLflow Tracking  │
               │  per-query metrics│
               └───────────────────┘
```

---

## Multi-Model Setup

| Model | Hardware | Port | Role |
|-------|----------|------|------|
| **Llama-3-70B-Instruct** | GPU 0–1, tensor-parallel=2 | 8000 | Router, cross-validator, fallback generator |
| **Qwen2.5-Coder-32B-Instruct** | GPU 2 | 8001 | Primary code generator |
| **all-MiniLM-L6-v2** | CPU | — | Document embeddings (384-dim) |

Both LLMs served via **vLLM** with OpenAI-compatible APIs.

---

## Key Features

### Agentic Self-Correction (`agent.py`)
- **Observe** — Execute generated code, capture output/errors
- **Diagnose** — LLM analyzes failure (syntax error, wrong columns, logic bug)
- **Act** — Choose action: `regenerate`, `retrieve_more`, `fix_code`
- Up to 3 attempts before graceful fallback

### Security Guardrails (`guardrails.py`)
- AST-based static analysis (no regex hacks)
- Blocked modules: `os`, `subprocess`, `sys`, `shutil`, `requests`, etc.
- Blocked calls: `exec`, `eval`, `open`, `__import__`
- Sandboxed execution with restricted globals

### Query Router (`router.py`)
- **simple** — Direct column lookup, basic aggregation
- **threshold** — Numeric rules from domain knowledge
- **multi_hop** — Multiple computation steps, derived features

### Domain Knowledge (`data/docs/`)
- `fraud_rules.md` — Balance mismatch, structuring, mule account detection
- `schema.md` — PaySim column definitions and data types
- `patterns.md` — Common fraud patterns and detection logic
- `metrics.md` — KPI definitions for fraud analysis
- `regulatory.md` — Compliance thresholds and reporting rules

---

## Project Structure

```bash
src/
├── app.py           # Gradio UI + MLflow logging
├── pipeline.py      # 4-stage cascade pipeline
├── agent.py         # Agentic self-correction loop
├── router.py        # Query complexity classification (Llama-70B)
├── chain.py         # Retrieval + prompt building + code generation
├── ingest.py        # Document ingestion → ChromaDB
├── models.py        # Unified LLM interface (vLLM/OpenAI client)
├── guardrails.py    # AST-based code safety checks
├── validator.py     # LLM cross-validation (Llama-70B)
├── validation.py    # Structured output parsing (Pydantic)
├── executor.py      # Sandboxed code execution
└── tracking.py      # MLflow experiment tracker


configs/
└── config.yaml      # Model endpoints, ingestion params

data/
├── docs/            # Domain knowledge documents (6 files)
└── eval/            # Expected outputs for evaluation
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start LLM servers (vLLM)
bash scripts/run_qwen.sh   # GPU 2, port 8001
bash scripts/run_llama.sh  # GPU 0-1, port 8000

# 3. Ingest documents (one-time)
cd src && python ingest.py

# 4. Run the app
python src/app.py
# Open http://localhost:7860

# 5. View MLflow metrics (optional)
python -m mlflow ui --port 5000
# Open http://localhost:5000
```

---

## Run Commands

```bash
# Health checks
bash scripts/run_health_qwen.sh
bash scripts/run_health_llama.sh
```

---

## Observability (MLflow)

Each query logs a run with:
- **Params**: query text, mode (pipeline/agent), model used, query type
- **Metrics**: latency breakdown (route, retrieve, generate, validate, exec), confidence, exec_success, code_safe

```bash
python -m mlflow ui --port 5000
```

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | 4× NVIDIA A100 80GB |
| LLM Serving | vLLM (tensor-parallel) |
| Embeddings | CPU (all-MiniLM-L6-v2) |
| Dataset | 6.3M rows, ~1.5GB |
| Python | 3.10 |

---

## Tech Stack

LangChain · ChromaDB · sentence-transformers · vLLM · Gradio · MLflow · pandas · Pydantic

---

## License

MIT License — see [LICENSE](LICENSE) for details.
