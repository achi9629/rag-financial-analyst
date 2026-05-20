<!-- markdownlint-disable MD036 -->

# RAG Financial Analyst — 5-Day Build Plan

## Status Legend

| Status      | Emoji |
|-------------|-------|
| Done        | ✅    |
| Not done    | ⬜    |
| In progress | 🔄    |
| Blocked     | 🚫    |

## Plan Summary

- **Day 1**: Model serving (vLLM × 2), LiteLLM gateway, basic connectivity
- **Day 2**: Knowledge docs, ChromaDB ingestion, embedding pipeline
- **Day 3**: LangChain RAG chain, query routing, multi-hop decomposition
- **Day 4**: Validation stack (Pydantic + Guardrails + Llama cross-validation + sandbox), cascading fallback
- **Day 5**: W&B tracking, Gradio UI, test 15+ queries, README, push to GitHub

**Total: 5 days × 5–6h = 25–30 hours**

**Dataset**: PaySim — 6.3M synthetic financial transactions (Kaggle)
**Code Generator**: vLLM serving Qwen2.5-Coder-32B-Instruct (GPU 2, port 8001)
**Planner / Router / Validator**: vLLM serving Llama-3-70B-Instruct (GPU 0–1, tensor-parallel=2, port 8000)
**Embeddings**: all-MiniLM-L6-v2 (GPU 3 / CPU)

---

## Tech Stack (10 Tools)

| # | Tool | Role |
|---|---|---|
| 1 | vLLM | Serve Qwen2.5-Coder-32B + Llama-3-70B |
| 2 | LangChain | RAG chain, retrievers, document loaders |
| 3 | ChromaDB | Vector store for domain knowledge docs |
| 4 | LiteLLM | Unified model gateway (swap models with config) |
| 5 | Pydantic | Structured output validation |
| 6 | Guardrails AI | LLM output safety checks |
| 7 | W&B | Experiment tracking (latency, quality, model comparison) |
| 8 | Gradio | UI |
| 9 | sentence-transformers | Embeddings (all-MiniLM-L6-v2) |
| 10 | pandas / numpy | Data processing layer |

---

## PaySim Dataset Schema

| Column | Type | Description |
|---|---|---|
| `step` | int | 1 step = 1 hour of simulation |
| `type` | str | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | float | Transaction amount |
| `nameOrig` | str | Sender ID |
| `oldbalanceOrg` | float | Sender balance before |
| `newbalanceOrig` | float | Sender balance after |
| `nameDest` | str | Receiver ID |
| `oldbalanceDest` | float | Receiver balance before |
| `newbalanceDest` | float | Receiver balance after |
| `isFraud` | int | Ground truth fraud label |
| `isFlaggedFraud` | int | Rule-based fraud flag |

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│  Llama-3-70B (Router)    │  ← Classify: simple | threshold | multi_hop
│  GPU 0–1, port 8000      │     Output: plan + retrieval queries
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  ChromaDB Retrieval      │  ← Retrieve: schema, fraud rules, regulatory docs
│  all-MiniLM-L6-v2        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Qwen-Coder-32B (Gen)    │  ← Generate: pandas code from plan + docs
│  GPU 2, port 8001        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Validation Stack        │
│  1. Pydantic (structure) │
│  2. Guardrails (safety)  │
│  3. ast.parse() (syntax) │
│  4. Llama (semantic)     │
│  5. Sandbox exec (test)  │
└──────────────────────────┘
    │
    ▼
  Result
```

### Cascading Fallback

```
Try 1: Qwen generates → validate → ✅ return
Try 2: Qwen regenerates with Llama feedback → validate → ✅ return
Try 3: Llama generates code directly → syntax + sandbox → ✅ return
Try 4: Return raw retrieved docs + plan + error message (graceful degradation)
```

---

## Pre-requisites (Do At Home Before Day 1)

- ✅ Download PaySim dataset from Kaggle (`kaggle datasets download -d ealaxi/paysim1`)
- ⬜ Download Llama-3-70B-Instruct weights (`huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct`)
- ✅ Verify Qwen2.5-Coder-32B-Instruct weights available locally
- ✅ Install dependencies:
  ```
  pip install vllm langchain langchain-community chromadb openai pydantic guardrails-ai/custom sentence-transformers wandb gradio
  ```
- ✅ **Exit**: all weights + dependencies on local disk, no downloads needed at office

---

## Day 1: Model Serving + Gateway (5–6h)

### Task 1.1 — vLLM Servers

- ⬜ Start Llama-3-70B-Instruct on GPU 0–1:
  ```bash
  vllm serve meta-llama/Meta-Llama-3-70B-Instruct --port 8000 --tensor-parallel-size 2 --max-model-len 4096
  ```
- ✅ Start Qwen2.5-Coder-32B-Instruct on GPU 2:
  ```bash
  CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8001 --max-model-len 4096
  ```
- ✅ Verify both endpoints respond to test prompts via `curl`
- ✅ **Exit**: both models serving on separate ports

### Task 1.2 — LiteLLM Gateway

- ✅ Implement model config in `configs/config.yaml` (endpoints, model names, temperature)
- ✅ Implement LiteLLM wrapper in `src/models.py`
- ⬜ Test switching between Llama and Qwen with one config change
- ✅ `pip freeze > requirements.txt` (pin all dependency versions)
- ✅ **Exit**: `call_llm(model="llama", prompt=...)` and `call_llm(model="qwen", prompt=...)` both working

### Task 1.3 — PaySim Data Exploration

- ⬜ Load PaySim CSV, verify schema, print column dtypes and sample rows
- ⬜ Note edge cases: null values, class imbalance (fraud is ~0.1%)
- ⬜ Save cleaned version to `data/transactions.csv`
- ⬜ **Exit**: data loaded, schema understood, ready to write knowledge docs

---

## Day 2: Knowledge Base + ChromaDB (5–6h)

### Task 2.1 — Write RAG Knowledge Documents

- ⬜ `docs/schema.md` — column names, types, value ranges, meaning of `step` (1 step = 1 hour)
- ⬜ `docs/fraud_rules.md` — velocity thresholds, amount deviation, balance mismatch detection
- ⬜ `docs/regulatory.md` — RBI ₹10L reporting threshold, structuring rules, KYC/AML definitions
- ⬜ `docs/patterns.md` — smurfing, layering, round-tripping, mule account definitions
- ⬜ `docs/metrics.md` — precision, recall, false positive rate, cost of missed fraud
- ⬜ `docs/domain_glossary.md` — CASH_IN vs CASH_OUT vs TRANSFER, isFlaggedFraud meaning
- ⬜ Write gold-standard expected outputs for 15 test queries in `data/eval/expected_outputs.jsonl`
- ⬜ **Exit**: 6 markdown files with real domain knowledge + gold eval set ready

### Task 2.2 — ChromaDB Ingestion Pipeline

- ⬜ Implement document loader + text splitter in `src/ingest.py`
- ⬜ Chunk documents (200 tokens, 30 token overlap — MiniLM max sequence length is 256)
- ⬜ Generate embeddings via sentence-transformers (all-MiniLM-L6-v2)
- ⬜ Store chunks + metadata (source file, section) in ChromaDB collection
- ⬜ Test retrieval: query "how to detect fraud" → verify relevant chunks returned
- ⬜ **Exit**: ChromaDB populated, retrieval returning relevant chunks for test queries

---

## Day 3: RAG Chain + Query Routing (5–6h)

### Task 3.1 — Query Router (Llama-3-70B)

- ⬜ Implement query classifier in `src/router.py`
- ⬜ Llama classifies query into: `simple` | `threshold` | `multi_hop`
- ⬜ For `multi_hop`: Llama outputs step-by-step plan as structured JSON
- ⬜ Pydantic model for router output: `{"complexity": str, "steps": [str], "retrieval_queries": [str]}`
- ⬜ Test on 5+ queries of each type
- ⬜ **Exit**: router correctly classifies and decomposes queries

### Task 3.2 — LangChain RAG Chain (Qwen-Coder-32B)

- ⬜ Implement core chain in `src/chain.py`
- ⬜ Simple path: query → ChromaDB retrieval → Qwen generates pandas code
- ⬜ Threshold path: query → retrieval (get thresholds/rules) → Qwen generates code
- ⬜ Multi-hop path: for each step in plan → retrieve → Qwen generates → combine
- ⬜ Prompt template: inject retrieved docs + schema + query → expect executable pandas code
- ⬜ **Exit**: end-to-end working: NL query → routing → retrieval → code generation

### Task 3.3 — Pydantic Output Schema

- ⬜ Define output models in `src/validation.py`
- ⬜ `CodeOutput`: code (str), explanation (str), confidence (float), columns_used (list[str])
- ⬜ Force Qwen to output structured JSON matching schema
- ⬜ Parse and validate every LLM response
- ⬜ **Exit**: all outputs are Pydantic-validated

### Task 3.4 — Pipeline Orchestrator

- ⬜ Wire router + retriever + generator in `src/pipeline.py`
- ⬜ Implement `run_query(user_query: str) -> CodeOutput`
- ⬜ Test 5 queries end-to-end (1 simple, 2 threshold, 2 multi-hop)
- ⬜ **Exit**: single entry point works for all query types

---

## Day 4: Validation + Fallback (5–6h)

### Task 4.1 — Guardrails AI Safety Layer

- ⬜ Implement safety checks in `src/guardrails.py`
- ⬜ Block: `os.system`, `subprocess`, `open()`, `requests`, file deletion, network calls
- ⬜ Block: SQL injection patterns, shell commands in generated code
- ⬜ Allow only: pandas, numpy, standard aggregation operations
- ⬜ **Exit**: dangerous code patterns rejected before execution

### Task 4.2 — Llama Cross-Validation

- ⬜ Implement validator in `src/validator.py`
- ⬜ After Qwen generates code → Llama reviews: "Does this code answer the query? Any bugs?"
- ⬜ Llama outputs: `{"valid": bool, "issues": [str]}`
- ⬜ **Exit**: Llama catches intentional bugs in test cases

### Task 4.3 — Sandbox Execution

- ⬜ Implement safe executor in `src/executor.py`
- ⬜ `exec()` generated code with restricted globals: only `pd`, `np`, `df`
- ⬜ Run on `df.head(100)` first as sanity check
- ⬜ Catch exceptions, timeout after 10s
- ⬜ Note in README: "For production, use RestrictedPython or nsjail"
- ⬜ **Exit**: generated code executes safely, results returned

### Task 4.4 — Cascading Fallback

- ⬜ Implement fallback logic in `src/pipeline.py`
- ⬜ Try 1: Qwen generates → validate
- ⬜ Try 2: Qwen regenerates with Llama's feedback → validate
- ⬜ Try 3: Llama generates code directly → syntax check + sandbox
- ⬜ Try 4: Return raw retrieved docs + plan + error message
- ⬜ Log which fallback stage was used (for W&B)
- ⬜ **Exit**: graceful degradation tested, no query crashes the system

---

## Day 5: Tracking + UI + Ship (5–6h)

### Task 5.1 — W&B Experiment Tracking

- ⬜ Implement tracking in `src/tracking.py`
- ⬜ Log per query: latency (routing / retrieval / generation / validation), model used, fallback stage, query type
- ⬜ Log aggregate: code correctness rate, avg latency per stage, fallback frequency
- ⬜ Run Llama vs Qwen comparison on 15+ queries, log results
- ⬜ **Exit**: W&B dashboard showing model comparison and per-stage metrics

### Task 5.2 — Gradio UI

- ⬜ Implement UI in `src/app.py`
- ⬜ Input: text box for NL query
- ⬜ Output: generated code (syntax highlighted), execution result (table/chart), explanation, confidence score
- ⬜ Show metadata: query type, retrieval sources, fallback stage, latency breakdown
- ⬜ **Exit**: demo-able UI running locally

### Task 5.3 — Test 15+ Demo Queries

- ⬜ Simple: "Show all fraudulent transactions"
- ⬜ Simple: "How many TRANSFER type transactions?"
- ⬜ Simple: "Average transaction amount"
- ⬜ Threshold: "Flag transactions above ₹10 lakh"
- ⬜ Threshold: "Find accounts with balance mismatch"
- ⬜ Threshold: "Large cash-outs that zeroed sender balance"
- ⬜ Derived: "Detect structuring — multiple txns just below ₹10L"
- ⬜ Derived: "Accounts with >5 transactions per hour"
- ⬜ Derived: "Sender-receiver pairs with round-trip transfers"
- ⬜ Multi-hop: "Identify mule accounts"
- ⬜ Multi-hop: "Compare fraud rate across transaction types"
- ⬜ Multi-hop: "Top 10 most suspicious accounts by risk score"
- ⬜ Multi-hop: "Hourly fraud pattern — which hours have highest fraud rate?"
- ⬜ Multi-hop: "Detect layering through 3+ accounts"
- ⬜ Multi-hop: "Build risk score: velocity + amount deviation + balance mismatch"
- ⬜ **Exit**: 80%+ queries produce correct, executable code

### Task 5.4 — README + Push to GitHub

- ⬜ Write README: architecture diagram, setup instructions, demo queries, W&B screenshots
- ⬜ Add `requirements.txt`
- ⬜ Add `data/download.sh` for PaySim
- ⬜ Clean up code, remove debug prints
- ⬜ Push to GitHub
- ⬜ **Exit**: repo is interview-ready

---

## Repo Structure

```
rag-financial-analyst/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml              # model endpoints, temperature, top-k
├── data/
│   └── download.sh              # kaggle download script
├── docs/
│   ├── schema.md
│   ├── fraud_rules.md
│   ├── regulatory.md
│   ├── patterns.md
│   ├── metrics.md
│   └── domain_glossary.md
├── src/
│   ├── models.py                # LiteLLM wrapper
│   ├── ingest.py                # doc loader → ChromaDB
│   ├── router.py                # Llama query classifier
│   ├── chain.py                 # LangChain RAG chain (Qwen)
│   ├── pipeline.py              # orchestrator + fallback
│   ├── validation.py            # Pydantic schemas
│   ├── guardrails.py            # safety checks
│   ├── validator.py             # Llama cross-validation
│   ├── executor.py              # sandboxed exec()
│   ├── tracking.py              # W&B logging
│   └── app.py                   # Gradio UI
└── notebooks/
    └── evaluation.ipynb         # W&B results, model comparison
```

---

## GPU Allocation

| GPU   | Model                         | VRAM    | Port |
|-------|-------------------------------|---------|------|
| 0–1   | Llama-3-70B-Instruct (TP=2)  | ~140 GB | 8000 |
| 2     | Qwen2.5-Coder-32B-Instruct   | ~65 GB  | 8001 |
| 3/CPU | all-MiniLM-L6-v2             | <1 GB   | —    |

---

## Key Metrics to Have Ready

| Metric | Target |
|---|---|
| Code correctness (executes + correct answer) | 80%+ on 15 test queries |
| Fallback rate | <20% of queries need retry |
| Routing accuracy | 90%+ queries classified correctly |
| End-to-end latency (simple query) | <5s |
| End-to-end latency (multi-hop) | <15s |
| Llama vs Qwen code quality | Measured in W&B |

---

## Resume Line

> Architected a multi-model RAG pipeline for financial data analysis — Llama-3-70B as query planner/router/validator (tensor-parallel, 2× A100) and Qwen2.5-Coder-32B as code generator — with complexity-based routing, cascading fallback, Pydantic output validation, Guardrails AI safety checks, and per-stage experiment tracking via W&B over 6M synthetic transactions.

---

## Interview Talking Points

| Question | Answer |
|---|---|
| "Why two models?" | 70B reasons better (planning, validation), 32B generates better code — each model plays to its strength |
| "Why not one model for everything?" | Separation of concerns — router/validator shouldn't need code specialization, code gen shouldn't waste capacity on planning |
| "How do you handle failures?" | 4-stage cascading fallback — retry with feedback, swap to Llama as generator, graceful degradation to raw docs |
| "Why ChromaDB?" | Persistence, metadata filtering by doc type, simpler API for prototyping. Would use FAISS/Pinecone at scale |
| "Why LiteLLM?" | One-line model swap — if tomorrow we add GPT-4o, zero code change |
| "Production considerations?" | All local (no data leaves the box), Guardrails blocks dangerous code, sandbox execution, W&B for monitoring |
| "Why PaySim?" | Derived features (velocity, structuring, mule detection) require domain knowledge — can't just filter columns. That's what makes it RAG |

---

## Extensions (Post Day 5)

### Extension 1 — Health Check + Circuit Breaker

- ⬜ Add `/health` endpoint polling before each LLM call (`GET http://localhost:800x/health`)
- ⬜ Implement circuit breaker: if vLLM returns 5xx or times out 3× consecutively, mark model as "down"
- ⬜ When Llama is down: skip validation step, rely on Guardrails + ast.parse + sandbox only
- ⬜ When Qwen is down: route directly to Llama for code gen (fallback Try 3 becomes Try 1)
- ⬜ Auto-retry with exponential backoff (1s, 2s, 4s) before tripping breaker
- ⬜ Log circuit breaker state changes to W&B
- ⬜ **Exit**: pipeline never hangs or crashes from OOM / vLLM restart
