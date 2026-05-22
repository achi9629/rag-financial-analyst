# Chain Module (`src/chain.py`)

## Purpose

The chain module implements the **RAG code generation pipeline** — it connects the query router, ChromaDB retrieval, and Qwen-Coder-32B to produce executable pandas code from natural language queries.

## Pipeline Flow

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
│  retrieve_context()         │  → ChromaDB similarity search (deduplicated chunks)
│  [all-MiniLM-L6-v2]         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  build_prompt()             │  → Assemble: query + plan + retrieved docs
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  call_llm(model="qwen")     │  → Qwen-Coder-32B generates pandas code
│  [GPU 2, port 8001]         │
└─────────────────────────────┘
    │
    ▼
  Generated Code (str)
```

## Functions

### `retrieve_context(vectorstore, queries, k=3) -> str`

Runs each retrieval query against ChromaDB, deduplicates results by content prefix, and returns a formatted string with source metadata and similarity scores.

- **Multi-hop queries**: called with `k+2` to get more context per step.
- **Simple/threshold**: called with default `k=3`.

### `build_prompt(user_query, context, plan) -> str`

Assembles the final prompt sent to Qwen. Structure:

```
## User Query
<original query>

## Query Complexity
<simple | threshold | multi_hop>

## Execution Plan
1. <step 1>
2. <step 2>
...

## Retrieved Domain Knowledge
<deduplicated chunks with source metadata>

## Instructions
Generate executable pandas code...
```

### `generate_code(config, user_query, vectorstore, plan=None, k=3) -> str`

Main entry point. Orchestrates the full pipeline:

1. Routes query (or uses pre-computed `plan`)
2. Retrieves context from ChromaDB
3. Builds prompt
4. Calls Qwen for code generation
5. Strips markdown fences if present

Returns raw Python code string.

## System Prompt Design

The `CODEGEN_SYSTEM_PROMPT` constrains Qwen to:

- Output **only** executable Python/pandas code
- Assume `df` is already loaded (no file I/O)
- Use only `pandas` and `numpy`
- Use exact PaySim column names
- Print final results via `print()`
- Use thresholds from retrieved domain knowledge, not hallucinated values

## Complexity-Based Behavior

| Complexity | Retrieval `k` | Behavior |
|---|---|---|
| `simple` | 3 | 1 step, minimal context needed |
| `threshold` | 3 | Needs domain rules/thresholds from retrieved docs |
| `multi_hop` | 5 | More context per retrieval query, multi-step plan |

## Usage

```python
from models import load_config
from ingest import load_vectorstore, get_embeddings
from chain import generate_code

config = load_config("configs/config.yaml")
ingestion_cfg = config["ingestion"]
embeddings = get_embeddings(ingestion_cfg["embedding_model"], ingestion_cfg["embedding_device"])
vectorstore = load_vectorstore(ingestion_cfg["chroma_dir"],
                               ingestion_cfg["collection_name"],
                               embeddings)

code = generate_code(config, "Identify mule accounts", vectorstore)
print(code)
```

## Dependencies

- `src/models.py` — `load_config()`, `call_llm()`
- `src/ingest.py` — `load_vectorstore()`, `get_embeddings()`
- `src/router.py` — `route_query()`, `RouterOutput`
- ChromaDB collection must be pre-populated (run `ingest.py` first)
