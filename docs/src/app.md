# App Module (`src/app.py`)

## Purpose

The app module implements the **Gradio web UI** for the RAG Financial Analyst. It exposes three tabs — **Pipeline** (static cascade), **Agent** (self-correcting loop), and **Stats** (session metrics dashboard) — providing a demo-ready interface over 6.3M PaySim transactions.

## UI Layout

```
┌──────────────────────────────────────────────────────┐
│               RAG Financial Analyst                  │
├──────────┬──────────┬────────────────────────────────┤
│ Pipeline │  Agent   │  Stats                         │
├──────────┴──────────┴────────────────────────────────┤
│                                                      │
│  [Query Input]                    [Analyze / Run]    │
│                                                      │
│  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │ Generated Code  │  │ Execution Result          │  │
│  │ (syntax-colored)│  │ (DataFrame / error)       │  │
│  ├─────────────────┤  ├───────────────────────────┤  │
│  │ Explanation     │  │ Metadata                  │  │
│  │                 │  │ (model, latency, stage)   │  │
│  └─────────────────┘  └───────────────────────────┘  │
│                                                      │
│  [Agent Trace] (Agent tab only)                      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Tabs

### Tab 1: Pipeline

Uses the static cascading fallback from `pipeline.py`.

**Inputs:**
- Text box for natural language query

**Outputs:**
- **Generated Code** — syntax-highlighted Python/pandas
- **Execution Result** — sandbox output (DataFrame string or error)
- **Explanation** — LLM's explanation of the generated code
- **Metadata** — fallback stage, model used, query type, confidence, columns used, per-stage latency breakdown

**Flow:**
```
user_query → pipe.run_query() → execute_code_safe(code, df) → display
```

### Tab 2: Agent

Uses the agentic self-correction loop from `agent.py`.

**Inputs:**
- Text box for natural language query

**Outputs:**
- **Generated Code** — final code after all correction attempts
- **Execution Result** — sandbox output or error
- **Explanation** — LLM explanation
- **Metadata** — final action, attempt count, query type, confidence, total latency
- **Agent Trace** — step-by-step log of each diagnosis and action taken

**Flow:**
```
user_query → agent.run(query, df) → display code + output + trace
```

### Tab 3: Stats

Session-level metrics aggregated across both Pipeline and Agent queries.

**Outputs:**
- **Summary** — text block with aggregate metrics
- **Query History** — DataFrame table with per-query details

## Functions

### `run_query(user_query: str) -> tuple[str, str, str, str]`

Runs a query through the static pipeline.

1. Calls `pipe.run_query(user_query)` to get `PipelineResult`
2. Executes generated code in sandbox via `execute_code_safe(code, df)`
3. Formats metadata string with latency breakdown
4. Appends to `session_history` for Stats tab
5. Returns `(code, exec_output, explanation, metadata)`

### `run_agent_query(user_query: str) -> tuple[str, str, str, str, str]`

Runs a query through the agentic self-correction loop.

1. Calls `agent.run(user_query, df)` to get `AgentResult`
2. Formats action trace as numbered list
3. Formats metadata string
4. Appends to `session_history` for Stats tab
5. Returns `(code, exec_output, explanation, metadata, trace)`

### `get_stats() -> tuple[pd.DataFrame, str]`

Computes aggregate session metrics from `session_history`:

| Metric | Description |
|---|---|
| First-Pass Rate | % of queries that succeeded on stage 1 / attempt 1 |
| Execution Success Rate | % of queries with successful sandbox execution |
| Code Safety Rate | % of queries not blocked by guardrails |
| Avg Confidence | Mean LLM confidence across all queries |
| Avg Total Latency | Mean end-to-end time per query |

Also computes distributions: modes (pipeline/agent), fallback stages, models used, query types, and per-stage latency breakdown for pipeline queries.

## Session History Schema

Each query appends a dict to `session_history`:

| Field | Type | Source |
|---|---|---|
| `Query` | str | User query (truncated to 60 chars) |
| `Mode` | str | `"pipeline"` or `"agent"` |
| `Stage` | int | Fallback stage (pipeline) or attempt count (agent) |
| `Model` | str | Model that produced final code |
| `Type` | str | Query complexity from router |
| `Confidence` | float | LLM confidence score |
| `Route (s)` | float | Routing latency (0 for agent) |
| `Retrieve (s)` | float | Retrieval latency (0 for agent) |
| `Generate (s)` | float | Generation latency (0 for agent) |
| `Validate (s)` | float | Validation latency (0 for agent) |
| `Total (s)` | float | Total pipeline/agent latency |
| `Exec OK` | bool | Whether sandbox execution succeeded |
| `Code Safe` | bool | Whether code passed guardrails |

## Startup Behavior

At import time, `app.py` loads:

1. `Pipeline()` — initializes config, vectorstore, embeddings
2. `AgentPipeline(pipe)` — wraps the pipeline for agentic mode
3. PaySim CSV — `pd.read_csv()` loads 6.3M rows into memory

This takes ~30s on first launch (CSV load dominates).

## Example Queries

### Pipeline Tab
- "Show all fraudulent transactions"
- "How many TRANSFER type transactions?"
- "Average transaction amount"
- "Flag transactions above 10 lakh"
- "Identify mule accounts"

### Agent Tab
- "Find accounts with balance mismatch"
- "Large cash-outs that zeroed sender balance"
- "Top 10 most suspicious accounts by risk score"
- "Hourly fraud pattern — which hours have highest fraud rate?"
- "Detect structuring — multiple txns just below 10 lakh"
- "Build risk score: velocity + amount deviation + balance mismatch"

## Running

```bash
# On remote server
cd /path/to/rag-financial-analyst/src
python app.py
# Gradio launches on 0.0.0.0:7860

# On local machine (SSH tunnel)
ssh -L 7860:localhost:7860 user@server
# Open http://localhost:7860
```

## Dependencies

- `src/pipeline.py` — `Pipeline` class, `PipelineResult`
- `src/agent.py` — `AgentPipeline` class, `AgentResult`
- `src/executor.py` — `execute_code_safe()`
- `gradio` — UI framework
- `pandas` — DataFrame for Stats tab and dataset loading
