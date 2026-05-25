# Tracking Module (`src/tracking.py`)

## Purpose

The tracking module implements **MLflow experiment tracking** for the RAG pipeline. It logs per-query metrics (latency, model, fallback stage) and computes aggregate summaries (correctness rate, average latency, fallback frequency) for model comparison and performance analysis. All data is stored locally — no account or cloud service required.

## Architecture

```
Pipeline.run_query()
    │
    ├── track_query()  ──→  MLflow step metrics  (per-query metrics)
    │
    └── (at session end)
         │
         ├── log_summary() ──→  MLflow summary metrics  (aggregate metrics)
         │                      JSON artifact           (query history)
         └── finish()      ──→  MLflow run closed
```

## Class: `QueryTracker`

### `__init__(self, project, run_name, config)`

Initializes an MLflow experiment and starts a run:

| Parameter | Default | Description |
|---|---|---|
| `project` | `"rag-financial-analyst"` | MLflow experiment name |
| `run_name` | `None` (auto-generated) | Human-readable run name |
| `config` | `{}` | Pipeline config dict logged as MLflow params for reproducibility |

Also initializes `_history: List[Dict]` to store all query records locally for summary computation, and `_step: int` counter for metric step tracking.

### `track_query(...) -> None`

Logs a single query's metrics to MLflow and stores in local history.

**Per-query metrics logged:**

| Metric | MLflow Key | Type | Description |
|---|---|---|---|
| Query type | (history only) | str | simple / threshold / multi_hop |
| Fallback stage | `fallback_stage` | int | 1–4 (which try succeeded) |
| Model used | (history only) | str | qwen / llama / blocked / none |
| Route latency | `latency_route` | float | Time for query routing |
| Retrieve latency | `latency_retrieve` | float | Time for ChromaDB retrieval |
| Generate latency | `latency_generate` | float | Time for LLM code generation |
| Validate latency | `latency_validate` | float | Time for guardrails + Llama validation |
| Total latency | `latency_total` | float | End-to-end wall time |
| Confidence | `confidence` | float | LLM confidence score |
| Code safe | `code_safe` | int | 1 if passed guardrails, 0 if blocked |
| Context length | `context_length` | int | Retrieved context size in chars |

**Not logged as metrics** (stored in history only): `query` text, `query_type`, `model_used`, `columns_used` list.

### `log_summary() -> None`

Computes and logs aggregate metrics from `_history`:

**Summary metrics:**

| Metric | MLflow Key | Description |
|---|---|---|
| Total queries | `summary_total_queries` | Number of queries in the session |
| First-pass rate | `summary_first_pass_rate` | % succeeded on Try 1 |
| Code safety rate | `summary_code_safe_rate` | % not blocked by guardrails |
| Avg confidence | `summary_avg_confidence` | Mean confidence across queries |
| Avg route latency | `summary_avg_latency_route` | Mean routing time |
| Avg retrieve latency | `summary_avg_latency_retrieve` | Mean retrieval time |
| Avg generate latency | `summary_avg_latency_generate` | Mean generation time |
| Avg validate latency | `summary_avg_latency_validate` | Mean validation time |
| Avg total latency | `summary_avg_latency_total` | Mean end-to-end time |

**Distribution metrics:**

| Metric | MLflow Key Pattern | Description |
|---|---|---|
| Fallback stage counts | `summary_fallback_stage_{1-4}_count` | How many queries hit each stage |
| Fallback stage % | `summary_fallback_stage_{1-4}_pct` | Percentage per stage |
| Query type counts | `summary_query_type_{type}_count` | Distribution by complexity |

Also logs a **JSON artifact** (`query_history.json`) with all queries for later analysis.

### `finish() -> None`

Calls `log_summary()` then `mlflow.end_run()` to close the MLflow run.

## Integration with Pipeline

Tracking is **opt-in** — disabled by default:

```python
pipe = Pipeline()
# pipe.tracker is None — no MLflow logging

pipe.enable_tracking(project="rag-financial-analyst", run_name="eval-v1")
# pipe.tracker is now a QueryTracker — every run_query() logs to MLflow

# After all queries:
pipe.tracker.log_summary()
pipe.tracker.finish()
```

The pipeline's `_track()` helper calls `tracker.track_query()` at every return point (all 4 fallback stages).

## Viewing Results

```bash
# Start the MLflow UI (local, no account needed)
mlflow ui --port 5000
# Open http://localhost:5000
```

The MLflow dashboard shows:

- **Metric charts**: latency_total over steps, confidence over time
- **Run comparison**: compare different eval runs side-by-side
- **Parameters panel**: model endpoints, temperatures, retrieval k values
- **Artifacts**: download `query_history.json` for detailed per-query analysis

## Usage

```python
from tracking import QueryTracker

tracker = QueryTracker(project="rag-financial-analyst", run_name="test-run")

tracker.track_query(
    user_query="Show all fraudulent transactions",
    query_type="simple",
    fallback_stage=1,
    model_used="qwen",
    latency_route=0.5,
    latency_retrieve=0.3,
    latency_generate=2.1,
    latency_validate=1.2,
    total_latency=4.1,
    confidence=0.95,
    code_safe=True,
    context_length=1200,
    columns_used=["isFraud"]
)

tracker.log_summary()
tracker.finish()
```

## Dependencies

- `mlflow` — experiment tracking (fully local, open-source)
- `src/pipeline.py` — calls `track_query()` via `_track()` helper
