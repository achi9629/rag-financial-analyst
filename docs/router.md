# Query Router — `src/router.py`

## Overview

The query router classifies incoming natural language queries by complexity and produces a structured execution plan. It uses Llama-3-70B as the classification model to determine how the downstream pipeline should handle each query.

## Pipeline Position

```
User Query  →  [Router (Llama)]  →  Plan  →  Retrieval  →  Code Generation (Qwen)
                     │
                     ▼
              RouterOutput {
                complexity: "simple" | "threshold" | "multi_hop",
                steps: [...],
                retrieval_queries: [...]
              }
```

## Query Complexity Classes

### simple
Direct column lookup, basic aggregation, or single-condition filtering. Requires 1 step and minimal retrieval.

**Examples:**
- "Show all fraudulent transactions"
- "How many TRANSFER type transactions?"
- "Average transaction amount"

### threshold
Requires applying a numeric threshold, rule, or condition from domain knowledge. Needs retrieval of rules/thresholds before code generation.

**Examples:**
- "Flag transactions above ₹10 lakh"
- "Find accounts with balance mismatch"
- "Large cash-outs that zeroed sender balance"

### multi_hop
Requires multiple computation steps, combining rules, or derived features. Needs decomposition into sub-steps and multiple retrievals.

**Examples:**
- "Identify mule accounts"
- "Detect structuring — multiple transactions just below ₹10L"
- "Top 10 most suspicious accounts by risk score"
- "Build composite risk score: velocity + amount deviation + balance mismatch"

## Output Schema (Pydantic)

```python
class RouterOutput(BaseModel):
    complexity: str         # "simple" | "threshold" | "multi_hop"
    steps: list[str]        # Execution steps (1 for simple, 2-5 for multi_hop)
    retrieval_queries: list[str]  # Queries to search the vector store (1-3)
```

### Example Outputs

**Simple query:** `"How many TRANSFER type transactions?"`
```json
{
    "complexity": "simple",
    "steps": ["Filter dataframe for type == TRANSFER and count rows"],
    "retrieval_queries": ["transaction type column schema"]
}
```

**Threshold query:** `"Flag transactions above 10 lakh"`
```json
{
    "complexity": "threshold",
    "steps": ["Filter transactions where amount > 1000000", "Flag them as suspicious"],
    "retrieval_queries": ["regulatory reporting threshold", "high value transaction rule"]
}
```

**Multi-hop query:** `"Identify mule accounts"`
```json
{
    "complexity": "multi_hop",
    "steps": [
        "Filter fraud-eligible types (TRANSFER, CASH_OUT)",
        "Find accounts that receive via TRANSFER then send via CASH_OUT",
        "Check for rapid turnaround (within same or next step)",
        "Flag accounts matching the mule pattern"
    ],
    "retrieval_queries": ["mule account pattern", "fraud transaction types", "sender receiver pair analysis"]
}
```

## Functions

### `route_query(config, user_query) → RouterOutput`
Main entry point. Sends the user query to Llama with a system prompt instructing JSON-only output. Parses the response into a validated `RouterOutput` object.

- **Model:** Llama-3-70B (via `call_llm` with `model="llama"`)
- **Temperature:** 0 (deterministic classification)
- **Fallback:** If JSON parsing fails, defaults to `complexity="simple"` with the original query as the single step and retrieval query

### `RouterOutput` (Pydantic model)
Validated structured output. The `complexity` field is constrained to exactly `{"simple", "threshold", "multi_hop"}` via a field validator.

## System Prompt Design

The router system prompt:
1. Defines the three complexity classes with examples
2. Specifies the exact JSON output format
3. Instructs "Respond ONLY with valid JSON (no markdown, no explanation)"
4. Provides rules for each field (step count ranges, retrieval query count)

## Error Handling

| Scenario | Behavior |
|---|---|
| Llama returns invalid JSON | Log error, fallback to `simple` classification |
| Llama wraps JSON in markdown code blocks | Automatically stripped before parsing |
| `complexity` value not in allowed set | Pydantic raises `ValidationError` |
| Llama server unreachable | Exception propagates from `call_llm` |

## Usage

```python
from models import load_config
from router import route_query

config = load_config("configs/config.yaml")
result = route_query(config, "Detect structuring in TRANSFER transactions")

print(result.complexity)         # "multi_hop"
print(result.steps)              # ["Filter TRANSFER...", "Group by sender...", ...]
print(result.retrieval_queries)  # ["structuring pattern", "regulatory threshold"]
```

## Testing

```bash
export no_proxy="localhost,127.0.0.1"
python src/router.py
```

Runs 9 test queries (3 per class) and reports classification accuracy. Target: 90%+ correct.

## Design Decisions

| Decision | Rationale |
|---|---|
| Llama for routing (not Qwen) | 70B reasons better for planning; Qwen is specialized for code |
| JSON-only output | Parseable, no regex needed, Pydantic-validatable |
| temperature=0 | Deterministic classification — same query always gets same plan |
| Graceful fallback | Pipeline never crashes from bad router output |
| retrieval_queries field | Router decides what to search — decouples routing from retrieval logic |
