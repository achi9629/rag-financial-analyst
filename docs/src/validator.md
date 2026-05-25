# Validator Module (`src/validator.py`)

## Purpose

The validator module implements **Llama-3-70B cross-validation** — a semantic review layer where Llama checks whether Qwen's generated code actually answers the user's query correctly. It combines a fast programmatic column check with an LLM-based code review.

## Validation Flow

```
Generated Code
    │
    ▼
┌───────────────────────────────┐
│  _check_columns_in_code()     │  Fast, deterministic
│  Regex: df['col'], df.col     │  Check against VALID_COLUMNS set
└───────────────────────────────┘
    │
    ├── ❌ Invalid column found → return immediately (skip LLM call)
    │
    └── ✅ All columns valid
         │
         ▼
┌───────────────────────────────┐
│  Llama-3-70B Code Review      │  Semantic validation
│  [GPU 0-1, port 8000]         │  Query alignment, logic, output
└───────────────────────────────┘
    │
    ▼
  ValidatorOutput(valid, issues)
```

**Why two layers?** The column check catches obvious errors (misspelled column names) instantly without an LLM call. This saves ~2-3 seconds of Llama inference when the error is trivially detectable.

## Valid Columns

The `VALID_COLUMNS` set contains all PaySim schema columns:

```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
```

## Classes

### `ValidatorOutput` (Pydantic BaseModel)

| Field | Type | Description |
|---|---|---|
| `valid` | `bool` | Whether the code correctly answers the query |
| `issues` | `List[str]` | Specific issues found (empty if valid) |

## Functions

### `_check_columns_in_code(code) -> List[str]`

Extracts column references from generated code using regex:

| Pattern | Matches |
|---|---|
| `df['col']` or `df["col"]` | Bracket-style column access |
| `df.col` | Dot-style column access (excluding pandas methods) |

**Excluded pandas methods**: `head`, `tail`, `describe`, `groupby`, `merge`, `sort_values`, `drop`, `reset_index`, `copy`, `loc`, `iloc`, `query`, `apply`, `agg`, `nunique`, `value_counts`, `shape`, `columns`, `dtypes`

Returns a list of issues like `"Invalid column name: 'fraud_flag' (not in PaySim schema)"`.

### `validate_code(config, user_query, code) -> ValidatorOutput`

Main validation entry point:

1. **Column check** — calls `_check_columns_in_code()`. If invalid columns found, returns immediately with `valid=False`
2. **Llama review** — sends code + query to Llama-3-70B with `VALIDATOR_SYSTEM_PROMPT`
3. **Parse response** — extracts JSON from Llama's response, strips markdown fences if present
4. **Fallback** — if JSON parsing fails, returns `valid=True` (assumes correct to avoid false blocking)

## Llama Review Criteria

The `VALIDATOR_SYSTEM_PROMPT` instructs Llama to check:

| Criterion | Description |
|---|---|
| Query Alignment | Does the code actually answer the user's question? |
| Column Names | Are all referenced columns valid PaySim columns? |
| Logic Errors | Off-by-one, wrong operators, incorrect aggregations |
| Data Assumptions | Wrong column types, non-existent derived columns |
| Output | Does the code print/display a result? |

Llama responds with JSON:

```json
{"valid": true, "issues": []}
```
or
```json
{"valid": false, "issues": ["Code filters by 'fraud' instead of 'isFraud'"]}
```

## Integration with Pipeline

The validator is called in `Pipeline._validate_code()` as the second layer (after guardrails):

```
_validate_code()
    │
    ├── 1. check_code_safety()   [guardrails.py — fast, security]
    │
    └── 2. validate_code()       [validator.py — semantic, correctness]
```

- **Guardrail failure** → hard stop, no retry
- **Validation failure** → cascade to Try 2 with feedback

The validator is only called on **Try 1** — Tries 2 and 3 only run guardrails (not Llama validation) to reduce latency.

## JSON Parsing

Llama's response is parsed with these fallbacks:

1. Strip leading/trailing whitespace
2. Remove markdown code fences (` ```json ... ``` `)
3. `json.loads()` the cleaned string
4. If parsing fails → return `ValidatorOutput(valid=True, issues=[])` (permissive fallback)
5. If `"valid"` key missing → default to `True`
6. If `"issues"` key missing → default to `[]`

## Usage

```python
from models import load_config
from validator import validate_code

config = load_config("configs/config.yaml")

result = validate_code(
    config=config,
    user_query="Show all fraudulent transactions",
    code="print(df[df['isFraud'] == 1])"
)

print(result.valid)   # True
print(result.issues)  # []
```

## Dependencies

- `src/models.py` — `load_config()`, `call_llm()`
- `pydantic` — `ValidatorOutput` schema
- Llama-3-70B — vLLM server on port 8000 (GPU 0-1)
