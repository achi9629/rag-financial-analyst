# Validation Module (`src/validation.py`)

## Purpose

Defines Pydantic output schemas for structured LLM responses and provides parsing utilities that enforce type safety and graceful fallback when the LLM doesn't comply.

## Pydantic Model

### `CodeOutput`

| Field | Type | Description |
|---|---|---|
| `code` | `str` | Executable pandas/Python code. Must not be empty. |
| `explanation` | `str` | 1-3 sentence description of what the code does. |
| `confidence` | `float` | Model confidence in correctness (0.0–1.0). |
| `columns_used` | `list[str]` | PaySim columns referenced in the code. |

**Validators:**
- `confidence` must be between 0.0 and 1.0
- `columns_used` warns on unexpected column names (doesn't reject — allows derived columns)
- `code` cannot be empty/whitespace

## System Prompt

`STRUCTURED_CODEGEN_SYSTEM_PROMPT` replaces the original `CODEGEN_SYSTEM_PROMPT` from `chain.py`. It forces Qwen to output:

```json
{
    "code": "# pandas code here\nresult = df[df['isFraud'] == 1]\nprint(result)",
    "explanation": "Filters transactions marked as fraudulent.",
    "confidence": 0.95,
    "columns_used": ["isFraud"]
}
```

## Parsing Functions

### `parse_code_output(response: str) -> CodeOutput`

Strict parser:
1. Strips markdown fences if present
2. Parses JSON
3. Converts `\\n` → actual newlines in code field
4. Validates via Pydantic

Raises `ValueError` on failure.

### `parse_code_output_with_fallback(response: str, raw_code_fallback=True) -> CodeOutput`

Graceful parser:
1. Tries `parse_code_output()` first
2. On failure (if `raw_code_fallback=True`): treats entire response as raw code
3. Wraps in `CodeOutput` with defaults:
   - `confidence = 0.5`
   - `explanation = "(Auto-extracted: LLM did not return structured JSON)"`
   - `columns_used` = auto-detected from code string

### `_extract_columns_from_code(code: str) -> list[str]`

Scans code string for known PaySim column names. Used in the fallback path.

## Confidence Guidelines

| Range | Meaning |
|---|---|
| 0.9–1.0 | Straightforward query, high certainty |
| 0.7–0.9 | Requires assumptions or domain inference |
| < 0.7 | Uncertain, may need validation |
| 0.5 | Fallback default (raw code extraction) |

## Integration with `chain.py`

To use structured output:

```python
from validation import STRUCTURED_CODEGEN_SYSTEM_PROMPT, parse_code_output_with_fallback

# In generate_code():
response = call_llm(config=config,
                    prompt=prompt,
                    model="qwen",
                    system=STRUCTURED_CODEGEN_SYSTEM_PROMPT,
                    temperature=0)

result = parse_code_output_with_fallback(response)
# result.code, result.explanation, result.confidence, result.columns_used
```

## Testing

Run standalone:
```bash
cd src && python validation.py
```

Tests:
1. Valid structured JSON → parses correctly
2. Raw code (no JSON) → fallback extracts code + auto-detects columns
3. Invalid confidence (>1.0) → raises ValidationError
