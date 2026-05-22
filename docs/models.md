# LLM Client — `src/models.py`

## Overview

The models module provides a unified interface for calling LLM endpoints (vLLM-served models) via the OpenAI-compatible API. It handles configuration loading, client management, and request formatting for both Qwen (code generation) and Llama (routing/validation).

## Pipeline Position

```
                    ┌─────────────────────────┐
                    │      models.py          │
                    │  ┌───────────────────┐  │
Router ─────────────┤  │  call_llm(llama)  │  ├──── http://localhost:8000/v1
                    │  └───────────────────┘  │
Chain ──────────────┤  ┌───────────────────┐  ├──── http://localhost:8001/v1
                    │  │  call_llm(qwen)   │  │
Validator ──────────┤  └───────────────────┘  │
                    └─────────────────────────┘
```

## Configuration

Models are defined in `configs/config.yaml`:

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
```

## Functions

### `load_config(config_path) → dict`
Loads the full YAML configuration file and returns it as a dictionary.

- **Input:** Path to `config.yaml`
- **Returns:** Complete config dict (models, proxy, ingestion sections)
- **Raises:** `FileNotFoundError` if path doesn't exist

### `_get_client(config, model_key) → tuple[OpenAI, dict]`
Internal function that returns a cached OpenAI client instance and model config for the given key.

- **Caching:** Clients are stored in the module-level `_clients` dict — one connection per model, reused across calls
- **Returns:** `(client, cfg)` tuple where `cfg` contains `base_url`, `model_name`, `temperature`, `max_tokens`
- **Raises:** `KeyError` if `model_key` not found in config

### `call_llm(config, prompt, model, system, temperature, max_tokens) → str`
Main entry point for all LLM calls in the pipeline.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | dict | required | Full config dict from `load_config` |
| `prompt` | str | required | User/input message content |
| `model` | str | `"qwen"` | Model key from config (`"qwen"` or `"llama"`) |
| `system` | str | `None` | Optional system prompt (prepended as system message) |
| `temperature` | float | `None` | Override config temperature (None = use config default) |
| `max_tokens` | int | `None` | Override config max_tokens (None = use config default) |

**Returns:** The model's response text (`response.choices[0].message.content`)

**Message construction:**
```python
messages = []
if system:
    messages.append({"role": "system", "content": system})
messages.append({"role": "user", "content": prompt})
```

**Message roles explained:**

| Role | `content` receives | Purpose |
|---|---|---|
| `"system"` | The `system` param (e.g., `ROUTER_SYSTEM_PROMPT`) | Behavioral instructions — tells the model *how* to respond. Not a question to answer. |
| `"user"` | The `prompt` param (the actual query) | The request the model generates a response *to* |
| `"assistant"` | (used in multi-turn, not used here) | Model's own prior responses — maintains conversation history |

**`client.chat.completions.create` parameters:**

| Parameter | What it controls |
|---|---|
| `model` | Which model on the vLLM server handles the request (maps to `cfg["model_name"]`) |
| `messages` | Conversation context — ordered list of `{"role": ..., "content": ...}` dicts |
| `temperature` | Sampling randomness: 0 = deterministic (same input → same output), higher = more varied |
| `max_tokens` | Maximum number of tokens the model can generate in its response |

## Client Architecture

```
call_llm(model="qwen")
    │
    ├── _get_client("qwen")
    │       │
    │       ├── Cache hit? → return existing client
    │       └── Cache miss? → OpenAI(base_url="http://localhost:8001/v1")
    │
    └── client.chat.completions.create(...)
            │
            └── HTTP POST → vLLM server → response
```

## Usage

```python
from models import load_config, call_llm

config = load_config("configs/config.yaml")

# Code generation (Qwen)
code = call_llm(config, "Write pandas to count fraudulent transactions", model="qwen")

# Routing/classification (Llama)
plan = call_llm(config, "Identify mule accounts", model="llama", system=ROUTER_PROMPT)

# Override temperature for creative output
response = call_llm(config, prompt, model="qwen", temperature=0.3)
```

## Error Scenarios

| Scenario | Error | Fix |
|---|---|---|
| vLLM server not running | `ConnectionRefusedError` | Start server with `run_qwen.sh` / `run_llama.sh` |
| Proxy intercepting request | `openai.PermissionDeniedError` (HTML response) | Set `no_proxy` env var |
| Model key typo | `KeyError` from `_get_client` | Check config.yaml model keys |
| Token limit exceeded | Truncated response | Increase `max_tokens` in config or call |

## Design Decisions

| Decision | Rationale |
|---|---|
| OpenAI client (not requests) | vLLM exposes OpenAI-compatible API; client handles retries, streaming |
| Module-level client cache | Avoids reconnection overhead on every call |
| Config-driven defaults | Change temperature/tokens without code changes |
| `api_key="not-needed"` | vLLM doesn't authenticate; OpenAI client requires a non-empty string |
| Separate model keys | Same `call_llm` interface for both models; caller picks via `model=` param |
