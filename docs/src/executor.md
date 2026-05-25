# Executor Module (`src/executor.py`)

## Purpose

The executor module implements **sandboxed code execution** — running LLM-generated pandas code against the PaySim DataFrame with restricted globals, timeout enforcement, and a sanity-check-first strategy. It ensures generated code cannot access the filesystem, network, or dangerous builtins.

## Security Model

```
Generated Code
    │
    ▼
┌───────────────────────────────────┐
│  Restricted Globals               │
│                                   │
│  ✅ Allowed:                      │
│    pd, np, df, print              │
│    len, range, int, float, str    │
│    list, dict, set, tuple, bool   │
│    round, abs, min, max, sum      │
│    sorted, enumerate, zip         │
│    isinstance, True, False, None  │
│                                   │
│  ❌ Blocked (not in globals):     │
│    open, import, os, sys          │
│    subprocess, requests, eval     │
│    exec, __import__, compile      │
│    file I/O, network, shell       │
└───────────────────────────────────┘
    │
    ▼
  exec(code, restricted_globals)
    │
    ▼
  ExecutionResult(success, output, error)
```

> **Note**: This is a defense-in-depth layer. `src/guardrails.py` blocks dangerous patterns via AST/regex **before** code reaches the executor. The restricted globals are a second barrier.

## Timeout Strategy

The executor uses two timeout mechanisms depending on the calling thread:

```
execute_code()
    │
    ├── Main thread?
    │   └── YES → _execute_with_signal()  [signal.SIGALRM]
    │
    └── Worker thread? (Gradio, etc.)
        └── YES → _execute_with_thread()  [threading.Thread + join]
```

| Method | Mechanism | When Used |
|---|---|---|
| `_execute_with_signal()` | `signal.SIGALRM` | Main thread (CLI, `__main__`) |
| `_execute_with_thread()` | `threading.Thread` + `join(timeout)` | Worker threads (Gradio, pytest) |

**Why two methods?** `signal.SIGALRM` only works in the main thread — calling it from a Gradio worker thread raises `ValueError: signal only works in main thread`. The thread-based fallback handles this case.

## Classes

### `TimeoutError`

Custom exception raised by `_timeout_handler` when `SIGALRM` fires.

### `ExecutionResult`

Container for sandbox execution results:

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether code executed without errors |
| `output` | `str` | Captured stdout from the code |
| `error` | `Optional[str]` | Error message if failed, `None` if succeeded |

## Functions

### `execute_code(code, df, timeout=10) -> ExecutionResult`

Core execution function. Builds restricted globals, detects thread context, and dispatches to the appropriate timeout implementation.

**Restricted globals include:**
- `pd` (pandas), `np` (numpy), `df` (the DataFrame)
- Safe builtins: `print`, `len`, `range`, `int`, `float`, `str`, `list`, `dict`, `set`, `tuple`, `bool`, `round`, `abs`, `min`, `max`, `sum`, `sorted`, `enumerate`, `zip`, `isinstance`
- Constants: `True`, `False`, `None`

**Everything else is blocked** — no `open()`, no `import`, no `os`, no `eval`.

### `_execute_with_signal(code, restricted_globals, timeout) -> ExecutionResult`

Signal-based timeout for main thread execution:

1. Install `SIGALRM` handler
2. Set alarm for `timeout` seconds
3. `exec(code, restricted_globals)` with stdout capture
4. Cancel alarm on success
5. Restore original signal handler in `finally`

### `_execute_with_thread(code, restricted_globals, timeout) -> ExecutionResult`

Thread-based timeout for worker thread execution (Gradio):

1. Spawn daemon thread to run `exec(code, restricted_globals)`
2. `thread.join(timeout)` — wait up to `timeout` seconds
3. If thread is still alive after join, report timeout
4. Otherwise, return captured output or error

**Limitation**: The daemon thread cannot be forcibly killed if it hangs on a blocking syscall. The thread is abandoned but the process continues. For production, use `nsjail` or `RestrictedPython`.

### `execute_code_safe(code, df, timeout=10) -> ExecutionResult`

Two-phase execution with sanity check:

```
Phase 1: execute_code(code, df.head(100), timeout=5)
    │
    ├── ❌ Failed → return error immediately (fast fail)
    │
    └── ✅ Passed
         │
         ▼
Phase 2: execute_code(code, df, timeout=timeout)
    │
    └── Return result (success or error)
```

**Why sanity check?** Running on 6.3M rows is slow. Catching errors on 100 rows first gives faster feedback and avoids wasting 30s on obviously broken code.

## Usage

```python
import pandas as pd
from executor import execute_code, execute_code_safe

df = pd.read_csv("assets/datasets/PaySim/PS_20174392719_1491204439457_log.csv")

# Direct execution
result = execute_code("print(df[df['isFraud'] == 1].shape[0])", df, timeout=10)
print(result.success)  # True
print(result.output)   # "8213\n"

# Safe execution (sanity check + full run)
result = execute_code_safe("print(df.groupby('type')['amount'].mean())", df)
print(result.output)
```

## Error Handling

| Scenario | Result |
|---|---|
| Code runs successfully | `ExecutionResult(success=True, output="...", error=None)` |
| Runtime error (KeyError, etc.) | `ExecutionResult(success=False, output="partial", error="KeyError: 'col'")` |
| Timeout (signal or thread) | `ExecutionResult(success=False, output="", error="Execution timed out after 10s")` |
| Sanity check fails | Returns sample error immediately, skips full run |
| Blocked builtin (e.g., `open`) | `NameError: name 'open' is not defined` |

## Production Considerations

The current sandbox is sufficient for a demo but has known limitations:

| Limitation | Mitigation |
|---|---|
| `exec()` runs in-process | Use `nsjail` or Docker container for isolation |
| Thread-based timeout can't kill blocking code | Use process-based timeout (`multiprocessing`) |
| No memory limit enforcement | Set `resource.setrlimit()` or use cgroups |
| No filesystem isolation | Use `chroot` or container |

These are noted in the project README.

## Dependencies

- `signal` — SIGALRM timeout (stdlib)
- `threading` — thread-based timeout fallback (stdlib)
- `pandas` — DataFrame execution context
- `numpy` — numeric operations in generated code
- `contextlib.redirect_stdout` — stdout capture
