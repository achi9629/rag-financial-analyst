# Guardrails — Code Safety Layer

## What Are Guardrails?

Guardrails are safety checks that validate LLM-generated code **before execution**. Since the pipeline `exec()`s code that an LLM wrote, guardrails act as a gatekeeper — blocking anything dangerous that the model might produce (intentionally or not).

## Why Are Guardrails Needed?

LLMs generate code based on statistical patterns, not intent. They can produce code that:

| Risk | Example | Impact |
|---|---|---|
| File system access | `open('/etc/passwd').read()` | Data exfiltration |
| Shell execution | `os.system('rm -rf /')` | System destruction |
| Network calls | `requests.get('http://evil.com')` | Data leak / C2 callback |
| Sandbox escape | `''.__class__.__bases__[0].__subclasses__()` | Arbitrary code execution |
| File I/O via pandas | `pd.read_csv('/sensitive/data.csv')` | Unauthorized file reads |
| Code injection | `eval('__import__("os").system("whoami")')` | Privilege escalation |

Even without malicious intent, an LLM might generate `pd.read_csv('transactions.csv')` simply because that's common in training data — but in our pipeline, `df` is pre-loaded, so any file I/O is unnecessary and risky.

## How It Works

`check_code_safety(code: str) -> tuple[bool, list[str]]`

Two-layer defense:

### Layer 1 — AST Analysis (Structural)

Parses the code into an Abstract Syntax Tree and walks every node:

- **Import check**: Blocks `os`, `subprocess`, `sys`, `shutil`, `requests`, `socket`, `pickle`, etc. Only allows `pandas`, `numpy`, `math`, `datetime`, `re`, `collections`.
- **Function call check**: Blocks `exec()`, `eval()`, `compile()`, `open()`, `__import__()`, `getattr()`, `globals()`, `input()`.
- **Attribute check**: Blocks `os.system()`, `subprocess.run()`, `shutil.rmtree()`, etc.
- **Pandas I/O check**: Blocks `pd.read_csv()`, `df.to_csv()`, `read_excel()`, `to_parquet()`, etc. — `df` is pre-loaded, no file I/O needed.

### Layer 2 — Regex Fallback (String-Level)

Catches obfuscation attempts that survive AST analysis:

- **Dunder access**: `__subclasses__`, `__builtins__`, `__globals__`, `__import__`, `__code__`, `__mro__`
- **Shell commands**: `rm -rf`, `curl`, `wget`, `chmod`, `dd if=`
- **SQL injection**: `DROP TABLE`, `DELETE FROM`, `UNION SELECT`, `; --`

## Where It Fits in the Pipeline

```
Qwen generates code
    │
    ▼
┌───────────────────────────────────────┐
│  Guardrails                           │  ← YOU ARE HERE
│  check_code_safety()                  │
│  is_safe? ──No──→ reject + log issues |
│     │                                 |
│    Yes                                |
│     ▼                                 |
│  Llama validation                     │  (Task 4.2)
│     │                                 |
│    Yes                                |
│     ▼                                 |
│  Sandbox exec()                       │  (Task 4.3)
└───────────────────────────────────────┘
```

Guardrails run **before** sandbox execution — they're the first gate. Even if the sandbox has restricted globals, guardrails provide defense-in-depth by rejecting dangerous code before it ever reaches `exec()`.

## API

```python
from guardrails import check_code_safety

code = "import os\nos.system('rm -rf /')"
is_safe, issues = check_code_safety(code)
# is_safe = False
# issues = ["Blocked import: os", "Blocked call: os.system()"]
```

## Blocklist Summary

| Category | Blocked | Count |
|---|---|---|
| Modules | os, subprocess, sys, shutil, pathlib, requests, urllib, socket, http, ftplib, ctypes, multiprocessing, signal, tempfile, importlib, pickle, shelve, webbrowser | 18 |
| Function calls | exec, eval, compile, open, `__import__`, getattr, setattr, delattr, globals, locals, breakpoint, exit, quit, input | 14 |
| Attribute calls | system, popen, remove, rmdir, unlink, rmtree, call, run, Popen, check_output, check_call | 11 |
| Pandas I/O | read_csv, read_excel, read_json, read_parquet, read_sql, read_html, read_clipboard, read_feather, to_csv, to_excel, to_json, to_parquet, to_sql, to_pickle, to_html, to_clipboard | 16 |

## Limitations

- **Not a sandbox**: Guardrails are static analysis only. A determined attacker could bypass them (e.g., building strings dynamically). That's why we also have sandbox execution (Task 4.3) with restricted `exec()` globals.
- **False positives possible**: A legitimate query mentioning "DROP TABLE" in a string literal would be flagged. Acceptable trade-off for safety.
- **Python-only**: Only validates Python/pandas code. Doesn't check for logic errors (that's Llama's job in Task 4.2).

## Production Notes

For production deployment, consider:
- **RestrictedPython**: Compile-time restrictions on Python bytecode
- **nsjail / gVisor**: OS-level sandboxing with namespace isolation
- **seccomp**: System call filtering at kernel level

Current approach (AST + regex + restricted exec) is appropriate for a demo/prototype with trusted users.
