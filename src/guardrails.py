import ast
import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# --- Blocklists ---

BLOCKED_MODULES = {
    "os", "subprocess", "sys", "shutil", "pathlib",
    "requests", "urllib", "socket", "http", "ftplib",
    "ctypes", "multiprocessing", "signal", "tempfile",
    "importlib", "pickle", "shelve", "webbrowser",
}

ALLOWED_MODULES = {"pandas", "pd", "numpy", "np", "math", "re", "datetime", "collections"}

BLOCKED_CALLS = {
    "exec", "eval", "compile", "open", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals",
    "breakpoint", "exit", "quit", "input",
}

BLOCKED_ATTRS = {
    "system", "popen", "remove", "rmdir", "unlink", "rmtree",
    "call", "run", "Popen", "check_output", "check_call",
}

# Pandas methods that do file I/O (df is pre-loaded, no reads/writes allowed)
BLOCKED_PANDAS_IO = {
    "read_csv", "read_excel", "read_json", "read_parquet",
    "read_sql", "read_html", "read_clipboard", "read_feather",
    "to_csv", "to_excel", "to_json", "to_parquet", "to_sql",
    "to_pickle", "to_html", "to_clipboard",
}

DANGEROUS_DUNDERS = r"__(?:subclasses|bases|globals|builtins|import|code|class|mro|dict)__"

SHELL_PATTERNS = re.compile(
    r"(?:rm\s+-rf|curl\s+|wget\s+|chmod\s+|chown\s+|mkfs|dd\s+if=)",
    re.IGNORECASE,
)

SQL_PATTERNS = re.compile(
    r"(?:DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UNION\s+SELECT|;\s*--|;\s*DROP)",
    re.IGNORECASE,
)


def check_code_safety(code: str) -> Tuple[bool, List[str]]:
    
    """
    Description:
        Validate generated code for safety before execution. Uses AST analysis
        for structural checks and regex for string-level pattern detection.
    Args:
        code: The generated Python code string to validate.
    Returns:
        Tuple[bool, List[str]]: (is_safe, issues)
            is_safe: True if code passed all safety checks, False otherwise.
            issues: List of detected safety issues (empty if is_safe is True).
    """
    issues = []

    # --- 1. AST structural checks ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in BLOCKED_MODULES:
                    issues.append(f"Blocked import: {alias.name}")
                elif root_module not in ALLOWED_MODULES:
                    issues.append(f"Unrecognized import: {alias.name} (only pandas/numpy allowed)")

        elif isinstance(node, ast.ImportFrom):
            root_module = (node.module or "").split(".")[0]
            if root_module in BLOCKED_MODULES:
                issues.append(f"Blocked import from: {node.module}")
            elif root_module not in ALLOWED_MODULES:
                issues.append(f"Unrecognized import from: {node.module} (only pandas/numpy allowed)")

        # Block dangerous function calls: exec(), eval(), open(), etc.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                issues.append(f"Blocked call: {node.func.id}()")

        # Block dangerous attribute calls: os.system(), subprocess.run(), etc.
        if isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                if isinstance(node.value, ast.Name) and node.value.id in BLOCKED_MODULES:
                    issues.append(f"Blocked call: {node.value.id}.{node.attr}()")

            # Block pandas file I/O: pd.read_csv(), df.to_csv(), etc.
            if node.attr in BLOCKED_PANDAS_IO:
                issues.append(f"Blocked pandas I/O: {node.attr}() -- df is pre-loaded")

    # --- 2. Regex fallback checks (catches string obfuscation) ---
    if re.search(DANGEROUS_DUNDERS, code):
        issues.append("Dangerous dunder attribute access detected")

    if SHELL_PATTERNS.search(code):
        issues.append("Shell command pattern detected in string literal")

    if SQL_PATTERNS.search(code):
        issues.append("SQL injection pattern detected")

    is_safe = len(issues) == 0
    if not is_safe:
        logger.warning(f"Code safety check FAILED: {issues}")
    else:
        logger.info("Code safety check passed")

    return is_safe, issues


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Test 1: Safe pandas code
    safe_code = "import pandas as pd\nimport numpy as np\nfraud = df[df[\'isFraud\'] == 1]\nprint(fraud.head(10))"
    is_safe, issues = check_code_safety(safe_code)
    logger.info(f"Test 1 (safe pandas): " + ("PASS" if is_safe else "FAIL") + " -- " + str(issues))

    # Test 2: os.system
    dangerous_os = "import os\nos.system(\'rm -rf /\')"
    is_safe, issues = check_code_safety(dangerous_os)
    logger.info("Test 2 (os.system):   " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 3: subprocess
    dangerous_sub = "import subprocess\nsubprocess.run([\'ls\'])"
    is_safe, issues = check_code_safety(dangerous_sub)
    logger.info("Test 3 (subprocess):  " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 4: open() for file access
    dangerous_open = "open(\'/etc/passwd\').read()"
    is_safe, issues = check_code_safety(dangerous_open)
    logger.info("Test 4 (open):        " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 5: eval/exec
    dangerous_eval = "eval(\'malicious code\')"
    is_safe, issues = check_code_safety(dangerous_eval)
    logger.info("Test 5 (eval):        " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 6: pandas read_csv (should be blocked)
    dangerous_io = "df = pd.read_csv(\'data.csv\')\nprint(df.head())"
    is_safe, issues = check_code_safety(dangerous_io)
    logger.info("Test 6 (pd.read_csv): " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 7: dunder escape attempt
    dangerous_dunder = "x = \'\'.__class__.__bases__[0].__subclasses__()"
    is_safe, issues = check_code_safety(dangerous_dunder)
    logger.info("Test 7 (dunders):     " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 8: SQL injection in string
    dangerous_sql = "query = \'DROP TABLE users\'"
    is_safe, issues = check_code_safety(dangerous_sql)
    logger.info("Test 8 (SQL inject):  " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 9: requests/network
    dangerous_net = "import requests\nrequests.get(\'http://evil.com\')"
    is_safe, issues = check_code_safety(dangerous_net)
    logger.info("Test 9 (network):     " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))

    # Test 10: Syntax error
    bad_syntax = "def foo(:\n  pass"
    is_safe, issues = check_code_safety(bad_syntax)
    logger.info("Test 10 (syntax err): " + ("PASS" if not is_safe else "FAIL") + " -- " + str(issues))
