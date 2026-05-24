import signal
import logging
import threading
import numpy as np
import pandas as pd
from io import StringIO
from typing import Optional
from contextlib import redirect_stdout

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

class ExecutionResult:
    
    """
    Description:
        Container for sandbox execution results.
    Fields:
        success: Whether the code executed without errors.
        output: Captured stdout from the code execution.
        error: Error message if execution failed, None otherwise.
    """
    
    def __init__(self, success: bool, output: str = "", error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error
        
def execute_code(code: str, 
                 df: pd.DataFrame, 
                 timeout: int = 10
        ) -> ExecutionResult:
    
    """
    Description:
        Execute generated pandas code in a restricted sandbox. Only pd, np, df,
        and print are available. Captures stdout and enforces a timeout.
        Uses signal.SIGALRM in main thread, threading.Thread timeout otherwise.
    Args:
        code: The generated Python code string to execute.
        df: The DataFrame to make available as 'df' in the execution context.
        timeout: Maximum execution time in seconds (default 10).
    Returns:
        ExecutionResult with success status, captured output, and error details.
    """
    
    # Restricted globals — only safe objects available
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "round": round,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "isinstance": isinstance,
            "True": True,
            "False": False,
            "None": None,
        },
        "pd": pd,
        "np": np,
        "df": df,
    }

    # Use signal-based timeout in main thread, thread-based otherwise
    if threading.current_thread() is threading.main_thread():
        return _execute_with_signal(code, restricted_globals, timeout)
    else:
        return _execute_with_thread(code, restricted_globals, timeout)


def _execute_with_signal(code, restricted_globals, timeout):
    """Execute code with signal.SIGALRM timeout (main thread only)."""
    stdout_capture = StringIO()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        with redirect_stdout(stdout_capture):
            exec(code, restricted_globals)
        signal.alarm(0)
        output = stdout_capture.getvalue()
        logger.info(f"Code executed successfully. Output length: {len(output)} chars")
        return ExecutionResult(success=True, output=output)

    except TimeoutError:
        signal.alarm(0)
        logger.warning(f"Code execution timed out after {timeout}s")
        return ExecutionResult(success=False, output="", error=f"Execution timed out after {timeout}s")

    except Exception as e:
        signal.alarm(0)
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"Code execution failed: {error_msg}")
        return ExecutionResult(success=False, output=stdout_capture.getvalue(), error=error_msg)

    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _execute_with_thread(code, restricted_globals, timeout):
    """Execute code with thread-based timeout (safe for worker threads / Gradio)."""
    stdout_capture = StringIO()
    result_container = {"success": False, "output": "", "error": None}

    def _run():
        try:
            with redirect_stdout(stdout_capture):
                exec(code, restricted_globals)
            result_container["success"] = True
            result_container["output"] = stdout_capture.getvalue()
        except Exception as e:
            result_container["error"] = f"{type(e).__name__}: {e}"
            result_container["output"] = stdout_capture.getvalue()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        logger.warning(f"Code execution timed out after {timeout}s (thread-based)")
        return ExecutionResult(success=False, output="", error=f"Execution timed out after {timeout}s")

    if result_container["error"]:
        logger.warning(f"Code execution failed: {result_container['error']}")
        return ExecutionResult(success=False, output=result_container["output"], error=result_container["error"])

    logger.info(f"Code executed successfully. Output length: {len(result_container['output'])} chars")
    return ExecutionResult(success=True, output=result_container["output"])

        
def execute_code_safe(code: str, 
                      df: pd.DataFrame, 
                      timeout: int = 10
        ) -> ExecutionResult:
    
    """
    Description:
        Execute code with a sanity check — first runs on df.head(100) to catch
        errors quickly, then runs on the full DataFrame.
    Args:
        code: The generated Python code string to execute.
        df: The full DataFrame.
        timeout: Maximum execution time in seconds.
    Returns:
        ExecutionResult from full execution if sanity check passes.
    """
    
    # Sanity check on small sample first
    sample_result = execute_code(code, df.head(100), timeout=5)
    if not sample_result.success:
        logger.warning(f"Sanity check failed on df.head(100): {sample_result.error}")
        return sample_result

    # Full execution
    return execute_code(code, df, timeout=timeout)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create sample DataFrame for testing
    sample_df = pd.DataFrame({
                            "step": [1, 2, 3, 4, 5],
                            "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "PAYMENT", "CASH_OUT"],
                            "amount": [100000, 50000, 200000, 5000, 150000],
                            "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
                            "oldbalanceOrg": [100000, 80000, 200000, 10000, 150000],
                            "newbalanceOrig": [0, 30000, 0, 5000, 0],
                            "nameDest": ["C10", "C20", "C30", "C40", "C50"],
                            "oldbalanceDest": [0, 0, 0, 0, 0],
                            "newbalanceDest": [100000, 50000, 200000, 5000, 150000],
                            "isFraud": [1, 0, 1, 0, 1],
                            "isFlaggedFraud": [0, 0, 0, 0, 0],
                            })

    # Test 1: Valid code
    logger.info("--- Test 1: Valid code ---")
    result = execute_code("fraud = df[df['isFraud'] == 1]\nprint(fraud.shape)", sample_df)
    logger.info(f"Success: {result.success}, Output: {result.output.strip()}")

    # Test 2: Runtime error (bad column)
    logger.info("\n--- Test 2: Runtime error ---")
    result = execute_code("print(df['nonexistent_col'])", sample_df)
    logger.info(f"Success: {result.success}, Error: {result.error}")

    # Test 3: Timeout (infinite loop)
    logger.info("\n--- Test 3: Timeout ---")
    result = execute_code("while True: pass", sample_df, timeout=2)
    logger.info(f"Success: {result.success}, Error: {result.error}")

    # Test 4: Blocked builtins (open not available)
    logger.info("\n--- Test 4: Blocked builtin ---")
    result = execute_code("open('/etc/passwd')", sample_df)
    logger.info(f"Success: {result.success}, Error: {result.error}")

    # Test 5: execute_code_safe (sanity check + full)
    logger.info("\n--- Test 5: Safe execution ---")
    result = execute_code_safe("print(df[df['isFraud'] == 1]['amount'].sum())", sample_df)
    logger.info(f"Success: {result.success}, Output: {result.output.strip()}")
