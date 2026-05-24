import re
import json
import logging
from typing import List
from pathlib import Path
from pydantic import BaseModel

from models import load_config, call_llm

logger = logging.getLogger(__name__)

VALID_COLUMNS = {
                "step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest",
                "newbalanceDest", "isFraud", "isFlaggedFraud"
                }

VALIDATOR_SYSTEM_PROMPT = """
You are a code reviewer for a financial fraud analysis system.
You will receive:
1. The original user query
2. Generated pandas code that is supposed to answer the query

Your job: Review the code for correctness and potential issues.

## Review Criteria

1. **Query Alignment**: Does the code actually answer the user's question?
2. **Column Names**: Are column names valid PaySim columns? Valid: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
3. **Logic Errors**: Off-by-one errors, wrong comparison operators, incorrect aggregations.
4. **Data Assumptions**: Does it assume columns that don't exist or data types that are wrong?
5. **Output**: Does it print/display a result?

## Output Format

Respond ONLY with valid JSON (no markdown, no explanation):
{
    "valid": true or false,
    "issues": ["issue 1 description", "issue 2 description"]
}

Rules:
- "valid": true if the code is correct and answers the query. false if there are bugs or it doesn't answer the query.
- "issues": List of specific issues found. Empty list [] if valid is true. Be concise (1 sentence per issue).
- If the code is mostly correct but has a minor style issue, still mark valid as true with an empty issues list.
"""

class ValidatorOutput(BaseModel):
    """
    Description:
        Structured output from Llama cross-validation.
    Fields:
        valid: Whether the generated code correctly answers the query.
        issues: List of specific issues found in the code (empty if valid).
    """
    valid: bool
    issues: List[str]
    
def _check_columns_in_code(code: str) -> List[str]:
    
    """
    Description:
        Helper function to extract column references from the code and check against valid columns.
    Args:
        code: The generated pandas code to analyze.
    Returns:
        List[str]: List of issues related to invalid column names found in the code.
    """
    
    # Match patterns like df['col'], df["col"], df.col
    bracket_refs = re.findall(r"df\[[\'\"](\w+)[\'\"]\]", code)
    dot_refs = re.findall(r"df\.(\w+)", code)
    # Exclude pandas methods
    pandas_methods = {"head", "tail", "describe", "groupby", "merge", "sort_values",
                      "drop", "reset_index", "copy", "loc", "iloc", "query", "apply",
                      "agg", "nunique", "value_counts", "shape", "columns", "dtypes"}
    
    all_refs = set(bracket_refs + [r for r in dot_refs if r not in pandas_methods])
    invalid = [col for col in all_refs if col not in VALID_COLUMNS]
    return [f"Invalid column name: '{col}' (not in PaySim schema)" for col in invalid]
    
def validate_code(config: dict, 
                  user_query: str, 
                  code: str
        ) -> ValidatorOutput:
    
    """
    Description:
        Send generated code to Llama-3-70B for cross-validation. Llama reviews
        the code against the original query for correctness, column validity,
        and logic errors.
    Args:
        config: Configuration dictionary with model endpoints.
        user_query: The original user query the code should answer.
        code: The generated pandas code to validate.
    Returns:
        ValidatorOutput: Structured output indicating if the code is valid and any issues found.
    """
    
    # --- Fast programmatic column check ---
    column_issues = _check_columns_in_code(code)
    if column_issues:
        logger.warning(f"Column check failed: {column_issues}")
        return ValidatorOutput(valid=False, issues=column_issues)
    
    # --- Llama semantic validation ---
    prompt = f"""
            ## User Query
            {user_query}

            ## Generated Code
            ```python
            {code}
            ```

            ## PaySim Valid Columns
            step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

            Review this code. Does it correctly answer the user query? Are there any bugs? Check column names carefully against the valid columns list.
            """
                
    response = call_llm(config = config,
                        prompt = prompt,
                        model = "llama",
                        system = VALIDATOR_SYSTEM_PROMPT,
                        temperature = 0,
                    )
    
    # Parse JSON from response
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0]
        
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Validator JSON parse failed: {e}. Raw: {response[:200]}")
        return ValidatorOutput(valid=True, issues=[])
    
    # Ensure expected keys exist with defaults
    if "valid" not in parsed:
        parsed["valid"] = True
    if "issues" not in parsed:
        parsed["issues"] = []

    return ValidatorOutput(**parsed)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    config = load_config(config_path)

    # Test 1: Correct code — should pass
    logger.info("--- Test 1: Correct code ---")
    result = validate_code(config = config,
                            user_query="Show all fraudulent transactions",
                            code="fraud = df[df['isFraud'] == 1]\nprint(fraud)"
                )
    logger.info(f"Valid: {result.valid}, Issues: {result.issues}")

    # Test 2: Wrong column name — should fail
    logger.info("\n--- Test 2: Wrong column name ---")
    result = validate_code(config = config,
                            user_query="Show all fraudulent transactions",
                            code="fraud = df[df['is_fraud'] == 1]\nprint(fraud)"
            )
    logger.info(f"Valid: {result.valid}, Issues: {result.issues}")

    # Test 3: Code doesn't answer query — should fail
    logger.info("\n--- Test 3: Code doesn't answer query ---")
    result = validate_code(config = config,
                            user_query="Find mule accounts",
                            code="print(df.describe())"
                )
    logger.info(f"Valid: {result.valid}, Issues: {result.issues}")