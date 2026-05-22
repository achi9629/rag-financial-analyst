import json
import logging
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

STRUCTURED_CODEGEN_SYSTEM_PROMPT = """You are a pandas code generator for financial fraud analysis.
You will receive:
1. A user query about transaction data
2. Retrieved domain knowledge (schema, fraud rules, regulatory info, patterns)
3. An execution plan with steps

Generate a JSON response with the following structure:

{
    "code": "<executable pandas/python code as a single string>",
    "explanation": "<1-3 sentence explanation of what the code does>",
    "confidence": <float between 0.0 and 1.0>,
    "columns_used": ["column1", "column2"]
}

## Rules for "code" field
- The DataFrame is already loaded as `df` — do NOT load or read any file.
- Use ONLY pandas and numpy. No other imports.
- Print the final result using `print()`.
- Use exact column names: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
- Add a brief comment explaining each step.
- If the query asks for "top N", sort descending and use `.head(N)`.
- For thresholds: use values from the retrieved domain knowledge, not made-up numbers.
- Use \\n for newlines within the code string.

## Rules for other fields
- "explanation": Concise summary of the approach and what the output will show.
- "confidence": 0.9+ if straightforward query, 0.7-0.9 if requires assumptions, <0.7 if uncertain.
- "columns_used": List ONLY PaySim columns actually referenced in the code.

Respond ONLY with valid JSON. No markdown fences, no text outside the JSON.
"""

class CodeOutput(BaseModel):
    
    '''
    Description:
        Structured output from the code generation chain. Contains the generated
        pandas code along with metadata for validation and transparency.
    Fields:
        code: The generated executable pandas/Python code.
        explanation: A brief explanation of what the code does and how it answers the query.
        confidence: A float between 0.0 and 1.0 indicating model confidence in correctness.
        columns_used: List of PaySim column names referenced in the generated code.
    '''
    
    code: str
    explanation: str
    confidence: float
    columns_used: list[str]

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("columns_used")
    @classmethod
    def validate_columns(cls, v):
        valid_columns = {
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest",
            "newbalanceDest", "isFraud", "isFlaggedFraud"
        }
        invalid = [col for col in v if col not in valid_columns]
        if invalid:
            logger.warning(f"Unexpected columns in output: {invalid}")
        return v

    @field_validator("code")
    @classmethod
    def validate_code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("code cannot be empty")
        return v
    
def _extract_columns_from_code(code: str) -> list[str]:
    
    '''
    Description:
        Heuristic extraction of PaySim column names from the generated code string.
    Args:
        - code: The generated code string from which to extract column names.
    Returns:
        - list[str]: A list of column names that are referenced in the code.
    '''
    
    valid_columns = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest",
        "newbalanceDest", "isFraud", "isFlaggedFraud"
    ]
    return [col for col in valid_columns if col in code]
    
def parse_code_output(response: str
        ) -> CodeOutput:
    
    '''
    Description:
        Parse and validate an LLM response string into a CodeOutput instance.
    Args:
        - response: Raw string response from the LLM.
    Returns:
        - CodeOutput: A validated instance containing the generated code and metadata.
    Raises:
        - ValueError: If the response cannot be parsed or validated.
    '''
    
    response = response.strip()
    
    # Handle markdown fences
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0]
    
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}\nResponse: {response[:200]}")
    
    # Handle escaped newlines in code field
    if "code" in parsed and isinstance(parsed["code"], str):
        parsed["code"] = parsed["code"].replace("\\n", "\n")
    
    return CodeOutput(**parsed)

def parse_code_output_with_fallback(response: str, 
                                    raw_code_fallback: bool = True
        ) -> CodeOutput:
    
    '''
    Description:
        Attempt to parse structured JSON output. If that fails and raw_code_fallback is True,
        treat the entire response as raw code and wrap it in a CodeOutput with default metadata.
    Args:
        - response: Raw string response from the LLM.
        - raw_code_fallback: If True, fall back to treating response as raw code on parse failure.
    Returns:
        - CodeOutput: A validated CodeOutput instance.
    '''
    
    try:
        return parse_code_output(response)
    except (ValueError, Exception) as e:
        if not raw_code_fallback:
            raise
        
        logger.warning(f"Structured parse failed ({e}), falling back to raw code extraction")
        
        # Strip markdown fences if present
        code = response.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1]
            code = code.rsplit("```", 1)[0]
        
        return CodeOutput(
            code=code,
            explanation="(Auto-extracted: LLM did not return structured JSON)",
            confidence=0.5,
            columns_used=_extract_columns_from_code(code)
        )
        
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Test 1: Valid structured JSON
    test_response = '''
                {
                    "code": "# Filter fraudulent transactions\\nfraud = df[df['isFraud'] == 1]\\nprint(fraud)",
                    "explanation": "Filters the DataFrame to show only transactions marked as fraudulent.",
                    "confidence": 0.95,
                    "columns_used": ["isFraud"]
                }
                '''
    
    result = parse_code_output(test_response)
    logger.info(f"Test 1 (structured): ✅")
    logger.info(f"  Code:\n{result.code}")
    logger.info(f"  Explanation: {result.explanation}")
    logger.info(f"  Confidence: {result.confidence}")
    logger.info(f"  Columns: {result.columns_used}")
    
    # Test 2: Fallback from raw code
    raw_response = """# Filter fraudulent transactions
                    fraud = df[df['isFraud'] == 1]
                    print(fraud)"""
    
    result2 = parse_code_output_with_fallback(raw_response)
    logger.info(f"\nTest 2 (fallback): ✅")
    logger.info(f"  Code:\n{result2.code}")
    logger.info(f"  Confidence: {result2.confidence}")
    logger.info(f"  Columns: {result2.columns_used}")
    
    # Test 3: Invalid confidence
    try:
        CodeOutput(code="print('hi')", explanation="test", confidence=1.5, columns_used=[])
        logger.info("\nTest 3 (invalid confidence): ❌ should have raised")
    except Exception as e:
        logger.info(f"\nTest 3 (invalid confidence): ✅ Caught: {e}")