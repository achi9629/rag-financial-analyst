import json
import logging
from pathlib import Path
from models import load_config, call_llm
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query complexity classifier for a financial fraud analysis system.
Given a user query about transaction data, classify it and output a JSON plan.

## Classification Rules

- **simple**: Direct column lookup, basic aggregation, or filtering on a single condition.
  Examples: "Show all fraudulent transactions", "How many TRANSFER type transactions?", "Average transaction amount"

- **threshold**: Requires applying a numeric threshold, rule, or condition from domain knowledge.
  Examples: "Flag transactions above ₹10 lakh", "Find accounts with balance mismatch", "Large cash-outs that zeroed sender balance"

- **multi_hop**: Requires multiple computation steps, combining rules, or derived features.
  Examples: "Identify mule accounts", "Detect structuring", "Top 10 most suspicious accounts by risk score", "Build composite risk score"

## Output Format

Respond ONLY with valid JSON (no markdown, no explanation):
{
    "complexity": "simple" | "threshold" | "multi_hop",
    "steps": ["step 1 description", "step 2 description"],
    "retrieval_queries": ["query for vector store retrieval 1", "query for vector store retrieval 2"]
}

Rules for the fields:
- "steps": Break the query into execution steps. For "simple", this is 1 step. For "multi_hop", 2-5 steps.
- "retrieval_queries": What to search in the knowledge base to get relevant schema/rules/patterns. 1-3 queries.
"""

class RouterOutput(BaseModel):
    
    '''
    Description:
        A structured output from the query router that classifies the complexity of a user query and outlines the execution plan.
    Fields:
        - complexity: A string indicating the complexity category of the query. Must be one of "simple", "threshold", or "multi_hop".
        - steps: A list of strings, each describing a step in the execution plan to answer the user query. For "simple" queries, this should contain exactly one step. For "multi_hop" queries, this should contain 2`-5 steps.
        - retrieval_queries: A list of strings, each representing a query that should be sent to the vector store to retrieve relevant information for executing the user query. This should contain 1-3 queries.
    '''
    
    complexity: str
    steps: list[str]
    retrieval_queries: list[str]

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v):
        allowed = {"simple", "threshold", "multi_hop"}
        if v not in allowed:
            raise ValueError(f"complexity must be one of {allowed}, got '{v}'")
        return v

def route_query(config: dict, user_query: str) -> RouterOutput:
    
    '''
    Description:
        Routes a user query by classifying its complexity and generating an execution plan.
    Args:
        - config: A dictionary containing configuration parameters for the LLM call.
        - user_query: A string representing the user's natural language query about transaction data.
    Returns:
        - An instance of RouterOutput containing the complexity classification, execution steps, and retrieval queries.
    '''
    
    response = call_llm(config = config,
                        prompt = user_query,
                        model = "llama",
                        system = ROUTER_SYSTEM_PROMPT,
                        temperature = 0,
                )
    
    # Parse JSON from response (handle potential markdown wrapping)
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1]  # remove ```json line
        response = response.rsplit("```", 1)[0]  # remove closing ```
    
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse router response: {response}: {e}")
        # Fallback: treat as simple query
        parsed = {
            "complexity": "simple",
            "steps": [user_query],
            "retrieval_queries": [user_query],
        }
    
    return RouterOutput(**parsed)

if __name__ == "__main__":
    
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")
    
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    config = load_config(str(config_path))
    
    test_queries = [
        # Simple
        ("Show all fraudulent transactions", "simple"),
        ("How many TRANSFER type transactions?", "simple"),
        ("Average transaction amount", "simple"),
        # Threshold
        ("Flag transactions above 10 lakh", "threshold"),
        ("Find accounts with balance mismatch", "threshold"),
        ("Large cash-outs that zeroed sender balance", "threshold"),
        # Multi-hop
        ("Identify mule accounts", "multi_hop"),
        ("Top 10 most suspicious accounts by risk score", "multi_hop"),
        ("Detect structuring — multiple transactions just below 10 lakh", "multi_hop"),
    ]
    
    correct = 0
    for query, expected in test_queries:
        result = route_query(config, query)
        match = "✅" if result.complexity == expected else "❌"
        if result.complexity == expected:
            correct += 1
        logger.info(f"{match} Query: '{query}'")
        logger.info(f"   Expected: {expected} | Got: {result.complexity}")
        logger.info(f"   Steps: {result.steps}")
        logger.info(f"   Retrieval: {result.retrieval_queries}")
        logger.info("")
    
    logger.info(f"Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.0f}%)")