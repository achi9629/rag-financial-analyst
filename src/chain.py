import logging
from pathlib import Path
from langchain_community.vectorstores import Chroma

from ingest import load_vectorstore
from models import load_config, call_llm
from router import route_query, RouterOutput

logger = logging.getLogger(__name__)

# To be used only for sanity check of chain.py module. The actual system prompt for code generation is sent to 
# the LLM in the `generate_code` function and can be modified there as needed.
CODEGEN_SYSTEM_PROMPT = """You are a pandas code generator for financial fraud analysis.
You will receive:
1. A user query about transaction data
2. Retrieved domain knowledge (schema, fraud rules, regulatory info, patterns)
3. An execution plan with steps

Generate ONLY executable Python/pandas code that answers the query.

## Rules
- The DataFrame is already loaded as `df` — do NOT load or read any file.
- Use ONLY pandas and numpy. No other imports.
- Print the final result using `print()`.
- Use exact column names: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
- Add a brief comment explaining each step.
- If the query asks for "top N", sort descending and use `.head(N)`.
- For thresholds: use values from the retrieved domain knowledge, not made-up numbers.
- Output ONLY the code block. No markdown fences, no explanation outside code.
"""

def retrieve_context(vectorstore: Chroma,
                     queries: list[str], 
                     k: int = 3
        ) -> str:
    
    '''
    Description:
        Retrieve relevant document chunks from ChromaDB for a list of queries and deduplicate them into a single context string.
    Args:
        - vectorstore: The Chroma vector store instance.
        - queries: A list of retrieval query strings.
        - k: Number of top results per query. Defaults to 3.
    Returns:
        - str: A single string containing the deduplicated retrieved document chunks, separated by "---".
    '''
    
    seen = set()
    chunks = []
    
    for query in queries:
        results = vectorstore.similarity_search_with_score(query, k=k)
        for doc, score in results:
            
            # Deduplicate by content
            content_key = doc.page_content[:100]
            if content_key not in seen:
                seen.add(content_key)
                source = doc.metadata.get("source", "unknown")
                section = doc.metadata.get("section", "")
                header = f"[Source: {source}"
                if section:
                    header += f" | Section: {section}"
                header += f" | Score: {score:.3f}]"
                chunks.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(chunks)

def build_prompt(user_query: str, 
                 context: str, 
                 plan: RouterOutput
        ) -> str:
    
    '''
    Description:
        Build the final prompt for code generation by combining the user query,
        execution plan, and retrieved context.
    Args:
        - user_query: The original natural language query.
        - context: Retrieved domain knowledge chunks formatted as a string.
        - plan: The RouterOutput containing complexity, steps, and retrieval queries.
    Returns:
        - str: The final prompt string to be sent to the LLM for code generation.
    '''
    
    steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan.steps))
    
    prompt = f"""## User Query
            {user_query}

            ## Query Complexity
            {plan.complexity}

            ## Execution Plan
            {steps_text}

            ## Retrieved Domain Knowledge
            {context}

            ## Instructions
            Generate executable pandas code that answers the user query.
            The DataFrame `df` is already loaded with the PaySim dataset.
            Output ONLY the Python code, nothing else."""
    
    return prompt

def generate_code(config: dict, 
                  user_query: str, 
                  vectorstore: Chroma, 
                  plan: RouterOutput = None,
                  k: int = 3
                  ) -> str:
    '''
    Description:
        End-to-end code generation: route the query, retrieve context, and generate pandas code.
    Args:
        - config: Configuration dictionary.
        - user_query: The user's natural language query.
        - vectorstore: The Chroma vector store instance for retrieval.
        - plan: Optional pre-computed RouterOutput. If None, will route the query first.
        - k: Number of retrieval results per query. Defaults to 3.
    Returns:
        - str: The generated Python code as a string.
    '''
    
    # Step 1: Route the query (if plan not provided)
    if plan is None:
        plan = route_query(config, user_query)
    logger.info(f"Complexity: {plan.complexity} | Steps: {len(plan.steps)} | Retrieval queries: {len(plan.retrieval_queries)}")
    
    # Step 2: Retrieve context from ChromaDB
    if plan.complexity == "multi_hop":
        # For multi-hop: retrieve more context per step
        context = retrieve_context(vectorstore, plan.retrieval_queries, k=k+2)
    else:
        context = retrieve_context(vectorstore, plan.retrieval_queries, k=k)
    logger.info(f"Retrieved context length: {len(context)} chars")
    
    # Step 3: Build prompt and generate code
    prompt = build_prompt(user_query, context, plan)
    
    code = call_llm(config=config,
                    prompt=prompt,
                    model="qwen",
                    system=CODEGEN_SYSTEM_PROMPT,
                    temperature=0)
    
    # Strip markdown fences if present
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
        code = code.rsplit("```", 1)[0]
    
    return code

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    config = load_config(str(config_path))
    
    # Load existing vectorstore
    ingestion_cfg = config["ingestion"]
    
    chroma_dir = ingestion_cfg["chroma_dir"]
    device = ingestion_cfg["embedding_device"]
    embedding_model = ingestion_cfg["embedding_model"]
    collection_name = ingestion_cfg["collection_name"]
    
    vectorstore = load_vectorstore(chroma_dir = chroma_dir,
                                   collection_name = collection_name,
                                   embedding_model = embedding_model,
                                   device = device
                    )
    
    test_queries = [
                    # Simple
                    "Show all fraudulent transactions",
                    # Threshold
                    "Flag transactions above 10 lakh",
                    # Multi-hop
                    "Identify mule accounts",
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}")
        code = generate_code(config, query, vectorstore)
        logger.info(f"\nGenerated code:\n{code}")
        logger.info("")