import logging
from pathlib import Path

from router import route_query
from ingest import load_vectorstore
from models import load_config, call_llm
from chain import retrieve_context, build_prompt
from validation import STRUCTURED_CODEGEN_SYSTEM_PROMPT, CodeOutput, parse_code_output_with_fallback

logger = logging.getLogger(__name__)

class Pipeline:
    
    def __init__(self, config_path: str = None) -> None:
        
        '''
        Description:
            Initialize the RAG pipeline by loading configuration and connecting to the vectorstore.
        Args:
            - config_path: Optional path to the YAML configuration file. If None, defaults to "configs/config.yaml".
        '''
        
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "configs" / "config.yaml")
        
        self.config = load_config(config_path)
        
        ingestion_cfg = self.config["ingestion"]
        self.vectorstore = load_vectorstore(chroma_dir = ingestion_cfg["chroma_dir"],
                                            collection_name = ingestion_cfg["collection_name"],
                                            embedding_model = ingestion_cfg["embedding_model"],
                                            device = ingestion_cfg["embedding_device"]
                                )
        logger.info("Pipeline initialized: config loaded, vectorstore connected")
        
    def run_query(self, 
                  user_query: str, 
                  k: int = 3
        ) -> CodeOutput:
        
        '''
        Description:
            Execute the full RAG pipeline: route → retrieve → generate → validate.
        Args:
            - user_query: Natural language query about transaction data.
            - k: Number of retrieval results per query. Defaults to 3.
        Returns:
            - CodeOutput: Validated structured output with code, explanation, confidence, columns_used.
        '''
        
        # Step 1: Route
        plan = route_query(self.config, user_query)
        logger.info(f"[Route] complexity={plan.complexity} |steps={len(plan.steps)} | queries={len(plan.retrieval_queries)}")
         
        # Step 2: Retrieve
        retrieval_k = k + 2 if plan.complexity == "multi_hop" else k
        context = retrieve_context(vectorstore = self.vectorstore, 
                                   queries = plan.retrieval_queries, 
                                   k = retrieval_k
                    )
        logger.info(f"[Retrieve] context_length={len(context)} chars")
        
        # Step 3: Generate
        prompt = build_prompt(user_query = user_query, 
                              context = context, 
                              plan = plan
                    )
        response = call_llm(config = self.config,
                            prompt = prompt,
                            model = "qwen",
                            system = STRUCTURED_CODEGEN_SYSTEM_PROMPT,
                            temperature = 0)
        logger.info(f"[Generate] response_length={len(response)} chars")
        
        # Step 4: Parse + Validate
        result = parse_code_output_with_fallback(response)
        logger.info(f"[Validate] confidence={result.confidence} | columns={result.columns_used}")
        
        return result
    
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    pipe = Pipeline()
    
    test_queries = [
        # Simple (1)
        ("Show all fraudulent transactions", "simple"),
        # Threshold (2)
        ("Flag transactions above 10 lakh", "threshold"),
        ("Large cash-outs that zeroed sender balance", "threshold"),
        # Multi-hop (2)
        ("Identify mule accounts", "multi_hop"),
        ("Top 10 most suspicious accounts by risk score", "multi_hop"),
    ]
    
    for query, expected_type in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"Expected type: {expected_type}")
        logger.info(f"{'='*60}")
        
        result = pipe.run_query(query)
        
        logger.info(f"Code:\n{result.code}\n")
        logger.info(f"  Explanation: {result.explanation}")
        logger.info(f"  Confidence: {result.confidence}")
        logger.info(f"  Columns used: {result.columns_used}")
        logger.info("")