import time
import logging
from pathlib import Path
from router import route_query
from tracking import QueryTracker
from validator import validate_code
from ingest import load_vectorstore
from models import load_config, call_llm
from guardrails import check_code_safety
from chain import retrieve_context, build_prompt
from validation import STRUCTURED_CODEGEN_SYSTEM_PROMPT, CodeOutput, parse_code_output_with_fallback

logger = logging.getLogger(__name__)


class PipelineResult:
    
    """
    Description:
        Extended result from the pipeline including execution metadata.
    Fields:
        code_output: The validated CodeOutput from generation.
        fallback_stage: Which try succeeded (1-4). 4 = graceful degradation.
        context: Retrieved domain knowledge (for fallback stage 4).
        plan_steps: Router plan steps (for fallback stage 4).
        model_used: Which model produced the final output (qwen/llama/blocked/none).
        query_type: Complexity from router (simple/threshold/multi_hop).
        latency_route: Routing latency in seconds.
        latency_retrieve: Retrieval latency in seconds.
        latency_generate: Total generation latency in seconds.
        latency_validate: Total validation latency in seconds.
        total_latency: End-to-end pipeline latency in seconds.
    """
    
    def __init__(self, code_output: CodeOutput, fallback_stage: int,
                 context: str = "", plan_steps: list = None,
                 model_used: str = "", query_type: str = "",
                 latency_route: float = 0.0, latency_retrieve: float = 0.0,
                 latency_generate: float = 0.0, latency_validate: float = 0.0,
                 total_latency: float = 0.0):
        self.code_output = code_output
        self.fallback_stage = fallback_stage
        self.context = context
        self.plan_steps = plan_steps or []
        self.model_used = model_used
        self.query_type = query_type
        self.latency_route = latency_route
        self.latency_retrieve = latency_retrieve
        self.latency_generate = latency_generate
        self.latency_validate = latency_validate
        self.total_latency = total_latency


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
        self.tracker = None
        logger.info("Pipeline initialized: config loaded, vectorstore connected")
    
    def enable_tracking(self,
                         project="rag-financial-analyst",
                         run_name=None
            ) -> None:
        self.tracker = QueryTracker(project=project, run_name=run_name, config=self.config)
    
    def _validate_code(self, code: str,
                        user_query: str
            ) -> tuple:
        
        """
        Description:
            Validate generated code using guardrails and Llama cross-validation.
        Args:
            - code: The generated code to validate.
            - user_query: The original user query (for context in validation).
        Returns:
            - Tuple[bool, list, str]: (is_valid, issues, fail_type).
              fail_type is "guardrail", "validation", or "passed".
        """
        
        # Guardrails (fast, deterministic)
        is_safe, safety_issues = check_code_safety(code)
        if not is_safe:
            return False, safety_issues, "guardrail"
        
        # Llama cross-validation (semantic)
        val_result = validate_code(self.config, user_query, code)
        if not val_result.valid:
            return False, val_result.issues, "validation"
        
        return True, [], "passed"
    
    def _track(self, user_query, plan, fallback_stage, model_used, 
            code_output, t_route, t_retrieve, t_generate, t_validate, t_start, context):
        """Log query metrics to W&B if tracking is enabled."""
        if self.tracker:
            self.tracker.track_query(
                user_query = user_query,
                query_type = plan.complexity,
                fallback_stage = fallback_stage,
                model_used = model_used,
                latency_route = t_route,
                latency_retrieve = t_retrieve,
                latency_generate = t_generate,
                latency_validate = t_validate,
                total_latency = time.time() - t_start,
                confidence = code_output.confidence,
                code_safe = fallback_stage < 4,
                context_length = len(context),
                columns_used = code_output.columns_used,
            )
    
    def run_query(self,
                   user_query: str,
                   k: int = 3
        ) -> PipelineResult:
        
        '''
        Description:
            Execute the full RAG pipeline with cascading fallback:
            Try 1: Qwen generates -> guardrails + validator -> return
            Try 2: Qwen regenerates with feedback -> guardrails -> return (only if Try 1 was a validation failure)
            Try 3: Llama generates directly -> syntax check -> return
            Try 4: Return raw docs + plan (graceful degradation)
            
            Guardrail failures are a hard stop — no retry (prevents LLM from circumventing safety checks).
            Only validation failures (wrong columns, logic bugs) trigger the cascade.
        Args:
            - user_query: Natural language query about transaction data.
            - k: Number of retrieval results per query. Defaults to 3.
        Returns:
            - PipelineResult: Contains CodeOutput + fallback_stage metadata.
        '''
        
        t_start = time.time()
        # Step 1: Route
        t0 = time.time()
        plan = route_query(self.config, user_query)
        t_route = time.time() - t0
        logger.info(f"[Route] complexity={plan.complexity} |steps={len(plan.steps)} | queries={len(plan.retrieval_queries)}")
         
        # Step 2: Retrieve
        retrieval_k = k + 2 if plan.complexity == "multi_hop" else k
        t1 = time.time()
        context = retrieve_context(vectorstore = self.vectorstore,
                                    queries = plan.retrieval_queries,
                                    k = retrieval_k
                    )
        t_retrieve = time.time() - t1
        logger.info(f"[Retrieve] context_length={len(context)} chars")
        
        # Step 3: Generate
        prompt = build_prompt(user_query = user_query,
                               context = context,
                               plan = plan
                    )
        
        # --- Try 1: Qwen generates -> full validation ---
        logger.info("[Try 1] Qwen generation + full validation")
        t2 = time.time()
        response = call_llm(config = self.config,
                            prompt = prompt,
                            model = "qwen",
                            system = STRUCTURED_CODEGEN_SYSTEM_PROMPT,
                            temperature = 0)
        
        result = parse_code_output_with_fallback(response)
        t_generate = time.time() - t2
        
        t3 = time.time()
        passed, issues, fail_type = self._validate_code(result.code, user_query)
        t_validate = time.time() - t3
        
        if passed:
            logger.info("[Try 1] PASSED -- returning result")
            total_latency = time.time() - t_start
            self._track(user_query, plan, 1, "qwen", result,
                        t_route, t_retrieve, t_generate, t_validate, t_start, context)
            return PipelineResult(code_output = result, 
                                   fallback_stage = 1,
                                  context = context, 
                                   plan_steps = plan.steps,
                                  model_used = "qwen",
                                  query_type = plan.complexity,
                                  latency_route = t_route,
                                  latency_retrieve = t_retrieve,
                                  latency_generate = t_generate,
                                  latency_validate = t_validate,
                                  total_latency = total_latency
                        )
        
        # Guardrail failure = hard stop, no retry
        if fail_type == "guardrail":
            logger.warning(f"[BLOCKED] Guardrail violation -- not retrying: {issues}")
            blocked_output = CodeOutput(
                code="# Query blocked by safety guardrails.",
                explanation=f"Generated code violated safety rules: {'; '.join(issues)}",
                confidence=0.0,
                columns_used=[]
            )
            total_latency = time.time() - t_start
            self._track(user_query, plan, 4, "blocked", blocked_output,
                        t_route, t_retrieve, t_generate, t_validate, t_start, context)
            return PipelineResult(code_output = blocked_output, 
                                   fallback_stage = 4,
                                  context = context, 
                                   plan_steps = plan.steps,
                                  model_used = "blocked",
                                  query_type = plan.complexity,
                                  latency_route = t_route,
                                  latency_retrieve = t_retrieve,
                                  latency_generate = t_generate,
                                  latency_validate = t_validate,
                                  total_latency = total_latency
                        )
        
        
        # --- Try 2: Qwen regenerates with Llama feedback (validation failures only) ---
        logger.info(f"[Try 2] Qwen retry with feedback: {issues}")
        feedback_prompt = f"""{prompt}

## Previous Attempt Failed
The previous code had these issues: {issues}
Please fix these issues and regenerate the code."""
        
        t4 = time.time()
        response2 = call_llm(config = self.config,
                             prompt = feedback_prompt,
                             model = "qwen",
                             system = STRUCTURED_CODEGEN_SYSTEM_PROMPT,
                             temperature = 0.1)
        
        result2 = parse_code_output_with_fallback(response2)
        t_generate += time.time() - t4
        
        t5 = time.time()
        is_safe, safety_issues = check_code_safety(result2.code)
        t_validate += time.time() - t5
        
        if is_safe:
            logger.info("[Try 2] PASSED -- returning result")
            total_latency = time.time() - t_start
            self._track(user_query, plan, 2, "qwen", result2,
                        t_route, t_retrieve, t_generate, t_validate, t_start, context)
            return PipelineResult(code_output = result2, 
                                   fallback_stage = 2,
                                  context = context, 
                                   plan_steps = plan.steps,
                                  model_used = "qwen",
                                  query_type = plan.complexity,
                                  latency_route = t_route,
                                  latency_retrieve = t_retrieve,
                                  latency_generate = t_generate,
                                  latency_validate = t_validate,
                                  total_latency = total_latency
                        )
        
        
        # --- Try 3: Llama generates code directly ---
        logger.info(f"[Try 3] Llama direct generation (safety issues: {safety_issues})")
        t6 = time.time()
        response3 = call_llm(config = self.config,
                             prompt = prompt,
                             model = "llama",
                             system = STRUCTURED_CODEGEN_SYSTEM_PROMPT,
                             temperature = 0)
        
        result3 = parse_code_output_with_fallback(response3)
        t_generate += time.time() - t6
        
        t7 = time.time()
        is_safe3, _ = check_code_safety(result3.code)
        t_validate += time.time() - t7
        
        if is_safe3:
            logger.info("[Try 3] PASSED -- returning Llama result")
            total_latency = time.time() - t_start
            self._track(user_query, plan, 3, "llama", result3,
                        t_route, t_retrieve, t_generate, t_validate, t_start, context)
            return PipelineResult(code_output = result3, 
                                   fallback_stage = 3,
                                  context = context, 
                                   plan_steps = plan.steps,
                                  model_used = "llama",
                                  query_type = plan.complexity,
                                  latency_route = t_route,
                                  latency_retrieve = t_retrieve,
                                  latency_generate = t_generate,
                                  latency_validate = t_validate,
                                  total_latency = total_latency
                        )
        
        # --- Try 4: Graceful degradation ---
        logger.info("[Try 4] All generation attempts failed -- returning raw context")
        fallback_output = CodeOutput(
            code="# All generation attempts failed. See context below.",
            explanation="Could not generate valid code. Retrieved context and plan provided for manual analysis.",
            confidence=0.0,
            columns_used=[]
        )
        total_latency = time.time() - t_start
        self._track(user_query, plan, 4, "none", fallback_output,
                    t_route, t_retrieve, t_generate, t_validate, t_start, context)
        return PipelineResult(code_output = fallback_output, 
                               fallback_stage = 4,
                              context = context, 
                               plan_steps = plan.steps,
                              model_used = "none",
                              query_type = plan.complexity,
                              latency_route = t_route,
                              latency_retrieve = t_retrieve,
                              latency_generate = t_generate,
                              latency_validate = t_validate,
                              total_latency = total_latency
                    )


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    pipe = Pipeline()
    pipe.enable_tracking(run_name="day5-demo")
    
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
        
        logger.info(f"Fallback stage: {result.fallback_stage}")
        logger.info(f"Code:\n{result.code_output.code}\n")
        logger.info(f"  Explanation: {result.code_output.explanation}")
        logger.info(f"  Confidence: {result.code_output.confidence}")
        logger.info(f"  Columns used: {result.code_output.columns_used}")
        logger.info("")
    
    pipe.tracker.finish()
