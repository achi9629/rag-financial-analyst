import json
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from models import call_llm
from router import route_query
from guardrails import check_code_safety
from executor import execute_code_safe
from chain import retrieve_context, build_prompt
from validation import parse_code_output_with_fallback, STRUCTURED_CODEGEN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class Action(Enum):
    REGENERATE = "regenerate"
    RETRIEVE_MORE = "retrieve_more"
    FIX_CODE = "fix_code"
    STOP_BLOCKED = "stop_blocked"
    STOP_SUCCESS = "stop_success"
    STOP_MAX_RETRIES = "stop_max"


@dataclass
class AgentState:
    
    """
    Description:
        Mutable state for the agentic self-correction loop.
    Fields:
        user_query: Original natural language query.
        code: Current generated code.
        error: Last execution error message.
        attempt: Current attempt number (0-indexed).
        context: Retrieved domain knowledge (grows with dynamic retrieval).
        action_history: List of (action, diagnosis) tuples for observability.
    """
    
    user_query: str
    code: str = ""
    error: str = ""
    attempt: int = 0
    context: str = ""
    action_history: list = field(default_factory=list)
    _retrieval_query: str = ""
    _fix: str = ""
    _diagnosis: str = ""


@dataclass
class AgentResult:
    
    """
    Description:
        Result from the agentic pipeline including full execution trace.
    Fields:
        code: Final generated code.
        explanation: LLM explanation of the code.
        exec_output: Output from sandbox execution (empty if failed).
        confidence: LLM confidence score.
        columns_used: PaySim columns referenced.
        attempts: Total number of attempts (1 = first-pass success).
        action_history: List of (action, diagnosis) strings for observability.
        final_action: Terminal action (stop_success / stop_blocked / stop_max).
        query_type: Complexity from router (simple / threshold / multi_hop).
        error: Last error message if failed, None if succeeded.
        total_latency: End-to-end time in seconds.
    """
    
    code: str
    explanation: str
    exec_output: str
    confidence: float
    columns_used: list
    attempts: int
    action_history: list
    final_action: str
    query_type: str
    error: Optional[str]
    total_latency: float


DIAGNOSIS_PROMPT = """You are a debugging agent for pandas code that analyzes PaySim financial transaction data.

The code below was generated to answer a user query but failed during execution.

User Query: {query}

Generated Code:
```python
{code}
```

Execution Error:
{error}

Diagnose the root cause and choose exactly ONE action:

1. "regenerate" — the overall approach is wrong, need completely fresh code with the error as feedback
2. "retrieve_more" — missing domain knowledge (unknown column name, unknown threshold, unknown formula). Provide a search query to retrieve the missing information.
3. "fix_code" — minor bug (typo, wrong operator, wrong column name, off-by-one). Provide the corrected code.

Respond with ONLY this JSON (no markdown, no explanation):
{{"action": "regenerate", "diagnosis": "one-line reason"}}
or
{{"action": "retrieve_more", "diagnosis": "one-line reason", "retrieval_query": "search query for missing knowledge"}}
or
{{"action": "fix_code", "diagnosis": "one-line reason", "fix": "corrected python code here"}}"""


FIX_CODE_PROMPT = """Fix this pandas code. The DataFrame `df` has the PaySim schema:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Original query: {query}

Code with bug:
```python
{code}
```

Error: {error}
Diagnosis: {diagnosis}

Respond with ONLY the corrected Python code. No markdown fences, no explanation."""


class AgentPipeline:
    
    """
    Description:
        Agentic self-correction loop that wraps the existing Pipeline.
        On execution failure, it diagnoses the error via Llama, then decides:
        - regenerate (fresh code with error feedback)
        - retrieve_more (dynamic ChromaDB retrieval for missing knowledge)
        - fix_code (patch the specific bug)
        Guardrail violations are a hard stop (no retry).
        Cap at max_retries with increasing temperature for diversity.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = pipeline.config
        self.vectorstore = pipeline.vectorstore
    
    def run(self, user_query: str, df, max_retries: int = 3, k: int = 3) -> AgentResult:
        
        """
        Description:
            Execute the agentic loop: generate → execute → observe → diagnose → act → repeat.
        Args:
            user_query: Natural language query about transaction data.
            df: The PaySim DataFrame for sandbox execution.
            max_retries: Maximum correction attempts (default 3).
            k: Number of retrieval results per query.
        Returns:
            AgentResult with code, execution output, action trace, and metadata.
        """
        
        t_start = time.time()
        state = AgentState(user_query=user_query)
        temperatures = [0.0, 0.2, 0.4]
        
        # Step 1: Route + Retrieve (done once, context grows with dynamic retrieval)
        plan = route_query(self.config, user_query)
        retrieval_k = k + 2 if plan.complexity == "multi_hop" else k
        state.context = retrieve_context(
            vectorstore=self.vectorstore,
            queries=plan.retrieval_queries,
            k=retrieval_k
        )
        logger.info(f"[Agent] Route: {plan.complexity} | Initial context: {len(state.context)} chars")
        
        last_result = None  # track last CodeOutput for metadata
        
        while state.attempt <= max_retries:
            temp = temperatures[min(state.attempt, len(temperatures) - 1)]
            
            # --- Generate or fix code ---
            if state.attempt == 0 or (state.action_history and 
                    state.action_history[-1][0] in (Action.REGENERATE, Action.RETRIEVE_MORE)):
                prompt = build_prompt(user_query=user_query, context=state.context, plan=plan)
                if state.error:
                    prompt += f"\n\n## Previous Attempt Failed\nError: {state.error}\nFix the approach and regenerate."
                
                response = call_llm(
                    config=self.config, prompt=prompt, model="qwen",
                    system=STRUCTURED_CODEGEN_SYSTEM_PROMPT, temperature=temp
                )
                last_result = parse_code_output_with_fallback(response)
                state.code = last_result.code
            
            # --- Guardrail check (hard stop) ---
            is_safe, safety_issues = check_code_safety(state.code)
            if not is_safe:
                logger.warning(f"[Agent] Attempt {state.attempt}: BLOCKED — {safety_issues}")
                state.action_history.append((Action.STOP_BLOCKED, f"Guardrail: {safety_issues}"))
                return self._build_result(state, plan, Action.STOP_BLOCKED, t_start, last_result)
            
            # --- Execute in sandbox ---
            exec_result = execute_code_safe(state.code, df, timeout=30)
            
            if exec_result.success:
                logger.info(f"[Agent] Attempt {state.attempt}: SUCCESS after {state.attempt + 1} attempt(s)")
                state.action_history.append((Action.STOP_SUCCESS, "Execution succeeded"))
                return self._build_result(state, plan, Action.STOP_SUCCESS, t_start, 
                                          last_result, exec_output=exec_result.output)
            
            # --- Failed — diagnose and decide ---
            state.error = exec_result.error
            logger.info(f"[Agent] Attempt {state.attempt}: FAILED — {state.error[:100]}")
            state.attempt += 1
            
            if state.attempt > max_retries:
                state.action_history.append((Action.STOP_MAX_RETRIES, f"Max retries reached: {state.error}"))
                logger.warning(f"[Agent] Max retries ({max_retries}) exhausted")
                return self._build_result(state, plan, Action.STOP_MAX_RETRIES, t_start, last_result)
            
            # --- Diagnose via Llama ---
            action = self._diagnose(state)
            state.action_history.append((action, state._diagnosis))
            logger.info(f"[Agent] Attempt {state.attempt}: action={action.value} | diagnosis={state._diagnosis}")
            
            # --- Act ---
            if action == Action.RETRIEVE_MORE:
                extra_context = self._dynamic_retrieve(state)
                state.context += "\n\n## Additional Retrieved Context\n" + extra_context
                logger.info(f"[Agent] Retrieved {len(extra_context)} chars of additional context")
            
            elif action == Action.FIX_CODE:
                state.code = self._fix_code(state)
            
            # REGENERATE: loop continues, new code generated at top of loop
        
        # Should not reach here, but safety fallback
        return self._build_result(state, plan, Action.STOP_MAX_RETRIES, t_start, last_result)
    
    def _diagnose(self, state: AgentState) -> Action:
        """Ask Llama to diagnose the error and choose an action."""
        prompt = DIAGNOSIS_PROMPT.format(
            query=state.user_query, code=state.code, error=state.error
        )
        response = call_llm(config=self.config, prompt=prompt, model="llama", temperature=0)
        
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            data = json.loads(response[start:end])
            action_str = data.get("action", "regenerate")
            state._diagnosis = data.get("diagnosis", "unknown")
            
            if action_str == "retrieve_more" and data.get("retrieval_query"):
                state._retrieval_query = data["retrieval_query"]
                return Action.RETRIEVE_MORE
            elif action_str == "fix_code" and data.get("fix"):
                state._fix = data["fix"]
                return Action.FIX_CODE
            else:
                return Action.REGENERATE
                
        except (ValueError, json.JSONDecodeError):
            state._diagnosis = "Could not parse diagnosis — defaulting to regenerate"
            return Action.REGENERATE
    
    def _dynamic_retrieve(self, state: AgentState) -> str:
        """Retrieve additional context based on the diagnosed knowledge gap."""
        query = state._retrieval_query or state.user_query
        return retrieve_context(vectorstore=self.vectorstore, queries=[query], k=3)
    
    def _fix_code(self, state: AgentState) -> str:
        """Fix code — use Llama's fix if available, otherwise ask Qwen."""
        if state._fix:
            fixed = state._fix
            state._fix = ""
            return fixed
        
        prompt = FIX_CODE_PROMPT.format(
            query=state.user_query, code=state.code,
            error=state.error, diagnosis=state._diagnosis
        )
        response = call_llm(config=self.config, prompt=prompt, model="qwen", temperature=0)
        result = parse_code_output_with_fallback(response)
        return result.code
    
    def _build_result(self, state, plan, final_action, t_start, 
                      last_result=None, exec_output="") -> AgentResult:
        """Build AgentResult with full trace."""
        return AgentResult(
            code=state.code,
            explanation=last_result.explanation if last_result else "",
            exec_output=exec_output,
            confidence=last_result.confidence if last_result else 0.0,
            columns_used=last_result.columns_used if last_result else [],
            attempts=state.attempt + (0 if final_action == Action.STOP_MAX_RETRIES else 1),
            action_history=[f"{a.value}: {d}" for a, d in state.action_history],
            final_action=final_action.value,
            query_type=plan.complexity,
            error=state.error if final_action != Action.STOP_SUCCESS else None,
            total_latency=time.time() - t_start,
        )


if __name__ == "__main__":
    
    import pandas as pd
    from pathlib import Path
    from pipeline import Pipeline
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    pipe = Pipeline()
    agent = AgentPipeline(pipe)
    
    DATA_PATH = str(Path(__file__).parent.parent / "assets" / "datasets" / "PaySim" / "PS_20174392719_1491204439457_log.csv")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset loaded: {len(df)} rows")
    
    test_queries = [
        "Show all fraudulent transactions",
        "Flag transactions above 10 lakh",
        "Identify mule accounts",
        "Top 10 most suspicious accounts by risk score",
        "Detect structuring — multiple txns just below 10 lakh",
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}")
        
        result = agent.run(query, df)
        
        logger.info(f"Final: {result.final_action} | Attempts: {result.attempts} | Latency: {result.total_latency:.2f}s")
        logger.info(f"Code:\n{result.code}")
        if result.error:
            logger.info(f"Error: {result.error}")
        else:
            logger.info(f"Output: {result.exec_output[:200]}")
        logger.info(f"Trace: {result.action_history}")
        logger.info("")
