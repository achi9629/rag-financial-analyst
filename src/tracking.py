import time
import wandb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class QueryTracker:
    
    """
    Description:
        W&B experiment tracker for the RAG pipeline.
        Logs per-query metrics (latency, model, fallback stage, query type)
        and aggregate metrics (correctness rate, avg latency, fallback frequency).
    """
    
    def __init__(self, project: str = "rag-financial-analyst", 
                 run_name: str = None,
                 config: dict = None):
        
        self.run = wandb.init(
                            project=project,
                            name=run_name,
                            config=config or {},
                            reinit=True
                        )
        self._history = []  # type: List[Dict]
        logger.info(f"W&B run initialized: {self.run.name} ({self.run.url})")
        
    def track_query(self, 
                    user_query: str,
                    query_type: str,
                    fallback_stage: int,
                    model_used: str,
                    latency_route: float,
                    latency_retrieve: float,
                    latency_generate: float,
                    latency_validate: float,
                    total_latency: float,
                    confidence: float,
                    code_safe: bool,
                    context_length: int,
                    columns_used: list = None
            ) -> None:
        
        """
        Description:
            Logs a single query's metrics to W&B and stores it in the local history.
        Args:
            user_query (str): The original user query.
            query_type (str): Type of query (e.g., "financial_analysis", "data_retrieval").
            fallback_stage (int): The stage at which fallback occurred (0 if no fallback).
            model_used (str): The model used to generate the response.
            latency_route (float): Time taken for routing/decision-making.
            latency_retrieve (float): Time taken for data retrieval.
            latency_generate (float): Time taken for response generation.
            latency_validate (float): Time taken for validation/safety checks.
            total_latency (float): Total time taken for the entire process.
            confidence (float): Model's confidence score for the generated response.
            code_safe (bool): Whether the generated code passed safety checks.
            context_length (int): Number of tokens in the context used for generation.
            columns_used (list, optional): List of database columns used in retrieval.
        """
        
        record = {
            "query": user_query,
            "query_type": query_type,
            "fallback_stage": fallback_stage,
            "model_used": model_used,
            "latency/route": latency_route,
            "latency/retrieve": latency_retrieve,
            "latency/generate": latency_generate,
            "latency/validate": latency_validate,
            "latency/total": total_latency,
            "confidence": confidence,
            "code_safe": int(code_safe),
            "context_length": context_length,
            "columns_used": columns_used or [],
        }
        
        self._history.append(record)
        
        # Log to W&B (step-level)
        wandb.log({k: v for k, v in record.items() 
                   if k not in ("query", "columns_used")})
        
        logger.info(f"[W&B] Logged query: type={query_type} | stage={fallback_stage} | "
                     f"total={total_latency:.2f}s | model={model_used}")
        
    def log_summary(self) -> None:
        
        """
        Description:
            Computes and logs aggregate metrics to W&B, such as average latency, 
            fallback frequencies, correctness rates, and query type distributions.
        """
        
        if not self._history:
            logger.warning("[W&B] No queries to summarize")
            return
        
        n = len(self._history)
        
        # Fallback frequency
        stage_counts = {}
        for r in self._history:
            s = r["fallback_stage"]
            stage_counts[s] = stage_counts.get(s, 0) + 1
        
        # Avg latencies
        avg_route = sum(r["latency/route"] for r in self._history) / n
        avg_retrieve = sum(r["latency/retrieve"] for r in self._history) / n
        avg_generate = sum(r["latency/generate"] for r in self._history) / n
        avg_validate = sum(r["latency/validate"] for r in self._history) / n
        avg_total = sum(r["latency/total"] for r in self._history) / n
        
        # Code safety rate
        safe_rate = sum(r["code_safe"] for r in self._history) / n
        
        # First-pass success rate (stage 1)
        first_pass_rate = stage_counts.get(1, 0) / n
        
        # Avg confidence
        avg_confidence = sum(r["confidence"] for r in self._history) / n
        
        summary = {
            "summary/total_queries": n,
            "summary/first_pass_rate": first_pass_rate,
            "summary/code_safe_rate": safe_rate,
            "summary/avg_confidence": avg_confidence,
            "summary/avg_latency_route": avg_route,
            "summary/avg_latency_retrieve": avg_retrieve,
            "summary/avg_latency_generate": avg_generate,
            "summary/avg_latency_validate": avg_validate,
            "summary/avg_latency_total": avg_total,
        }
        
        # Fallback distribution
        for stage in range(1, 5):
            summary[f"summary/fallback_stage_{stage}_count"] = stage_counts.get(stage, 0)
            summary[f"summary/fallback_stage_{stage}_pct"] = stage_counts.get(stage, 0) / n
        
        # Query type distribution
        type_counts = {}
        for r in self._history:
            t = r["query_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        for qtype, count in type_counts.items():
            summary[f"summary/query_type_{qtype}_count"] = count
        
        # Log W&B table with all queries
        table = wandb.Table(
            columns=["query", "query_type", "fallback_stage", "model_used",
                      "latency_total", "confidence", "code_safe"],
            data=[[r["query"], r["query_type"], r["fallback_stage"], 
                   r["model_used"], round(r["latency/total"], 2), 
                   r["confidence"], r["code_safe"]] 
                  for r in self._history]
        )
        summary["summary/query_table"] = table
        
        wandb.log(summary)
        
        logger.info(f"[W&B] Summary: {n} queries | first_pass={first_pass_rate:.0%} | "
                     f"avg_latency={avg_total:.2f}s | safe_rate={safe_rate:.0%}")
        
    def finish(self):
        
        """Description:
            Finalizes the W&B run and logs the final summary.
        """
        
        self.log_summary()
        self.run.finish()
        logger.info("[W&B] Run finished")