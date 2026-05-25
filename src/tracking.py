import time
import json
import mlflow
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class QueryTracker:
    
    """
    Description:
        MLflow experiment tracker for the RAG pipeline.
        Logs per-query metrics (latency, model, fallback stage, query type)
        and aggregate metrics (correctness rate, avg latency, fallback frequency).
    """
    
    def __init__(self, project: str = "rag-financial-analyst", 
                 run_name: str = None,
                 config: dict = None):
        
        mlflow.set_experiment(project)
        self.run = mlflow.start_run(run_name=run_name)
        if config:
            # MLflow params must be strings; flatten top-level keys
            flat_config = {k: str(v) for k, v in config.items()}
            mlflow.log_params(flat_config)
        self._history = []  # type: List[Dict]
        self._step = 0
        logger.info(f"MLflow run initialized: {self.run.info.run_name} (run_id={self.run.info.run_id})")
        
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
            Logs a single query's metrics to MLflow and stores it in the local history.
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
            "latency_route": latency_route,
            "latency_retrieve": latency_retrieve,
            "latency_generate": latency_generate,
            "latency_validate": latency_validate,
            "latency_total": total_latency,
            "confidence": confidence,
            "code_safe": int(code_safe),
            "context_length": context_length,
            "columns_used": columns_used or [],
        }
        
        self._history.append(record)
        
        # Log to MLflow (step-level)
        mlflow.log_metrics(
            {k: v for k, v in record.items() 
             if k not in ("query", "query_type", "model_used", "columns_used")},
            step=self._step
        )
        self._step += 1
        
        logger.info(f"[MLflow] Logged query: type={query_type} | stage={fallback_stage} | "
                     f"total={total_latency:.2f}s | model={model_used}")
        
    def log_summary(self) -> None:
        
        """
        Description:
            Computes and logs aggregate metrics to MLflow, such as average latency, 
            fallback frequencies, correctness rates, and query type distributions.
        """
        
        if not self._history:
            logger.warning("[MLflow] No queries to summarize")
            return
        
        n = len(self._history)
        
        # Fallback frequency
        stage_counts = {}
        for r in self._history:
            s = r["fallback_stage"]
            stage_counts[s] = stage_counts.get(s, 0) + 1
        
        # Avg latencies
        avg_route = sum(r["latency_route"] for r in self._history) / n
        avg_retrieve = sum(r["latency_retrieve"] for r in self._history) / n
        avg_generate = sum(r["latency_generate"] for r in self._history) / n
        avg_validate = sum(r["latency_validate"] for r in self._history) / n
        avg_total = sum(r["latency_total"] for r in self._history) / n
        
        # Code safety rate
        safe_rate = sum(r["code_safe"] for r in self._history) / n
        
        # First-pass success rate (stage 1)
        first_pass_rate = stage_counts.get(1, 0) / n
        
        # Avg confidence
        avg_confidence = sum(r["confidence"] for r in self._history) / n
        
        summary = {
            "summary_total_queries": n,
            "summary_first_pass_rate": first_pass_rate,
            "summary_code_safe_rate": safe_rate,
            "summary_avg_confidence": avg_confidence,
            "summary_avg_latency_route": avg_route,
            "summary_avg_latency_retrieve": avg_retrieve,
            "summary_avg_latency_generate": avg_generate,
            "summary_avg_latency_validate": avg_validate,
            "summary_avg_latency_total": avg_total,
        }
        
        # Fallback distribution
        for stage in range(1, 5):
            summary[f"summary_fallback_stage_{stage}_count"] = stage_counts.get(stage, 0)
            summary[f"summary_fallback_stage_{stage}_pct"] = stage_counts.get(stage, 0) / n
        
        # Query type distribution
        type_counts = {}
        for r in self._history:
            t = r["query_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        for qtype, count in type_counts.items():
            summary[f"summary_query_type_{qtype}_count"] = count
        
        mlflow.log_metrics(summary)
        
        # Log query history as JSON artifact
        mlflow.log_text(
            json.dumps(self._history, indent=2),
            "query_history.json"
        )
        
        logger.info(f"[MLflow] Summary: {n} queries | first_pass={first_pass_rate:.0%} | "
                     f"avg_latency={avg_total:.2f}s | safe_rate={safe_rate:.0%}")
        
    def finish(self):
        
        """Description:
            Finalizes the MLflow run and logs the final summary.
        """
        
        self.log_summary()
        mlflow.end_run()
        logger.info("[MLflow] Run finished")
