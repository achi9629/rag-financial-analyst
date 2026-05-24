import time
import logging
import pandas as pd
import gradio as gr
from pathlib import Path

from pipeline import Pipeline
from executor import execute_code_safe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Load pipeline and dataset once at startup ---
DATA_PATH = str(Path(__file__).parent.parent / "assets" / "datasets" / "PaySim" / "PS_20174392719_1491204439457_log.csv")

pipe = Pipeline()
df = pd.read_csv(DATA_PATH)
logger.info(f"Dataset loaded: {len(df)} rows")

# --- Session stats accumulator ---
session_history = []


def run_query(user_query: str) -> tuple[str, str, str, str]:
    
    """
    Description:
        Main function to run a user query through the pipeline, execute the generated code, and return results and metadata.
    Args:
        user_query (str): The natural language query input by the user.
    Returns:
        code (str): The generated pandas code.
        exec_output (str): The output from executing the code or an error message.
        explanation (str): The LLM's explanation of the generated code.
        metadata (str): A formatted string containing fallback stage, confidence, columns used, and latency metrics.
    """
    
    if not user_query.strip():
        return "", "", "", ""
    
    t_start = time.time()
    result = pipe.run_query(user_query)
    total_latency = time.time() - t_start
    
    code = result.code_output.code
    explanation = result.code_output.explanation
    confidence = result.code_output.confidence
    columns = result.code_output.columns_used
    stage = result.fallback_stage
    
    # Execute the generated code
    t_exec = time.time()
    exec_result = execute_code_safe(code, df, timeout=30)
    exec_latency = time.time() - t_exec
    
    if exec_result.success:
        exec_output = exec_result.output
    else:
        exec_output = f"Execution error: {exec_result.error}"
    
    # Build metadata string
    metadata = (
        f"Fallback Stage: {stage}/4\n"
        f"Model: {result.model_used}\n"
        f"Query Type: {result.query_type}\n"
        f"Confidence: {confidence}\n"
        f"Columns Used: {', '.join(columns) if columns else 'N/A'}\n"
        f"Latency — Route: {result.latency_route:.2f}s | Retrieve: {result.latency_retrieve:.2f}s | "
        f"Generate: {result.latency_generate:.2f}s | Validate: {result.latency_validate:.2f}s\n"
        f"Pipeline: {result.total_latency:.2f}s | Exec: {exec_latency:.2f}s | Total: {total_latency + exec_latency:.2f}s"
    )
    
    # Accumulate session stats (tracking.py-level detail)
    session_history.append({
        "Query": user_query[:60],
        "Stage": stage,
        "Model": result.model_used,
        "Type": result.query_type,
        "Confidence": confidence,
        "Route (s)": round(result.latency_route, 2),
        "Retrieve (s)": round(result.latency_retrieve, 2),
        "Generate (s)": round(result.latency_generate, 2),
        "Validate (s)": round(result.latency_validate, 2),
        "Total (s)": round(result.total_latency, 2),
        "Exec OK": exec_result.success,
        "Code Safe": stage != 4 or result.model_used != "blocked",
    })
    
    return code, exec_output, explanation, metadata


def get_stats() -> tuple[pd.DataFrame, str]:
    
    """
    Description:
        Computes and returns session statistics matching tracking.py W&B-level detail.
    Returns:
        stats_df (pd.DataFrame): A DataFrame containing the history of queries and their metrics.
        summary (str): A formatted string summarizing key statistics.
    """
    
    if not session_history:
        return pd.DataFrame(), "No queries run yet."
    
    stats_df = pd.DataFrame(session_history)
    n = len(stats_df)
    
    # Core rates
    first_pass_rate = (stats_df["Stage"] == 1).sum() / n
    exec_success_rate = stats_df["Exec OK"].sum() / n
    code_safe_rate = stats_df["Code Safe"].sum() / n
    avg_confidence = stats_df["Confidence"].mean()
    
    # Latency breakdown
    avg_route = stats_df["Route (s)"].mean()
    avg_retrieve = stats_df["Retrieve (s)"].mean()
    avg_generate = stats_df["Generate (s)"].mean()
    avg_validate = stats_df["Validate (s)"].mean()
    avg_total = stats_df["Total (s)"].mean()
    
    # Fallback distribution
    stage_dist = stats_df["Stage"].value_counts().sort_index()
    stage_str = " | ".join([f"Stage {k}: {v}" for k, v in stage_dist.items()])
    
    # Model distribution
    model_dist = stats_df["Model"].value_counts()
    model_str = " | ".join([f"{k}: {v}" for k, v in model_dist.items()])
    
    # Query type distribution
    type_dist = stats_df["Type"].value_counts()
    type_str = " | ".join([f"{k}: {v}" for k, v in type_dist.items()])
    
    summary = (
        f"{'='*50}\n"
        f"  SESSION METRICS ({n} queries)\n"
        f"{'='*50}\n\n"
        f"First-Pass Rate (Stage 1): {first_pass_rate:.0%}\n"
        f"Execution Success Rate:    {exec_success_rate:.0%}\n"
        f"Code Safety Rate:          {code_safe_rate:.0%}\n"
        f"Avg Confidence:            {avg_confidence:.2f}\n\n"
        f"--- Avg Latency Breakdown ---\n"
        f"  Route:      {avg_route:.2f}s\n"
        f"  Retrieve:   {avg_retrieve:.2f}s\n"
        f"  Generate:   {avg_generate:.2f}s\n"
        f"  Validate:   {avg_validate:.2f}s\n"
        f"  Total:      {avg_total:.2f}s\n\n"
        f"--- Distributions ---\n"
        f"  Fallback: {stage_str}\n"
        f"  Models:   {model_str}\n"
        f"  Types:    {type_str}"
    )
    
    return stats_df, summary


# --- Gradio UI ---
with gr.Blocks(title="RAG Financial Analyst", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# RAG Financial Analyst")
    gr.Markdown("Ask natural language questions about 6.3M PaySim financial transactions. "
                "The pipeline routes your query, retrieves domain knowledge, generates pandas code, "
                "validates it, and executes it in a sandbox.")
    
    with gr.Tabs():
        
        # --- Tab 1: Query ---
        with gr.Tab("Query"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="e.g., Show all fraudulent transactions",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Analyze", variant="primary", scale=1)
            
            with gr.Row():
                with gr.Column():
                    code_output = gr.Code(label="Generated Code", language="python")
                    explanation_output = gr.Textbox(label="Explanation", lines=3)
                with gr.Column():
                    exec_output = gr.Textbox(label="Execution Result", lines=12)
                    metadata_output = gr.Textbox(label="Metadata", lines=8)
            
            gr.Examples(
                examples=[
                    "Show all fraudulent transactions",
                    "How many TRANSFER type transactions?",
                    "Average transaction amount",
                    "Flag transactions above 10 lakh",
                    "Find accounts with balance mismatch",
                    "Large cash-outs that zeroed sender balance",
                    "Identify mule accounts",
                    "Top 10 most suspicious accounts by risk score",
                    "Hourly fraud pattern — which hours have highest fraud rate?",
                    "Detect structuring — multiple txns just below 10 lakh",
                ],
                inputs=query_input,
            )
            
            submit_btn.click(
                fn=run_query,
                inputs=query_input,
                outputs=[code_output, exec_output, explanation_output, metadata_output],
            )
            query_input.submit(
                fn=run_query,
                inputs=query_input,
                outputs=[code_output, exec_output, explanation_output, metadata_output],
            )
        
        # --- Tab 2: Session Stats ---
        with gr.Tab("Stats"):
            gr.Markdown("## Session Metrics (W&B-equivalent tracking)")
            gr.Markdown("Per-query latency breakdown, model/type distributions, and aggregate stats — "
                        "same metrics that W&B `tracking.py` logs, rendered live in-session.")
            
            refresh_btn = gr.Button("Refresh Stats", variant="secondary")
            stats_summary = gr.Textbox(label="Summary", lines=18, interactive=False)
            stats_table = gr.Dataframe(label="Query History", interactive=False)
            
            refresh_btn.click(
                fn=get_stats,
                inputs=[],
                outputs=[stats_table, stats_summary],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
