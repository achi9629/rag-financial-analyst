import time
import logging
import pandas as pd
import mlflow
import gradio as gr
from pathlib import Path

from pipeline import Pipeline
from agent import AgentPipeline
from executor import execute_code_safe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Load pipeline, agent, and dataset once at startup ---
DATA_PATH = str(Path(__file__).parent.parent / "assets" / "datasets" / "PaySim" / "PS_20174392719_1491204439457_log.csv")

pipe = Pipeline()
agent = AgentPipeline(pipe)
df = pd.read_csv(DATA_PATH)
logger.info(f"Dataset loaded: {len(df)} rows")

# --- MLflow experiment setup (optional, logs to ./mlruns by default) ---
MLFLOW_EXPERIMENT = "rag-financial-analyst"
mlflow.set_tracking_uri("file:mlruns")
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
except Exception:
    pass
# --- Session stats accumulator ---
session_history = []


def run_unified(user_query: str, mode: str) -> tuple[str, str, str, str, str]:
    """Route to pipeline or agent based on mode selection."""
    
    if not user_query.strip():
        return "", "", "", "", ""
    
    if mode == "Agent":
        return _run_agent(user_query)
    else:
        return _run_pipeline(user_query)


def _run_pipeline(user_query: str) -> tuple[str, str, str, str, str]:
    """Run query through static pipeline."""
    
    t_start = time.time()
    result = pipe.run_query(user_query)
    total_latency = time.time() - t_start
    
    code = result.code_output.code
    explanation = result.code_output.explanation
    confidence = result.code_output.confidence
    columns = result.code_output.columns_used
    stage = result.fallback_stage
    
    t_exec = time.time()
    exec_result = execute_code_safe(code, df, timeout=30)
    exec_latency = time.time() - t_exec
    
    if exec_result.success:
        exec_output = exec_result.output
    else:
        exec_output = f"Execution error: {exec_result.error}"
    
    metadata = (
        f"Mode: Pipeline (static cascade)\n"
        f"Fallback Stage: {stage}/4\n"
        f"Model: {result.model_used}\n"
        f"Query Type: {result.query_type}\n"
        f"Confidence: {confidence}\n"
        f"Columns Used: {', '.join(columns) if columns else 'N/A'}\n"
        f"Latency — Route: {result.latency_route:.2f}s | Retrieve: {result.latency_retrieve:.2f}s | "
        f"Generate: {result.latency_generate:.2f}s | Validate: {result.latency_validate:.2f}s\n"
        f"Pipeline: {result.total_latency:.2f}s | Exec: {exec_latency:.2f}s | Total: {total_latency + exec_latency:.2f}s"
    )
    
    session_history.append({
        "Query": user_query[:60],
        "Mode": "pipeline",
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
    
    # --- Log to MLflow (per-query run, non-blocking) ---
    mlflow.set_tracking_uri("file:mlruns")
    try:
        with mlflow.start_run(run_name=f"pipeline-{int(time.time())}"):
            mlflow.log_params({
                "query": user_query[:250],
                "mode": "pipeline",
                "model": result.model_used,
                "query_type": result.query_type,
            })
            mlflow.log_metrics({
                "fallback_stage": stage,
                "confidence": confidence,
                "latency_route": result.latency_route,
                "latency_retrieve": result.latency_retrieve,
                "latency_generate": result.latency_generate,
                "latency_validate": result.latency_validate,
                "latency_total": result.total_latency,
                "latency_exec": exec_latency,
                "exec_success": int(exec_result.success),
                "code_safe": int(stage != 4 or result.model_used != "blocked"),
        })
    except Exception:
        logger.debug("MLflow logging skipped")
    return code, exec_output, explanation, metadata, ""


def _run_agent(user_query: str) -> tuple[str, str, str, str, str]:
    """Run query through agentic self-correction loop."""
    
    result = agent.run(user_query, df)
    
    # Format action trace
    trace_lines = []
    for i, action_str in enumerate(result.action_history):
        trace_lines.append(f"  [{i+1}] {action_str}")
    trace = "\n".join(trace_lines) if trace_lines else "  (no actions)"
    
    if result.exec_output:
        exec_output = result.exec_output
    elif result.error:
        exec_output = f"Error: {result.error}"
    else:
        exec_output = ""
    
    metadata = (
        f"Mode: Agent (self-correcting loop)\n"
        f"Result: {result.final_action}\n"
        f"Attempts: {result.attempts}\n"
        f"Query Type: {result.query_type}\n"
        f"Confidence: {result.confidence}\n"
        f"Columns Used: {', '.join(result.columns_used) if result.columns_used else 'N/A'}\n"
        f"Total Latency: {result.total_latency:.2f}s"
    )
    
    session_history.append({
        "Query": user_query[:60],
        "Mode": "agent",
        "Stage": result.attempts,
        "Model": "qwen+llama",
        "Type": result.query_type,
        "Confidence": result.confidence,
        "Route (s)": "N/A",
        "Retrieve (s)": "N/A",
        "Generate (s)": "N/A",
        "Validate (s)": "N/A",
        "Total (s)": round(result.total_latency, 2),
        "Exec OK": result.final_action == "stop_success",
        "Code Safe": result.final_action != "stop_blocked",
    })
    
    # --- Log to MLflow (per-query run, non-blocking) ---
    mlflow.set_tracking_uri("file:mlruns")
    try:
        with mlflow.start_run(run_name=f"agent-{int(time.time())}"):
            mlflow.log_params({
                "query": user_query[:250],
                "mode": "agent",
                "model": "qwen+llama",
                "query_type": result.query_type,
                "attempts": str(result.attempts),
            })
            mlflow.log_metrics({
                "confidence": result.confidence,
                "latency_total": result.total_latency,
                "attempts": result.attempts,
                "exec_success": int(result.final_action == "stop_success"),
                "code_safe": int(result.final_action != "stop_blocked"),
            })
    except Exception:
        logger.debug("MLflow logging skipped (server not available)")
    return result.code, exec_output, result.explanation, metadata, trace


def toggle_trace(mode: str):
    """Show trace field only in Agent mode."""
    return gr.update(visible=(mode == "Agent"))


def get_stats() -> tuple[pd.DataFrame, str]:
    """Compute session stats across both pipeline and agent modes."""
    
    if not session_history:
        return pd.DataFrame(), "No queries run yet."
    
    stats_df = pd.DataFrame(session_history)
    n = len(stats_df)
    
    first_pass_rate = (stats_df["Stage"] == 1).sum() / n
    exec_success_rate = stats_df["Exec OK"].sum() / n
    code_safe_rate = stats_df["Code Safe"].sum() / n
    avg_confidence = stats_df["Confidence"].mean()
    avg_total = stats_df["Total (s)"].mean()
    
    stage_dist = stats_df["Stage"].value_counts().sort_index()
    stage_str = " | ".join([f"Stage {k}: {v}" for k, v in stage_dist.items()])
    
    model_dist = stats_df["Model"].value_counts()
    model_str = " | ".join([f"{k}: {v}" for k, v in model_dist.items()])
    
    mode_dist = stats_df["Mode"].value_counts()
    mode_str = " | ".join([f"{k}: {v}" for k, v in mode_dist.items()])
    
    type_dist = stats_df["Type"].value_counts()
    type_str = " | ".join([f"{k}: {v}" for k, v in type_dist.items()])
    
    # Pipeline-only latency breakdown
    pipe_df = stats_df[stats_df["Mode"] == "pipeline"]
    if len(pipe_df) > 0:
        latency_block = (
            f"\n--- Avg Latency Breakdown (Pipeline mode) ---\n"
            f"  Route:      {pd.to_numeric(pipe_df['Route (s)'], errors='coerce').mean():.2f}s\n"
            f"  Retrieve:   {pd.to_numeric(pipe_df['Retrieve (s)'], errors='coerce').mean():.2f}s\n"
            f"  Generate:   {pd.to_numeric(pipe_df['Generate (s)'], errors='coerce').mean():.2f}s\n"
            f"  Validate:   {pd.to_numeric(pipe_df['Validate (s)'], errors='coerce').mean():.2f}s\n"
        )
    else:
        latency_block = ""
    
    summary = (
        f"{'='*50}\n"
        f"  SESSION METRICS ({n} queries)\n"
        f"{'='*50}\n\n"
        f"First-Pass Rate:           {first_pass_rate:.0%}\n"
        f"Execution Success Rate:    {exec_success_rate:.0%}\n"
        f"Code Safety Rate:          {code_safe_rate:.0%}\n"
        f"Avg Confidence:            {avg_confidence:.2f}\n"
        f"Avg Total Latency:         {avg_total:.2f}s\n"
        f"{latency_block}\n"
        f"--- Distributions ---\n"
        f"  Modes:    {mode_str}\n"
        f"  Fallback: {stage_str}\n"
        f"  Models:   {model_str}\n"
        f"  Types:    {type_str}"
    )
    
    return stats_df, summary



# --- Custom CSS for richer styling ---
CUSTOM_CSS = """
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}
.main-header h1 {
    color: #7c3aed;
    margin: 0;
    font-size: 2em;
}
.main-header p {
    color: #a8a8b8;
    margin: 5px 0 0 0;
}
.metadata-box {
    background: #1a1a2e;
    border-left: 3px solid #7c3aed;
    padding: 10px;
    border-radius: 5px;
}
"""

# --- Gradio UI (themed, syntax-highlighted) ---
THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)

with gr.Blocks(
    title="RAG Financial Analyst",
) as demo:
    
    gr.HTML("""
        <div class="main-header">
            <h1>RAG Financial Analyst</h1>
            <p>Natural language queries over 6.3M PaySim transactions · Powered by Qwen2.5-Coder-32B + Llama-3-70B via vLLM</p>
        </div>
    """)
    
    with gr.Row():
        mode = gr.Radio(
            ["Pipeline", "Agent"], value="Pipeline", label="Mode",
            info="Pipeline = 4-stage static cascade | Agent = self-correcting loop with execution feedback"
        )
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Query", 
            placeholder="e.g., Identify mule accounts that receive transfers and immediately cash out",
            lines=2, scale=4
        )
        submit_btn = gr.Button("Run", variant="primary", scale=1, size="lg")
    
    gr.Examples(
        examples=[
            "Show all fraudulent transactions",
            "Flag transactions above 10 lakh",
            "Large cash-outs that zeroed sender balance",
            "Identify mule accounts",
            "Detect structuring - multiple transactions just below 10 lakh",
        ],
        inputs=query_input,
        label="Example Queries",
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            code_output = gr.Code(
                label="Generated Code", 
                language="python",
                lines=14,
                interactive=False,
            )
            explanation_output = gr.Textbox(label="Explanation", lines=3, interactive=False)
        with gr.Column(scale=1):
            exec_output = gr.Textbox(label="Execution Result", lines=14, interactive=False)
            metadata_output = gr.Textbox(
                label="Metadata", lines=8, interactive=False,
                elem_classes=["metadata-box"]
            )
    
    trace_output = gr.Textbox(
        label="Agent Trace (diagnosis → action at each step)",
        lines=5, visible=False, interactive=False
    )
    
    # Toggle trace visibility when mode changes
    mode.change(fn=toggle_trace, inputs=mode, outputs=trace_output)
    
    # Run query
    submit_btn.click(
        fn=run_unified, inputs=[query_input, mode],
        outputs=[code_output, exec_output, explanation_output, metadata_output, trace_output],
    )
    query_input.submit(
        fn=run_unified, inputs=[query_input, mode],
        outputs=[code_output, exec_output, explanation_output, metadata_output, trace_output],
    )
    
    # Stats section
    with gr.Accordion("Session Stats", open=False):
        refresh_btn = gr.Button("Refresh Stats", variant="secondary")
        stats_summary = gr.Textbox(label="Summary", lines=15, interactive=False)
        stats_table = gr.Dataframe(label="Query History", interactive=False)
        refresh_btn.click(fn=get_stats, inputs=[], outputs=[stats_table, stats_summary])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, theme=THEME, css=CUSTOM_CSS)
