import time
import logging
import pandas as pd
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


# --- Gradio UI (single page, no tabs) ---
with gr.Blocks(title="RAG Financial Analyst") as demo:
    
    gr.Markdown("# RAG Financial Analyst")
    gr.Markdown("Ask natural language questions about 6.3M PaySim financial transactions.")
    
    with gr.Row():
        mode = gr.Radio(
            ["Pipeline", "Agent"], value="Pipeline", label="Mode",
            info="Pipeline = static cascade | Agent = self-correcting loop"
        )
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Query", placeholder="e.g., Show all fraudulent transactions",
            lines=2, scale=4
        )
        submit_btn = gr.Button("Run", variant="primary", scale=1)
    
    with gr.Row():
        with gr.Column():
            code_output = gr.Textbox(label="Generated Code", lines=12)
            explanation_output = gr.Textbox(label="Explanation", lines=3)
        with gr.Column():
            exec_output = gr.Textbox(label="Execution Result", lines=12)
            metadata_output = gr.Textbox(label="Metadata", lines=8)
    
    trace_output = gr.Textbox(
        label="Agent Trace (diagnosis → action at each step)",
        lines=4, visible=False, interactive=False
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
