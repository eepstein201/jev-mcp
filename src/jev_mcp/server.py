import sys
import os

# Shared server state (mcp, provider, linter, logger, chat helpers, notices)
# lives in jev_mcp.app so tool modules never import this __main__ module.
# The names below stay re-exported: tests and downstream callers import them
# from jev_mcp.server.
from jev_mcp.app import (
    _chat_completion,
    _clean_json,
    _safe_truncate,
    _sync_models,
    call_fast_autofixer,
    linter,
    logger,
    mcp,
    provider,
    start_model_sync,
)
from jev_mcp.router_config import save_router_config
from jev_mcp.provider import NoulQuestion
from jev_mcp.tools.browser import jev_run_browser_agent
from jev_mcp.tools.calibrate import jev_calibrate_threshold, jev_manage_temperature
from jev_mcp.tools.prompts import jev_optimize_prompt
from jev_mcp.tools.router import (
    jev_agent_handoff,
    jev_determine_best_model,
    jev_manage_router_config,
)
from jev_mcp.tools.training import jev_train_lora
from jev_mcp.tools.files import jev_compact_context, jev_read_file, jev_scan_repo
from jev_mcp.tools.evaluate import (
    jev_evaluate_batch,
    jev_explain_decision,
    jev_generate_synthetic_dataset,
)
from jev_mcp.tools.triage import (
    tool_configure_triage_labels,
    tool_setup_gas_workflow,
    tool_triage_email_content,
)


@mcp.prompt(
    name="calibrate",
    description="Calibrate the logit threshold for a dataset and automatically fit/save the global MLX temperature.",
)
def calibrate_prompt() -> str:
    return "I want to calibrate Jev. Ask me for the dataset. Then run `jev_calibrate_threshold` with `save_to_config=True`. When it completes, you MUST explicitly notify me that 'The global Jev MLX temperature was dynamically calibrated and saved to your router config, altering default behavior.' If I declined auto-saving, let me know we will fallback to manual Platt scaling."


@mcp.prompt(
    name="compact",
    description="Compress your current context and codebase state to save tokens and eliminate hallucination risk, using Jev's logit confidence gating.",
)
def compact_jev_prompt() -> str:
    return "Gather all your current conversation history, scratchpads, and the contents of any relevant codebase files you currently have open or in your working memory. Immediately execute the `jev_compact_context` MCP tool, passing this massive payload as the `state` argument, and your current goal as the `goal` argument. Do not generate a summary yourself; wait for the Jev tool to return the mathematically compressed payload, and then silently update your working memory with the results to proceed."


@mcp.prompt(
    name="evaluate",
    description="Ask Jev to mathematically evaluate your current codebase or context against a specific set of questions.",
)
def evaluate_prompt() -> str:
    return "I want to mathematically evaluate my current context. Please ask me what questions or criteria I want to evaluate. Once I provide them, gather my relevant codebase state and use the `jev_evaluate_batch` tool to get the mathematical logit probabilities for each question, then present the results."


@mcp.prompt(
    name="explain-decision",
    description="Ask Jev to extract the exact reasoning behind a specific mathematical score or decision.",
)
def explain_decision_prompt() -> str:
    return "I want to understand why Jev gave a specific mathematical score. Please ask me which evaluation or task I want explained. Then use the `jev_explain_decision` tool to extract the generative reasoning behind the logit probability, and present it to me."


@mcp.prompt(
    name="generate-data",
    description="Use the local Jev engine to quickly spin up massive synthetic test datasets.",
)
def generate_data_prompt() -> str:
    return "I want to generate a synthetic dataset locally. Ask me what kind of data I need and the exact schema. Then use the `jev_generate_synthetic_dataset` tool to generate it, and present a sample or save it to my workspace."


@mcp.prompt(
    name="handoff",
    description="Ask Jev to evaluate your current context and mathematically route it to the best specialized subagent using full agent profiles.",
)
def handoff_prompt() -> str:
    return "I want to hand off this task to a specialist. Please generate a list of 3-4 highly relevant specialized agents for this specific task, including their names and a detailed 1-sentence description of their capabilities. Present this list to me for confirmation or edits. Once I approve, pass the task context and the dictionary of agent names/descriptions to the `jev_agent_handoff` tool so Jev can mathematically route the context to the best option."


@mcp.prompt(
    name="model-router",
    description="Ask Jev to calculate the Complexity Index of your current goal and recommend the optimal LLM model to use.",
)
def model_router_prompt() -> str:
    return "Please take my current primary task/goal and execute the `jev_determine_best_model` tool to calculate its mathematical Complexity Index. Once Jev returns the recommended model and complexity breakdown, present the results to me. If Jev warns about massive context, or recommends a more powerful model than I am currently using, please proactively ask me if I want to switch models before we proceed."


@mcp.prompt(
    name="manage-router",
    description="View or modify your dynamic LLM routing configuration (buckets and exception rules) using natural language.",
)
def router_config_prompt() -> str:
    return "I would like to configure my Jev model routing settings. Please use the `jev_manage_router_config` tool with the `view_all` action to retrieve my current settings. Present my current buckets and exception rules to me in a clean, readable format. Then, ask me what I would like to change (e.g., adding a rule, removing a rule, or reassigning a bucket model)."


@mcp.prompt(
    name="optimize-prompt",
    description="Mathematically optimize a prompt for maximum LLM adherence.",
)
def optimize_prompt() -> str:
    return "I want to mathematically optimize a prompt. Ask me what prompt I want to optimize and what my goals are. Then use the `jev_optimize_prompt` tool to generate and validate the optimal version of the prompt using logit extraction, and present the final optimized prompt to me."


@mcp.prompt(
    name="temperature",
    description="View or change Jev's global MLX calibration temperature (which affects all evaluations).",
)
def temperature_prompt() -> str:
    return "I want to manage Jev's global calibration temperature. First, use `jev_manage_temperature` to view the current temperature. Then ask me if I want to keep it, manually set a new value, or reset it to the default 1.0."


@mcp.prompt(
    name="train",
    description="Train a local MLX LoRA adapter on your Apple Silicon GPU using any dataset format to customize Jev.",
)
def train_prompt() -> str:
    return "I want to fine-tune Jev to my codebase. If I provide a dataset in CSV, Markdown, or another raw format, you MUST first convert it into the strict `.jsonl` schema required by Jev and save it locally. (If I don't have data, use `jev_generate_synthetic_dataset` to generate it). Once the `.jsonl` file is ready, pass its path to the `jev_train_lora` tool to instantly train a custom adapter on the GPU."


@mcp.prompt(
    name="scan-repo",
    description="Ask Jev to surgically scan the current workspace repository to find and extract the AST chunks relevant to a specific task.",
)
def scan_repo_prompt() -> str:
    return "I want to scan this repository for code relevant to my current task. Please ask me for the target directory path (if not the root) and the specific bug or feature I am looking for. Then execute the `jev_scan_repo` tool with those arguments and report back the mathematically extracted AST chunks."


@mcp.prompt(
    name="read-file",
    description="Ask Jev to read a specific file and mathematically extract only the AST chunks relevant to a task, dropping irrelevant noise.",
)
def read_file_prompt() -> str:
    return "I want to read a file, but I only want the parts relevant to my current task. Please ask me for the file path and my objective. Then execute the `jev_read_file` tool with filter_by_chunk=True and report back the extracted AST blocks."


def main():
    import sys
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == "setup-gas":
        parser = argparse.ArgumentParser(prog="jev-mcp setup-gas")
        parser.add_argument(
            "--webhook-url", type=str, help="The Webhook URL to route emails to."
        )

        # Parse arguments starting from the 3rd argument since sys.argv[1] is 'setup-gas'
        args = parser.parse_args(sys.argv[2:])

        from jev_mcp.email_triage.gas_setup import setup_gas_workflow

        setup_gas_workflow(webhook_url=args.webhook_url)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "serve-webhook":
        parser = argparse.ArgumentParser(prog="jev-mcp serve-webhook")
        parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Interface to bind. The default is local only.",
        )
        parser.add_argument(
            "--port", type=int, default=8090, help="Port for POST /api/v1/triage/email."
        )
        args = parser.parse_args(sys.argv[2:])

        credential_vars = ("JEV_MCP_API_KEY", "JEV_MCP_JWT_SECRET", "JEV_MCP_JWKS_URL")
        if not any(os.environ.get(name) for name in credential_vars):
            sys.exit(
                f"Refusing to start: set one of {', '.join(credential_vars)} so the webhook can authenticate callers."
            )

        import uvicorn
        from jev_mcp.email_triage.api import app

        uvicorn.run(app, host=args.host, port=args.port)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from jev_mcp.doctor import main as doctor_main

        sys.exit(doctor_main(sys.argv[2:]))

    start_model_sync()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
