import sys
import os
import json
import subprocess
from typing import Annotated, Any, Dict, List, Optional, Literal
from pydantic import Field

# Shared server state (mcp, provider, linter, logger, chat helpers, notices)
# lives in jev_mcp.app so tool modules never import this __main__ module.
from jev_mcp.app import (
    PLATT_CV_FOLDS,
    SKIP_MODEL_SYNC_ENV,
    _chat_completion,
    _chat_completion_json,
    _clean_json,
    _safe_truncate,
    _sync_models,
    call_fast_autofixer,
    get_model_notice,
    get_temp_notice,
    linter,
    logger,
    mcp,
    model_update_notice,
    provider,
    start_model_sync,
)
from jev_mcp import router_config
from jev_mcp.router_config import apply_model_updates, load_router_config, pending_model_updates, save_router_config
from jev_mcp.provider import QuestionType, NoulQuestion, ChoiceQuestion, ScoreQuestion
from jev_mcp.routing_provider import RoutingProvider
from jev_mcp.tools.browser import jev_run_browser_agent
from jev_mcp.tools.calibrate import jev_calibrate_threshold, jev_manage_temperature
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
import re

@mcp.tool(
    name="optimize-prompt",
    description="Automated Prompt Engineer: Generates semantic variations of a prompt and scores them. With `dataset` (rows of state + expected) it returns the phrasing with the highest measured accuracy. With only a single `state` it can only return the most confident phrasing, which is not the same as the most accurate.",
)
def jev_optimize_prompt(
    question: Annotated[QuestionType, Field(description="The draft question to optimize.")],
    state: Annotated[Optional[Dict[str, Any]], Field(description="One sample context state. Used only when no dataset is given; selection is then by confidence.")] = None,
    dataset: Annotated[Optional[List[Dict[str, Any]]], Field(description="Labelled rows, each {\"state\": ..., \"expected\": ...}: true/false for a noul question, the option for a choice question, the label for a score question. Selection is then by accuracy.")] = None,
) -> str:
    import json
    from jev_mcp import prompt_selection
    from jev_mcp.security import sanitize_payload

    if dataset is not None:
        rows_ok = isinstance(dataset, list) and bool(dataset) and all(
            isinstance(row, dict) and "state" in row and "expected" in row for row in dataset
        )
        if not rows_ok:
            return json.dumps({"status": "ERROR", "message": "dataset must be a non-empty list of rows, each with \"state\" and \"expected\"."})
    elif state is None:
        return json.dumps({"status": "ERROR", "message": "Provide a dataset of labelled rows (preferred) or a single state."})

    logger.info("Starting Prompt Optimization...")
    q_dict: Dict[str, Any] = {}
    if hasattr(question, "type"):
        if question.type == "noul":
            q_dict = {"type": "noul", "key": question.key, "prompt": question.prompt}
        elif question.type == "choice":
            q_dict = {
                "type": "choice",
                "key": question.key,
                "prompt": question.prompt,
                "options": getattr(question, "options", []),
            }
        elif question.type == "score":
            q_dict = {
                "type": "score",
                "key": question.key,
                "prompt": question.prompt,
                "labels": getattr(question, "labels", []),
            }
    else:
        q_dict = question.dict() if hasattr(question, "dict") else vars(question)

    q_dict_str = sanitize_payload(json.dumps(q_dict))

    system_prompt = (
        "You are an expert Prompt Engineer for a Logit-based System-1 decision model. "
        "The user will provide a draft question. Generate exactly 5 semantic variations of the prompt phrasing. "
        "The variations must test different semantic angles (e.g. active vs passive, formal vs informal, broad vs specific) but must NOT change the core intent. "
        "Output ONLY a JSON object.\n"
        'Format: {"variations": ["Variation 1", "Variation 2", ...]}'
    )
    try:
        # Guaranteed JSON via Constrained Decoding
        variations_data = _chat_completion_json(
            user=f"Optimize this question: {q_dict_str}",
            system=system_prompt,
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        variations = variations_data.get("variations", [])
        if isinstance(variations, str):
            variations = [variations]
        if not isinstance(variations, list):
            variations = []
    except Exception as e:
        return f"Optimizer Generator Failed: {str(e)}"

    variations.insert(0, q_dict["prompt"])  # include original
    test_questions: List[QuestionType] = []

    for i, var_prompt in enumerate(variations):
        key = f"var_{i}"
        if q_dict["type"] == "noul":
            test_questions.append(NoulQuestion(key=key, prompt=var_prompt))
        elif q_dict["type"] == "choice":
            test_questions.append(
                ChoiceQuestion(key=key, prompt=var_prompt, options=q_dict["options"])
            )
        elif q_dict["type"] == "score":
            test_questions.append(
                ScoreQuestion(key=key, prompt=var_prompt, labels=q_dict["labels"])
            )

    if dataset is not None:
        try:
            row_results = [provider.evaluate_batch(row["state"], test_questions) for row in dataset]
        except Exception as e:
            return f"Optimizer Evaluation Failed: {str(e)}"
        per_variation = [[res.get(q.key, {}) for res in row_results] for q in test_questions]
        stats = prompt_selection.score_variations(per_variation, [row["expected"] for row in dataset])
        best = prompt_selection.best_variation(stats)
        gain = (stats[best]["accuracy"] - stats[0]["accuracy"]) * 100
        return json.dumps(
            {
                "selected_by": "accuracy",
                "rows": len(dataset),
                "original_prompt": q_dict["prompt"],
                "original_accuracy": f"{stats[0]['accuracy'] * 100:.2f}%",
                "optimized_prompt": variations[best],
                "optimized_accuracy": f"{stats[best]['accuracy'] * 100:.2f}%",
                "improvement_points": f"{gain:.2f}",
                "recommendation": (
                    "Keep the original: no variation scored better on this dataset."
                    if best == 0
                    else f"Adopt the optimized prompt: it scored better on these {len(dataset)} rows. Confirm on rows it was not selected on."
                ),
                "variations": [
                    {
                        "prompt": variations[i],
                        "accuracy": f"{s['accuracy'] * 100:.2f}%",
                        "mean_correct_probability": round(s["mean_correct_probability"], 4),
                    }
                    for i, s in enumerate(stats)
                ],
            },
            indent=2,
        )

    try:
        results = provider.evaluate_batch(state, test_questions)
    except Exception as e:
        return f"Optimizer Evaluation Failed: {str(e)}"

    best_var: Dict[str, Any] | None = None
    best_conf = -1.0
    original_conf = 0.0

    for i, res in enumerate(results.values()):
        # Calculate a simple confidence metric (max prob)
        conf: float = 0.0
        if "confidence" in res:
            conf = res["confidence"]
        elif "noul" in res:
            conf = max(res["noul"], 1.0 - res["noul"])
        elif "probabilities" in res:
            conf = max(res["probabilities"].values())

        if i == 0:
            original_conf = conf

        if conf > best_conf:
            best_conf = conf
            best_var = {
                "optimized_prompt": variations[i],
                "optimized_confidence": f"{conf * 100:.2f}%",
                "raw_result": res,
            }

    if not best_var:
        return json.dumps({"error": "No variations evaluated."})
    return json.dumps(
        {
            "original_prompt": q_dict["prompt"],
            "original_confidence": f"{original_conf * 100:.2f}%",
            "optimized_prompt": best_var["optimized_prompt"],
            "optimized_confidence": best_var["optimized_confidence"],
            "improvement_points": f"{(best_conf - original_conf) * 100:.2f}",
            "selected_by": "confidence",
            "recommendation": "Selected by confidence on one state, which is not accuracy: a phrasing can be more confident and wrong. Pass `dataset` (rows of state + expected) to select by measured accuracy.",
            "all_variations_tested": len(variations),
        },
        indent=2,
    )

@mcp.tool(
    name="manage-router",
    description="Interactive configuration manager for Jev's Model Router. Use this to view, add, or remove custom bucket models and exception rules. view_all also reports newer models available for each bucket; apply_updates switches to them; set_auto_update (enabled=true/false) records whether Jev should do that automatically at start-up.",
)
def jev_manage_router_config(
    action: Literal["view_all", "add_rule", "remove_rule", "set_bucket", "apply_updates", "set_auto_update"] = Field(description="The action to perform."),
    bucket_id: Optional[str] = Field(default=None, description="The bucket (b1, b2, b3, b4) to modify."),
    model_name: Optional[str] = Field(default=None, description="The model name to assign to the bucket."),
    condition: Optional[str] = Field(default=None, description="The natural language condition for the rule (e.g., 'Touches > 10 files')."),
    target: Optional[str] = Field(default=None, description="The target bucket or model if the condition is met."),
    rule_id: Optional[int] = Field(default=None, description="The index of the rule to remove."),
    enabled: Optional[bool] = None,
) -> str:
    config = load_router_config()
    
    if action == "view_all":
        updates = pending_model_updates(config)
        if not updates:
            return json.dumps(config, indent=2)
        return json.dumps({
            **config,
            "available_model_updates": updates,
            "_model_update_notice": model_update_notice(config, updates),
        }, indent=2)

    if action == "apply_updates":
        updates = pending_model_updates(config)
        if not updates:
            return json.dumps({"status": "UP_TO_DATE", "message": "Every bucket already holds the newest model in its family."})
        save_router_config(apply_model_updates(config, updates))
        return json.dumps({"status": "UPDATED", "updated": {bucket: u["newer"] for bucket, u in updates.items()}}, indent=2)

    if action == "set_auto_update":
        if not isinstance(enabled, bool):
            return "Error: enabled (true or false) required."
        updates = pending_model_updates(config) if enabled else {}
        new_config = apply_model_updates(config, updates) if updates else config
        save_router_config({**new_config, "auto_update": enabled})
        return json.dumps({"status": "SAVED", "auto_update": enabled, "updated": {bucket: u["newer"] for bucket, u in updates.items()}}, indent=2)

    if action == "set_bucket":
        if not bucket_id or not model_name:
            return "Error: bucket_id and model_name required."
        config["buckets"][bucket_id] = model_name
        save_router_config(config)
        return f"Bucket {bucket_id} successfully mapped to {model_name}."
        
    if action == "add_rule":
        if not condition or not target:
            return "Error: condition and target required."
        from jev_mcp.security import sanitize_payload
        config["rules"].append({"condition": sanitize_payload(condition), "target": target})
        save_router_config(config)
        return f"Rule added. If '{condition}', route to '{target}'."
        
    if action == "remove_rule":
        if rule_id is None or rule_id < 0 or rule_id >= len(config["rules"]):
            return "Error: Valid rule_id required."
        removed = config["rules"].pop(rule_id)
        save_router_config(config)
        return f"Rule removed: {removed}"

@mcp.tool(
    name="model-router",
    description="Complexity Index Router: Analyzes a task description using the Hybrid local engine and custom rules to determine the mathematically optimal LLM model to use.",
)
def jev_determine_best_model(
    task_description: str = Field(description="The prompt or goal the CLI is about to execute."),
    estimated_tokens: Optional[int] = Field(default=0, description="The estimated token count of the context payload.")
) -> str:
    config = load_router_config()
    
    # 1. Context Window Check
    if estimated_tokens and estimated_tokens > 100000:
        return json.dumps({
            "status": "MASSIVE_CONTEXT_DETECTED",
            "warning": "Massive context detected (>100k tokens). Please ensure your client enables extra token context flags or select a specialized 1M+ context model. Provide the custom model name if you wish to override.",
            "recommended_tier": "b3",
            "recommended_model": config["buckets"].get("b3", router_config.FALLBACK_BUCKETS["b3"])
        }, indent=2)
        
    # 2. Evaluate Custom Rules (Hybrid Evidence-Based)
    provider = RoutingProvider()
    state = {"task": task_description}
    for rule in config["rules"]:
        condition = rule["condition"]
        q = NoulQuestion(key="rule", prompt=f"Does the following task meet this condition: '{condition}'?")
        try:
            res = provider.evaluate_batch(state, [q])
            if res.get("rule", {}).get("noul", 0.0) > 0.85:
                return json.dumps({
                    "status": "RULE_TRIGGERED",
                    "triggered_rule": condition,
                    "recommended_target": rule["target"],
                    "model": config["buckets"].get(rule["target"], rule["target"])
                }, indent=2)
        except Exception as e:
            logger.warning(f"Rule evaluation failed: {e}")
            
    # 3. Complexity Index (Multi-Class Logit Routing)
    complexity_q = ScoreQuestion(
        key="complexity",
        prompt="Score the complexity of this task on a scale from 1 to 4. 1 is trivial, 4 is an agentic refactor.",
        labels=["1", "2", "3", "4"]
    )

    try:
        res = provider.evaluate_batch(state, [complexity_q])
        score = res.get("complexity", {}).get("score", "1")
        probs = res.get("complexity", {}).get("probabilities", {})
        
        bucket = f"b{score}"
        recommended_model = config["buckets"].get(bucket, router_config.FALLBACK_BUCKETS["b1"])
        
        res_dict = {
            "status": "EVALUATED",
            "recommended_model": recommended_model,
            "complexity_score": int(score),
            "reasoning": f"Task complexity scored {score}/4. Routed to {recommended_model}.",
            "probabilities": probs
        }
        notice = get_temp_notice()
        if notice:
            res_dict["_system_notice"] = notice
        model_notice = get_model_notice()
        if model_notice:
            res_dict["_model_update_notice"] = model_notice
            
        return json.dumps(res_dict, indent=2)
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})

@mcp.prompt(
    name="calibrate",
    description="Calibrate the logit threshold for a dataset and automatically fit/save the global MLX temperature."
)
def calibrate_prompt() -> str:
    return "I want to calibrate Jev. Ask me for the dataset. Then run `calibrate` with `save_to_config=True`. When it completes, you MUST explicitly notify me that 'The global Jev MLX temperature was dynamically calibrated and saved to your router config, altering default behavior.' If I declined auto-saving, let me know we will fallback to manual Platt scaling."

@mcp.prompt(
    name="compact",
    description="Compress your current context and codebase state to save tokens and eliminate hallucination risk, using Jev's logit confidence gating."
)
def compact_jev_prompt() -> str:
    return "Gather all your current conversation history, scratchpads, and the contents of any relevant codebase files you currently have open or in your working memory. Immediately execute the `compact` MCP tool, passing this massive payload as the `state` argument, and your current goal as the `goal` argument. Do not generate a summary yourself; wait for the Jev tool to return the mathematically compressed payload, and then silently update your working memory with the results to proceed."

@mcp.prompt(
    name="evaluate",
    description="Ask Jev to mathematically evaluate your current codebase or context against a specific set of questions."
)
def evaluate_prompt() -> str:
    return "I want to mathematically evaluate my current context. Please ask me what questions or criteria I want to evaluate. Once I provide them, gather my relevant codebase state and use the `evaluate` tool to get the mathematical logit probabilities for each question, then present the results."

@mcp.prompt(
    name="explain-decision",
    description="Ask Jev to extract the exact reasoning behind a specific mathematical score or decision."
)
def explain_decision_prompt() -> str:
    return "I want to understand why Jev gave a specific mathematical score. Please ask me which evaluation or task I want explained. Then use the `explain-decision` tool to extract the generative reasoning behind the logit probability, and present it to me."

@mcp.prompt(
    name="generate-data",
    description="Use the local Jev engine to quickly spin up massive synthetic test datasets."
)
def generate_data_prompt() -> str:
    return "I want to generate a synthetic dataset locally. Ask me what kind of data I need and the exact schema. Then use the `generate-data` tool to generate it, and present a sample or save it to my workspace."

@mcp.prompt(
    name="handoff",
    description="Ask Jev to evaluate your current context and mathematically route it to the best specialized subagent using full agent profiles."
)
def handoff_prompt() -> str:
    return "I want to hand off this task to a specialist. Please generate a list of 3-4 highly relevant specialized agents for this specific task, including their names and a detailed 1-sentence description of their capabilities. Present this list to me for confirmation or edits. Once I approve, pass the task context and the dictionary of agent names/descriptions to the `handoff` tool so Jev can mathematically route the context to the best option."

@mcp.prompt(
    name="model-router",
    description="Ask Jev to calculate the Complexity Index of your current goal and recommend the optimal LLM model to use."
)
def model_router_prompt() -> str:
    return "Please take my current primary task/goal and execute the `model-router` tool to calculate its mathematical Complexity Index. Once Jev returns the recommended model and complexity breakdown, present the results to me. If Jev warns about massive context, or recommends a more powerful model than I am currently using, please proactively ask me if I want to switch models before we proceed."

@mcp.prompt(
    name="manage-router",
    description="View or modify your dynamic LLM routing configuration (buckets and exception rules) using natural language."
)
def router_config_prompt() -> str:
    return "I would like to configure my Jev model routing settings. Please use the `manage-router` tool with the `view_all` action to retrieve my current settings. Present my current buckets and exception rules to me in a clean, readable format. Then, ask me what I would like to change (e.g., adding a rule, removing a rule, or reassigning a bucket model)."

@mcp.prompt(
    name="optimize-prompt",
    description="Mathematically optimize a prompt for maximum LLM adherence."
)
def optimize_prompt() -> str:
    return "I want to mathematically optimize a prompt. Ask me what prompt I want to optimize and what my goals are. Then use the `optimize-prompt` tool to generate and validate the optimal version of the prompt using logit extraction, and present the final optimized prompt to me."

@mcp.tool(
    name="handoff",
    description="Uses multi-class logit routing to mathematically determine which specialized agent should take over the current task based on their full descriptions."
)
def jev_agent_handoff(task_description: str, available_agents: Dict[str, str]) -> str:
    """
    Passes the task description and agent profiles to Jev as a ChoiceQuestion. 
    """
    provider = RoutingProvider()
    
    agent_profiles = "\n".join([f"- {name}: {desc}" for name, desc in available_agents.items()])
    instructions = f"Which of the following specialized AI agents is best suited to execute this task based on their profiles?\nProfiles:\n{agent_profiles}"
    
    q = ChoiceQuestion(
        key="best_agent",
        prompt=instructions,
        options=list(available_agents.keys())
    )
    res = provider.evaluate_batch(task_description, [q])
    best_agent = res.get("best_agent", {}).get("choice")
    probs = res.get("best_agent", {}).get("probabilities", {})
    
    res_dict = {
        "status": "HANDOFF_RECOMMENDATION",
        "recommended_agent": best_agent,
        "probabilities": probs
    }
    notice = get_temp_notice()
    if notice:
        res_dict["_system_notice"] = notice
        
    return json.dumps(res_dict, indent=2)

@mcp.tool(
    name="train",
    description="Trains a local MLX LoRA adapter on your Apple Silicon GPU. Options: fuse=True (default) merges the adapter into a Q4_K_M .gguf file (llama.cpp's converter is fetched automatically for the installed llama-server commit), and if target_gguf_path is provided, replaces that model (keeping the original as <target>.bak) and restarts the launchd agent serving it. fuse=False keeps the standalone MLX adapter. iters sets the number of training iterations (default 500). action='restore' with target_gguf_path puts the .bak back and restarts that agent."
)
def jev_train_lora(dataset_path: str = "", model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit", engine: str = "mlx", fuse: bool = True, target_gguf_path: Optional[str] = None, iters: int = 500, action: Literal["train", "restore"] = "train") -> str:
    """
    Launches mlx_lm.lora fine-tuning in a detached background process
    and returns the live pid plus adapter/log locations.
    """
    import importlib.util
    import time
    from jev_mcp import lora_training
    from jev_mcp.security import is_safe_path

    try:
        if target_gguf_path:
            if not is_safe_path(target_gguf_path):
                return json.dumps({"status": "BLOCKED", "message": f"Access denied: '{target_gguf_path}' resolves to a sensitive system or credential location."})
            if not target_gguf_path.endswith(".gguf"):
                return json.dumps({"status": "ERROR", "message": f"target_gguf_path must be a .gguf file: {target_gguf_path}"})
            target_gguf_path = os.path.abspath(os.path.expanduser(target_gguf_path))

        if action == "restore":
            if not target_gguf_path:
                return json.dumps({"status": "ERROR", "message": "restore needs target_gguf_path (the model whose .bak should be put back)."})
            return json.dumps(lora_training.restore_model(target_gguf_path), indent=2)

        if not is_safe_path(dataset_path):
            return json.dumps({"status": "BLOCKED", "message": f"Access denied: '{dataset_path}' resolves to a sensitive system or credential location."})

        dataset_path = os.path.expanduser(dataset_path)

        if not os.path.isfile(dataset_path) or not dataset_path.endswith(".jsonl"):
            return json.dumps({"status": "ERROR", "message": f"Dataset not found or not a .jsonl file: {dataset_path}"})

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", model_name):
            return json.dumps({"status": "ERROR", "message": f"Invalid model_name (expected a HuggingFace id like 'org/model-name'): {model_name}"})

        if not isinstance(iters, int) or iters < 1:
            return json.dumps({"status": "ERROR", "message": f"iters must be a positive integer, got: {iters}"})

        # Reuse a single training dir under ~/.jev (no mkdtemp litter) and link
        # the dataset in under the name mlx_lm.lora expects.
        run_root = os.path.expanduser("~/.jev/training")
        os.makedirs(run_root, exist_ok=True)
        if os.path.basename(dataset_path) != "train.jsonl":
            link = os.path.join(run_root, "train.jsonl")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(dataset_path), link)

        # Adapters and logs get a fresh per-run directory instead of the CWD.
        run_id = time.strftime("%Y%m%d-%H%M%S")
        restart_label: Optional[str] = None
        adapter_path = os.path.expanduser(f"~/.jev/adapters/{run_id}")
        os.makedirs(adapter_path, exist_ok=True)
        log_path = os.path.join(adapter_path, "train.log")

        if engine == "llama.cpp":
            # Experimental pure GGUF training using llama-finetune
            if not model_name.endswith(".gguf"):
                return json.dumps({"status": "ERROR", "message": "llama.cpp engine requires a .gguf model_name path."})
                
            cmd = [
                "llama-finetune",
                "--model", model_name,
                "--train-data", dataset_path,
                "--lora-out", os.path.join(adapter_path, "adapter.gguf"),
                "--epochs", "2"
            ]
        else:
            if importlib.util.find_spec("mlx_lm") is None:
                return json.dumps({"status": "NOT_AVAILABLE", "message": "mlx_lm is not installed in this environment. Run: pip install mlx-lm"})
            
            
            # The default approach: MLX for training, then convert to gguf via llama.cpp
            run_script_path = os.path.join(adapter_path, "run_train.sh")
            if fuse:
                problem = lora_training.ensure_converter()
                if problem:
                    return json.dumps({"status": "NOT_AVAILABLE", "message": f"{problem} Or pass fuse=False to train the adapter only."})
                restart_label = lora_training.find_launchd_label(target_gguf_path) if target_gguf_path else None
                fuse_cmd = lora_training.build_fuse_script(
                    python=sys.executable,
                    model_name=model_name,
                    adapter_path=adapter_path,
                    converter_src=str(lora_training.llama_cpp_src()),
                    target_gguf_path=target_gguf_path,
                    label=restart_label,
                )
            else:
                fuse_cmd = f"""
echo "Training complete. Adapter saved to {adapter_path} (fusion skipped)."
"""

            with open(run_script_path, "w") as f:
                f.write(f'''#!/bin/bash
set -e
echo "Starting MLX LoRA training..."
{sys.executable} -m mlx_lm.lora \
    --model {model_name} \
    --data {run_root} \
    --adapter-path {adapter_path} \
    --train \
    --iters {iters}
{fuse_cmd}
''')
            os.chmod(run_script_path, 0o755)
            cmd = ["bash", run_script_path]

        with open(log_path, "a") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, start_new_session=True)

        return json.dumps({
            "status": "TRAINING_STARTED",
            "dataset": dataset_path,
            "data_dir": run_root,
            "adapter_path": adapter_path,
            "base_model": model_name,
            "pid": proc.pid,
            "log": log_path,
            "restart_agent": restart_label,
            "converter_commit": lora_training.converter_commit() if fuse and engine != "llama.cpp" else None,
            "command": " ".join(cmd),
            "note": f"Training runs in a detached background process. Monitor with: tail -f {log_path}",
        }, indent=2)
    except Exception as e:
        logger.error(f"train tool failed: {e}")
        return json.dumps({"status": "ERROR", "message": str(e)})

@mcp.prompt(
    name="temperature",
    description="View or change Jev's global MLX calibration temperature (which affects all evaluations)."
)
def temperature_prompt() -> str:
    return "I want to manage Jev's global calibration temperature. First, use `temperature` to view the current temperature. Then ask me if I want to keep it, manually set a new value, or reset it to the default 1.0."

@mcp.prompt(
    name="train",
    description="Train a local MLX LoRA adapter on your Apple Silicon GPU using any dataset format to customize Jev."
)
def train_prompt() -> str:
    return "I want to fine-tune Jev to my codebase. If I provide a dataset in CSV, Markdown, or another raw format, you MUST first convert it into the strict `.jsonl` schema required by Jev and save it locally. (If I don't have data, use `generate-data` to generate it). Once the `.jsonl` file is ready, pass its path to the `train` tool to instantly train a custom adapter on the GPU."

@mcp.tool(
    name="read-file",
    description="File Context Filter: Reads a file locally and uses the Dual-Engine logit cascade to mathematically evaluate if the file is relevant to the current task. Prevents context bloat by blocking irrelevant files or filtering them down to only the relevant chunks.",
)
def jev_read_file(
    file_path: str = Field(
        description="The absolute path to the file you want to read."
    ),
    task_description: str = Field(
        description="A description of the current task or goal. Used to filter the file content."
    ),
    filter_by_chunk: bool = Field(
        default=True,
        description="If True, the tool will chunk the file and return only the relevant chunks. If False, it evaluates the file as a whole and either returns the entire file or blocks it completely."
    )
    # TODO: Expose `confidence_threshold: float = 0.50` parameter here later to allow dynamic strictness
) -> str:
    import os
    import json
    from jev_mcp.security import is_safe_path

    if not is_safe_path(file_path):
        return json.dumps({"status": "BLOCKED", "message": f"Access denied: '{file_path}' resolves to a sensitive system or credential location."})

    file_path = os.path.expanduser(file_path)

    if not os.path.exists(file_path):
        return json.dumps({"status": "ERROR", "message": f"File not found: {file_path}"})
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": f"Failed to read file: {e}"})
        
    if not file_content.strip():
        return json.dumps({"status": "ERROR", "message": "File is empty."})
        
    if filter_by_chunk:
        logger.info(f"AST Chunk-filtering {file_path} for task: {task_description}")
        from jev_mcp.chunker import SemanticChunker
        from jev_mcp.routing_provider import RoutingProvider
        
        chunker = SemanticChunker()
        file_chunks = chunker.get_semantic_chunks(file_path, file_content)
        
        provider = RoutingProvider()
        kept_chunks = []
        dropped = 0
        
        for i, chunk in enumerate(file_chunks):
            q = NoulQuestion(
                key=f"chunk_{i}",
                prompt=f"Does this specific code block contain logic or variables highly relevant to the task: '{task_description}'?"
            )
            res = provider.evaluate_batch(chunk, [q])
            true_prob = res.get(f"chunk_{i}", {}).get("probabilities", {}).get("true", 0.0)
            
            if true_prob >= 0.50:
                kept_chunks.append(chunk)
            else:
                dropped += 1
                
        if not kept_chunks:
             return json.dumps({"status": "BLOCKED", "message": f"Jev evaluated {file_path} and found 0 relevant chunks for the task. The file has been blocked to save context window."})
             
        raw_content = "\n\n".join(kept_chunks)
        safe_content, was_truncated = _safe_truncate(raw_content)
        
        msg = f"File {file_path} filtered successfully using AST chunking. Dropped {dropped} irrelevant chunks."
        if was_truncated:
            msg += " WARNING: Output was truncated to protect the context window."
            
        return json.dumps({
            "status": "SUCCESS", 
            "message": msg,
            "content": safe_content
        })
        
    else:
        logger.info(f"Whole-file filtering {file_path} for task: {task_description}")
        from jev_mcp.routing_provider import RoutingProvider
        provider = RoutingProvider()
        
        eval_content = file_content[:16000] 
        
        q = NoulQuestion(
            key="file_eval",
            prompt=f"Is this file highly relevant to achieving the goal: '{task_description}'?"
        )
        res = provider.evaluate_batch(eval_content, [q])
        true_prob = res.get("file_eval", {}).get("probabilities", {}).get("true", 0.0)
        
        if true_prob >= 0.50:
            safe_content, was_truncated = _safe_truncate(file_content)
            msg = f"File passed relevance check ({round(true_prob * 100, 1)}% confident)."
            if was_truncated:
                msg += " WARNING: Output was truncated to protect the context window. Use filter_by_chunk=True to extract only relevant parts."
            
            return json.dumps({
                "status": "SUCCESS",
                "message": msg,
                "content": safe_content
            })
        else:
            return json.dumps({
                "status": "BLOCKED",
                "message": f"Jev determined this file is irrelevant to the task ({round(true_prob * 100, 1)}% confident). Blocked to save context."
            })


@mcp.tool(
    name="compact",
    description="Context Compressor: Slices massive state contexts into chunks and uses Confidence-Gated Logits to keep only the verbatim chunks strictly relevant to the user's goal. Eliminates hallucination risk of generative summarization.",
)
def jev_compact_context(
    state: Dict[str, Any] = Field(
        description="The massive JSON state or text to compress."
    ),
    goal: str = Field(
        description="The user's current goal or the objective the context is needed for."
    ),
    confidence_threshold: float = Field(
        default=0.3,
        description="The relevance score (0-1) a chunk needs to be kept. The fast engine scores relevant chunks roughly 0.3-0.9 and irrelevant ones below 0.35, so values above about 0.4 discard relevant context."
    )
) -> str:
    import json
    import textwrap
    logger.info("Starting Confidence-Gated Context Compaction...")

    state_str = state.get("text", "") if isinstance(state, dict) and "text" in state else json.dumps(state)
    if not state_str.strip():
        return json.dumps({"status": "ERROR", "message": "State is empty."})
        
    # Semantic chunking (simple paragraph/block chunking for now)
    # Using double newline or single newline if too long
    if isinstance(state, dict) and state and "text" not in state:
        # One chunk per top-level key, so unrelated fields are scored separately.
        raw_chunks = [f"{k}: {v if isinstance(v, str) else json.dumps(v)}" for k, v in state.items()]
    else:
        raw_chunks = [c.strip() for c in state_str.split("\n\n") if c.strip()]
        if not raw_chunks:
            raw_chunks = [c.strip() for c in state_str.split("\n") if c.strip()]
        
    chunks = []
    for c in raw_chunks:
        if len(c) > 2000:
            chunks.extend(textwrap.wrap(c, 2000))
        else:
            chunks.append(c)

    kept_chunks = []
    total_chunks = len(chunks)
    dropped_chunks = 0
    chunk_scores: List[Optional[float]] = []
    
    # We will evaluate these in batches if needed, but for now sequentially 
    # to avoid context overflow on the provider.
    for i, chunk in enumerate(chunks):
        q = NoulQuestion(
            key=f"chunk_{i}",
            prompt=f"Is the information in this text snippet directly relevant to achieving the goal: '{goal}'?"
        )
        
        try:
            # We force the fast provider directly to save time and prevent escalation
            # Since we just want a fast filter
            if hasattr(provider, "fast_provider"):
                res = provider.fast_provider.evaluate_batch({"text": chunk}, [q])
            else:
                res = provider.evaluate_batch({"text": chunk}, [q])
                
            chunk_res = res.get(q.key, {})
            score = chunk_res.get("noul", 0.0)
            chunk_scores.append(round(score, 3))
            
            if score >= confidence_threshold:
                kept_chunks.append(chunk)
            else:
                dropped_chunks += 1
                
        except Exception as e:
            logger.warning(f"Failed to evaluate chunk {i}: {e}. Keeping chunk by default.")
            kept_chunks.append(chunk)
            chunk_scores.append(None)

    raw_text = "\n\n".join(kept_chunks)
    safe_text, was_truncated = _safe_truncate(raw_text)
    compression_ratio = (1.0 - (len(kept_chunks) / total_chunks)) * 100 if total_chunks > 0 else 0
    
    status_msg = "COMPACTION_COMPLETE"
    if was_truncated:
        status_msg += " (TRUNCATED TO PROTECT CONTEXT WINDOW)"
        
    return json.dumps({
        "status": status_msg,
        "original_chunks": total_chunks,
        "kept_chunks": len(kept_chunks),
        "dropped_chunks": dropped_chunks,
        "compression_ratio": f"{compression_ratio:.1f}%",
        "chunk_scores": chunk_scores,
        "compacted_state": safe_text
    }, indent=2)

@mcp.tool(
    name="scan-repo",
    description="Repository Scanner: Walks a codebase, uses Tree-sitter to break code into semantic blocks (functions/classes), and mathematically filters them using the Dual-Engine cascade to find only the code relevant to the task.",
)
def jev_scan_repo(
    directory_path: str = Field(description="The absolute path to the directory to scan."),
    task_description: str = Field(description="The goal or bug description used to filter the codebase.")
) -> str:
    import os
    import json
    from jev_mcp.scanner import walk_repository, generate_repo_map
    from jev_mcp.chunker import SemanticChunker
    from jev_mcp.routing_provider import RoutingProvider
    from jev_mcp.security import is_safe_path

    if not is_safe_path(directory_path):
        return json.dumps({"status": "BLOCKED", "message": f"Access denied: '{directory_path}' resolves to a sensitive system or credential location."})

    directory_path = os.path.expanduser(directory_path)

    if not os.path.isdir(directory_path):
        return json.dumps({"status": "ERROR", "message": f"Directory not found: {directory_path}"})
        
    logger.info(f"Scanning repository at {directory_path} for task: {task_description}")
    
    files = walk_repository(directory_path)
    if not files:
        return json.dumps({"status": "BLOCKED", "message": "No valid files found in directory."})
        
    target_files = files
    
    if len(files) > 20:
        logger.info(f"Repository is massive ({len(files)} files). Escalating to Smart Model for Surgical Pointing...")
        repo_map = generate_repo_map(files, directory_path)
        
        prompt = f"Given the following task: '{task_description}'\\n\\nHere is the repository structure:\\n{repo_map}\\n\\nRespond with ONLY a comma-separated list of the 5-10 file paths most likely to contain the code needed for this task. Do not include any other text."
        
        try:
            content = _chat_completion(
                prompt,
                temperature=0.0,
                max_tokens=200,
                timeout=120,
            )

            suggested_paths = [p.strip() for p in content.replace("\\n", ",").split(",") if p.strip()]

            target_files = []
            for sp in suggested_paths:
                clean_sp = sp.lstrip("- ").strip()
                abs_path = os.path.abspath(os.path.join(directory_path, clean_sp))
                if abs_path in files:
                    target_files.append(abs_path)

            if not target_files:
                logger.warning("Smart model failed to select valid paths. Falling back to first 20 files.")
                target_files = files[:20]
        except Exception as e:
            logger.warning(f"Surgical pointing failed: {e}. Falling back to first 20 files.")
            target_files = files[:20]

    chunker = SemanticChunker()
    all_chunks = []
    
    for fpath in target_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
                
            file_chunks = chunker.get_semantic_chunks(fpath, code)
            for c in file_chunks:
                rel_path = os.path.relpath(fpath, directory_path)
                contextualized_chunk = f"### FILE: {rel_path}\\n{c}"
                all_chunks.append(contextualized_chunk)
        except Exception:
            pass

    if not all_chunks:
        return json.dumps({"status": "BLOCKED", "message": "Failed to extract chunks from repository."})

    provider = RoutingProvider()
    kept_chunks = []
    dropped = 0
    
    from concurrent.futures import ThreadPoolExecutor
    
    def eval_chunk(args):
        i, chunk = args
        q = NoulQuestion(
            key=f"chunk_{i}",
            prompt=f"Does this specific code block contain logic or variables highly relevant to the task: '{task_description}'?"
        )
        res = provider.evaluate_batch(chunk, [q])
        true_prob = res.get(f"chunk_{i}", {}).get("probabilities", {}).get("true", 0.0)
        return (chunk, true_prob >= 0.50)

    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk, keep in executor.map(eval_chunk, enumerate(all_chunks)):
            if keep:
                kept_chunks.append(chunk)
            else:
                dropped += 1
            
    if not kept_chunks:
        return json.dumps({"status": "BLOCKED", "message": f"Scanned {len(target_files)} files and {len(all_chunks)} semantic chunks. Jev determined 0 chunks were relevant to the task. Blocked to save context window."})
        
    raw_content = "\n\n".join(kept_chunks)
    safe_content, was_truncated = _safe_truncate(raw_content)
    
    msg = f"Surgically scanned {len(target_files)} files. Extracted {len(kept_chunks)} highly relevant chunks. Dropped {dropped} noisy chunks."
    if was_truncated:
        msg += " WARNING: Output was truncated to protect the context window. Refine task_description or use read-file."
        
    return json.dumps({
        "status": "SUCCESS",
        "message": msg,
        "content": safe_content
    })



@mcp.prompt(
    name="scan-repo",
    description="Ask Jev to surgically scan the current workspace repository to find and extract the AST chunks relevant to a specific task."
)
def scan_repo_prompt() -> str:
    return "I want to scan this repository for code relevant to my current task. Please ask me for the target directory path (if not the root) and the specific bug or feature I am looking for. Then execute the `scan-repo` tool with those arguments and report back the mathematically extracted AST chunks."

@mcp.prompt(
    name="read-file",
    description="Ask Jev to read a specific file and mathematically extract only the AST chunks relevant to a task, dropping irrelevant noise."
)
def read_file_prompt() -> str:
    return "I want to read a file, but I only want the parts relevant to my current task. Please ask me for the file path and my objective. Then execute the `read-file` tool with filter_by_chunk=True and report back the extracted AST blocks."

def main():
    import sys
    import argparse
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup-gas":
        parser = argparse.ArgumentParser(prog="jev-mcp setup-gas")
        parser.add_argument("--webhook-url", type=str, help="The Webhook URL to route emails to.")
        parser.add_argument("--api-key", type=str, help="The secure API Key to protect the webhook.")
        
        # Parse arguments starting from the 3rd argument since sys.argv[1] is 'setup-gas'
        args = parser.parse_args(sys.argv[2:])
        
        from jev_mcp.email_triage.gas_setup import setup_gas_workflow
        setup_gas_workflow(webhook_url=args.webhook_url, api_key=args.api_key)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "serve-webhook":
        parser = argparse.ArgumentParser(prog="jev-mcp serve-webhook")
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Interface to bind. The default is local only.")
        parser.add_argument("--port", type=int, default=8090, help="Port for POST /api/v1/triage/email.")
        args = parser.parse_args(sys.argv[2:])

        credential_vars = ("JEV_MCP_API_KEY", "JEV_MCP_JWT_SECRET", "JEV_MCP_JWKS_URL")
        if not any(os.environ.get(name) for name in credential_vars):
            sys.exit(f"Refusing to start: set one of {', '.join(credential_vars)} so the webhook can authenticate callers.")

        import uvicorn
        from jev_mcp.email_triage.api import app
        uvicorn.run(app, host=args.host, port=args.port)
        return

    start_model_sync()
    mcp.run(transport='stdio')

if __name__ == '__main__':
    main()
