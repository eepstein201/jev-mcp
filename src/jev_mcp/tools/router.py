"""The jev_manage_router_config, jev_determine_best_model, and jev_agent_handoff tools."""

import json
from typing import Any, Dict, Literal, Optional

from mcp.types import ToolAnnotations
from pydantic import Field

from jev_mcp import app, router_config
from jev_mcp.app import json_errors, logger, model_update_notice, mcp
from jev_mcp.provider import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    escalation_failures,
    mark_degraded,
)
from jev_mcp.router_config import apply_model_updates, pending_model_updates


def _error(message: str) -> str:
    return json.dumps({"status": "ERROR", "message": message})


def _saved(message: str) -> str:
    return json.dumps({"status": "SAVED", "message": message})


@mcp.tool(
    name="jev_manage_router_config",
    title="View or change model routing",
    description="Interactive configuration manager for Jev's Model Router. Use this to view, add, or remove custom bucket models and exception rules. view_all also reports newer models available for each bucket; apply_updates switches to them; set_auto_update (enabled=true/false) records whether Jev should do that automatically at start-up.",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=False,
    ),
)
@json_errors
def jev_manage_router_config(
    action: Literal[
        "view_all",
        "add_rule",
        "remove_rule",
        "set_bucket",
        "apply_updates",
        "set_auto_update",
    ] = Field(description="The action to perform."),
    bucket_id: Optional[str] = Field(
        default=None, description="The bucket (b1, b2, b3, b4) to modify."
    ),
    model_name: Optional[str] = Field(
        default=None, description="The model name to assign to the bucket."
    ),
    condition: Optional[str] = Field(
        default=None,
        description="The natural language condition for the rule (e.g., 'Touches > 10 files').",
    ),
    target: Optional[str] = Field(
        default=None, description="The target bucket or model if the condition is met."
    ),
    rule_id: Optional[int] = Field(
        default=None, description="The index of the rule to remove."
    ),
    enabled: Optional[bool] = None,
) -> str:
    config = app.load_router_config()

    if action == "view_all":
        updates = pending_model_updates(config)
        if not updates:
            return json.dumps(config, indent=2)
        return json.dumps(
            {
                **config,
                "available_model_updates": updates,
                "_model_update_notice": model_update_notice(config, updates),
            },
            indent=2,
        )

    if action == "apply_updates":
        updates = pending_model_updates(config)
        if not updates:
            return json.dumps(
                {
                    "status": "UP_TO_DATE",
                    "message": "Every bucket already holds the newest model in its family.",
                }
            )
        app.save_router_config(
            {**apply_model_updates(config, updates), "buckets_source": "user"}
        )
        return json.dumps(
            {
                "status": "UPDATED",
                "updated": {bucket: u["newer"] for bucket, u in updates.items()},
            },
            indent=2,
        )

    if action == "set_auto_update":
        if not isinstance(enabled, bool):
            return _error("enabled (true or false) required.")
        updates = pending_model_updates(config) if enabled else {}
        new_config = apply_model_updates(config, updates) if updates else config
        stamped = {**new_config, "buckets_source": "user"} if enabled else new_config
        app.save_router_config({**stamped, "auto_update": enabled})
        return json.dumps(
            {
                "status": "SAVED",
                "auto_update": enabled,
                "updated": {bucket: u["newer"] for bucket, u in updates.items()},
            },
            indent=2,
        )

    if action == "set_bucket":
        if not bucket_id or not model_name:
            return _error("bucket_id and model_name required.")
        buckets = {**config["buckets"], bucket_id: model_name}
        app.save_router_config({**config, "buckets": buckets, "buckets_source": "user"})
        return _saved(f"Bucket {bucket_id} successfully mapped to {model_name}.")

    if action == "add_rule":
        if not condition or not target:
            return _error("condition and target required.")
        from jev_mcp.security import sanitize_payload

        rule = {"condition": sanitize_payload(condition), "target": target}
        app.save_router_config({**config, "rules": [*config["rules"], rule]})
        return _saved(f"Rule added. If '{condition}', route to '{target}'.")

    if action == "remove_rule":
        if rule_id is None or rule_id < 0 or rule_id >= len(config["rules"]):
            return _error("Valid rule_id required.")
        removed = config["rules"][rule_id]
        remaining = [
            rule for index, rule in enumerate(config["rules"]) if index != rule_id
        ]
        app.save_router_config({**config, "rules": remaining})
        return _saved(f"Rule removed: {removed}")

    return _error(f"Unknown action: {action}")


@mcp.tool(
    name="jev_determine_best_model",
    title="Recommend a model for a task",
    description="Complexity Index Router: Analyzes a task description using the Hybrid local engine and custom rules to determine the mathematically optimal LLM model to use.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
@json_errors
def jev_determine_best_model(
    task_description: str = Field(
        description="The prompt or goal the CLI is about to execute."
    ),
    estimated_tokens: Optional[int] = Field(
        default=0, description="The estimated token count of the context payload."
    ),
) -> str:
    config = app.load_router_config()

    # 1. Context Window Check
    if estimated_tokens and estimated_tokens > 100000:
        return json.dumps(
            {
                "status": "MASSIVE_CONTEXT_DETECTED",
                "warning": "Massive context detected (>100k tokens). Please ensure your client enables extra token context flags or select a specialized 1M+ context model. Provide the custom model name if you wish to override.",
                "recommended_tier": "b3",
                "recommended_model": config["buckets"].get(
                    "b3", router_config.FALLBACK_BUCKETS["b3"]
                ),
            },
            indent=2,
        )

    # 2. Evaluate Custom Rules (Hybrid Evidence-Based)
    provider = app.RoutingProvider()
    state = {"task": task_description}
    for rule in config["rules"]:
        condition = rule["condition"]
        q = NoulQuestion(
            key="rule",
            prompt=f"Does the following task meet this condition: '{condition}'?",
        )
        try:
            res = provider.evaluate_batch(state, [q])
            if res.get("rule", {}).get("noul", 0.0) > 0.85:
                return json.dumps(
                    {
                        "status": "RULE_TRIGGERED",
                        "triggered_rule": condition,
                        "recommended_target": rule["target"],
                        "model": config["buckets"].get(rule["target"], rule["target"]),
                    },
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Rule evaluation failed: {e}")

    # 3. Complexity Index (Multi-Class Logit Routing)
    complexity_q = ScoreQuestion(
        key="complexity",
        prompt="Score the complexity of this task on a scale from 1 to 4. 1 is trivial, 4 is an agentic refactor.",
        labels=["1", "2", "3", "4"],
    )

    try:
        res = provider.evaluate_batch(state, [complexity_q])
        score = res.get("complexity", {}).get("score", "1")
        probs = res.get("complexity", {}).get("probabilities", {})

        bucket = f"b{score}"
        recommended_model = config["buckets"].get(
            bucket, router_config.FALLBACK_BUCKETS["b1"]
        )

        res_dict = mark_degraded(
            {
                "status": "EVALUATED",
                "recommended_model": recommended_model,
                "complexity_score": int(score),
                "reasoning": f"Task complexity scored {score}/4. Routed to {recommended_model}.",
                "probabilities": probs,
            },
            escalation_failures(res),
        )
        notice = app.get_temp_notice(provider.smart_provider)
        if notice:
            res_dict["_system_notice"] = notice
        model_notice = app.get_model_notice()
        if model_notice:
            res_dict["_model_update_notice"] = model_notice

        return json.dumps(res_dict, indent=2)
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


@mcp.tool(
    name="jev_agent_handoff",
    title="Choose an agent to hand a task to",
    description="Uses multi-class logit routing to mathematically determine which specialized agent should take over the current task based on their full descriptions.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
@json_errors
def jev_agent_handoff(task_description: str, available_agents: Dict[str, str]) -> str:
    """
    Passes the task description and agent profiles to Jev as a ChoiceQuestion.
    """
    provider = app.RoutingProvider()

    agent_profiles = "\n".join(
        [f"- {name}: {desc}" for name, desc in available_agents.items()]
    )
    instructions = f"Which of the following specialized AI agents is best suited to execute this task based on their profiles?\nProfiles:\n{agent_profiles}"

    q = ChoiceQuestion(
        key="best_agent", prompt=instructions, options=list(available_agents.keys())
    )
    res = provider.evaluate_batch(task_description, [q])
    best_agent = res.get("best_agent", {}).get("choice")
    probs = res.get("best_agent", {}).get("probabilities", {})

    res_dict = mark_degraded(
        {
            "status": "HANDOFF_RECOMMENDATION",
            "recommended_agent": best_agent,
            "probabilities": probs,
        },
        escalation_failures(res),
    )
    notice = app.get_temp_notice(provider.smart_provider)
    if notice:
        res_dict["_system_notice"] = notice

    return json.dumps(res_dict, indent=2)
