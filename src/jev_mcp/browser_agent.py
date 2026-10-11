"""Runs the jev-ultrafast browser agent with every decision made by the local engines.

jev-ultrafast normally asks a hosted API which element to act on and a hosted LLM what to
type. Here both are replaced for the length of one run: choices go through the hybrid router
(`ultrafast_adapter.local_choose`) and field text comes from the smart daemon.
"""

import importlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jev_mcp.provider import post_json
from jev_mcp.ultrafast_adapter import local_choose

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 20
MAX_FIELD_TEXT_CHARS = 2000
TERMINAL_STATUSES = ("done", "blocked")
OPERATION_LABELS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field.",
    "SELECT": "Select an observed dropdown value.",
}

# The run swaps two functions on the jev_ultrafast module, so only one run at a time.
_RUN_LOCK = threading.Lock()


def ultrafast_path() -> Path:
    override = os.environ.get("JEV_ULTRAFAST_PATH")
    return Path(override) if override else Path(__file__).resolve().parents[2] / "jev-ultrafast"


def load_ultrafast() -> tuple[Any, Any]:
    """Import the jev-ultrafast submodule's agent and model modules."""
    path = ultrafast_path()
    if not (path / "jev_ultrafast" / "agent.py").exists():
        raise FileNotFoundError(
            f"jev-ultrafast not found at {path}. Run `git submodule update --init` in the jev-mcp "
            "checkout, or set JEV_ULTRAFAST_PATH to a jev-ultrafast checkout."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("jev_ultrafast.agent"), importlib.import_module("jev_ultrafast.model")


def choose_locally(model: Any, page: dict[str, Any], goal: str, history: list[Any]) -> dict[str, Any]:
    """Local stand-in for jev_ultrafast.model.choose: same decision shape, no hosted API."""
    started = time.perf_counter()
    _, targets, controls = model.action_space(page["actions"])
    operations = {key: OPERATION_LABELS[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")

    answers = local_choose({"page": page}, goal, history, operations, targets, controls)["answers"]
    # validate_choice raises unless the answer is one of the offered ids with sane probabilities,
    # so a confused model can never trigger an action that was not observed on the page.
    operation_answer = model.validate_choice(answers.get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    if operation in targets:
        target_answer = model.validate_choice(answers.get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities = {choice: operation_answer["probabilities"][operation]}
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "usage": {},
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


def local_field_text(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Local stand-in for jev_ultrafast.model.field_text: the smart daemon writes the field value."""
    instructions = importlib.import_module("jev_ultrafast.questions").TEXT_VALUE
    port = os.getenv("JEV_SMART_PORT", "8081")
    started = time.perf_counter()
    response = post_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(context)},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    try:
        value = json.loads(response["choices"][0]["message"]["content"])["text"]
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_FIELD_TEXT_CHARS:
            raise ValueError()
    except (ValueError, KeyError, TypeError, IndexError):
        raise ValueError("Local text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": "jev-mcp-local",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": {},
    }


def _error(message: str) -> dict[str, Any]:
    return {"status": "ERROR", "message": message}


def run_browser_agent(url: str, goal: str, max_steps: int = DEFAULT_MAX_STEPS) -> dict[str, Any]:
    """Drive a browser toward `goal`, starting at `url`, until done, blocked, or `max_steps` actions."""
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return _error("url must be an http:// or https:// address.")
    if not goal or not goal.strip():
        return _error("goal must not be empty.")
    try:
        agent_module, model_module = load_ultrafast()
    except (FileNotFoundError, ImportError) as e:
        return _error(str(e))

    with _RUN_LOCK:
        original = (agent_module.choose, agent_module.field_text)
        agent_module.choose = lambda page, task, history: choose_locally(model_module, page, task, history)
        agent_module.field_text = local_field_text
        try:
            with agent_module.Agent(url, goal) as agent:
                for snapshot in agent.run():
                    if len(snapshot["history"]) >= max_steps:
                        break
                state = agent.state
                return {
                    "status": "SUCCESS",
                    "result": state["status"] if state["status"] in TERMINAL_STATUSES else "step_limit",
                    "final_url": state["page"]["url"],
                    "elapsed_ms": state["elapsed_ms"],
                    "steps": [
                        {k: h.get(k) for k in ("step", "kind", "action", "text", "url")} for h in state["history"]
                    ],
                }
        except Exception as e:
            logger.error(f"Browser agent run failed: {e}")
            return _error(str(e))
        finally:
            agent_module.choose, agent_module.field_text = original
