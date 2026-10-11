"""Shared application state for the jev-mcp server.

Tool modules must NOT import `jev_mcp.server`: clients launch the server as
`python -m jev_mcp.server`, so that module runs as `__main__` and importing it
from elsewhere would build a second `mcp` instance. Everything the tool
families need at call time lives here instead, and they reference it as
`app.X` so tests can patch `jev_mcp.app.X`.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
import json
import logging
from typing import Any, Dict, List, Optional

# Must immediately hijack stdout to prevent ANY random library (like torch)
# from printing warnings that would corrupt the JSON-RPC stream.

from mcp.server.mcpserver import MCPServer
from jev_mcp import model_catalog, router_config
from jev_mcp.router_config import apply_model_updates, load_router_config, pending_model_updates, save_router_config
from jev_mcp.provider import post_json
from jev_mcp.linter import DecisionPreflightLinter
from jev_mcp.routing_provider import RoutingProvider

# Setup global logging to ~/.jev/jev.log
log_dir = os.path.expanduser("~/.jev")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "jev.log"),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("jev-mcp")

def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Aggressively extract the JSON object/array to ignore leading/trailing conversational chatter
    obj_start = text.find("{")
    arr_start = text.find("[")

    start_idx = -1
    end_char = ""

    if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
        start_idx = obj_start
        end_idx = text.rfind("}")
    elif arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        start_idx = arr_start
        end_idx = text.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    return text.strip()

def _safe_truncate(content: str, max_length: int = 24000) -> tuple[str, bool]:
    """Truncates content safely to prevent MCP Client JSON-breakage."""
    if len(content) <= max_length:
        return content, False

    truncated_msg = f"\n\n... [TRUNCATED: Content exceeded {max_length} characters. Please refine your search or read specific files for more detail.] ..."
    return content[:max_length - len(truncated_msg)] + truncated_msg, True

def _chat_completion(
    user: str,
    system: Optional[str] = None,
    *,
    port_env: str = "JEV_SMART_PORT",
    default_port: str = "8081",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    response_format: Optional[Dict[str, str]] = None,
    timeout: int = 45,
) -> str:
    """Single entry point for local mlx_lm chat-completions calls.

    Returns the raw message content; raises on transport/API errors so callers
    keep their existing try/except contracts.
    """
    messages: List[Dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    port = os.getenv(port_env, default_port)
    if not port.isdigit():
        raise ValueError(f"Invalid {port_env} value: {port!r} (expected a port number)")
    res = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=timeout)
    return res["choices"][0]["message"]["content"]

def _chat_completion_json(**kwargs: Any) -> Any:
    """_chat_completion + fence stripping + JSON parsing."""
    return json.loads(_clean_json(_chat_completion(**kwargs)))

# Initialize MCP and Provider
mcp = MCPServer("jev-mcp")

# The hybrid router, not a bare DaemonProvider: a bare one falls back to
# JEV_DAEMON_PORT (8080), which is the Kev daemon in the dual-engine setup.
provider = RoutingProvider()
linter = DecisionPreflightLinter()

# Platt scaling is cross-validated when every class has at least this many rows.
PLATT_CV_FOLDS = 5

def call_fast_autofixer(state, questions, errors):
    """Hits the persistent local llama-server to fix prompts in <300ms using Constrained Decoding."""
    system_prompt = (
        "You are an expert prompt engineer for a fast-decision classification model. "
        "The user submitted questions that violate the system's framing rules. "
        "Your job is to FIX the questions based on the provided Linter Errors. "
        "RULES:\n"
        "1. Do NOT change the core intent of the questions.\n"
        "2. Fix overlapping options, add fallback options (e.g. 'Unknown', 'None'), and balance option lengths.\n"
        "3. Remove any conversational boilerplate (e.g. 'You are an expert').\n"
        "4. If a question is generative ('Summarize...'), rewrite it into a multiple choice (choice) or yes/no (noul) question.\n"
        "5. Output ONLY a valid JSON object.\n"
        'Format: {"fixed_questions": [{"key": "...", "prompt": "...", "type": "choice|noul|score", "options": ["..."], "labels": ["..."]}]}'
    )
    user_prompt = f"STATE/CONTEXT:\n{json.dumps(state)[:1000]}...\n\nBROKEN QUESTIONS:\n{json.dumps(questions, indent=2)}\n\nLINTER ERRORS:\n{json.dumps(errors, indent=2)}\n\nReturn ONLY the JSON object."

    # Constrained Decoding guarantees JSON; no regex needed.
    data = _chat_completion_json(
        user=user_prompt,
        system=system_prompt,
        port_env="JEV_FAST_PORT",
        default_port="8080",
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return data.get("fixed_questions", [])

_has_notified_temp = False
_has_notified_models = False

SKIP_MODEL_SYNC_ENV = "JEV_SKIP_MODEL_SYNC"


def model_update_notice(config: Dict[str, Any], updates: Dict[str, Dict[str, str]]) -> str | None:
    """What to tell the user about newer models, and the questions to put to them."""
    if not updates:
        return None
    listed = "; ".join(f"{bucket}: {u['current']} -> {u['newer']} (released {u['released']})" for bucket, u in sorted(updates.items()))
    notice = (
        f"NOTICE: Newer models are available for these router buckets: {listed}. "
        "The current models are kept until the user decides. Ask the user whether to switch; "
        "if yes, call manage-router with action=apply_updates."
    )
    if config.get("auto_update") is None:
        notice += (
            " Also ask whether Jev should move buckets to the newest model in the same family automatically "
            "from now on, and record the answer with manage-router action=set_auto_update, enabled=true or false."
        )
    return notice


def get_model_notice() -> str | None:
    """The model update notice, at most once per server process."""
    global _has_notified_models
    if _has_notified_models:
        return None
    try:
        config = load_router_config()
        notice = model_update_notice(config, pending_model_updates(config))
        if notice:
            _has_notified_models = True
        return notice
    except Exception as e:
        logger.warning(f"Could not check for newer router models: {e}")
        return None


def _sync_models() -> None:
    """Refresh the model catalog; move buckets to newer models only if the user turned that on."""
    try:
        model_catalog.refresh()
        if not os.path.exists(router_config.ROUTER_CONFIG_PATH):
            return
        config = load_router_config()
        if config.get("auto_update") is not True:
            return
        updates = pending_model_updates(config)
        if updates:
            save_router_config(apply_model_updates(config, updates))
            logger.info(f"Router buckets updated automatically: {updates}")
    except Exception as e:
        logger.warning(f"Model sync failed: {e}")


def start_model_sync() -> None:
    """Run the model sync in the background so server start-up never waits on the network."""
    if os.getenv(SKIP_MODEL_SYNC_ENV, ""):
        return
    import threading
    threading.Thread(target=_sync_models, name="jev-model-sync", daemon=True).start()


def get_temp_notice() -> str | None:
    global _has_notified_temp
    if _has_notified_temp:
        return None

    try:
        config = load_router_config()
        temp = config.get("fitted_temperature", 1.0)
        if temp != 1.0:
            _has_notified_temp = True
            updated_at = config.get("temperature_updated_at", "an unknown time")
            return f"NOTICE: Jev is running with a custom calibrated temperature of {temp} (set on {updated_at}). This globally affects ALL evaluations. You can ask me to reset it to default (1.0) at any time using the /jev-mcp:temperature command or temperature tool."
    except Exception:
        pass
    return None
