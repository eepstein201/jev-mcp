"""The agent loop of the tool-usage evaluation.

The mcp-builder runner's own loop answers only the first tool call of a turn, so a model that
calls two tools in parallel gets an API 400. This version answers every tool call in the turn.
It takes the client and connection as arguments and imports neither SDK, so it is testable.
"""

import asyncio
import json
import time
import traceback
from typing import Any

MAX_TOKENS = 4096

# The tools the model may call. A usage evaluation must not launch processes, drive a browser or swap
# model files, so only the read-only tools are offered, plus the two config tools whose read actions
# the questions need (against the seeded temp home they can only write there). tests/test_usage_eval_loop.py
# keeps this list in step with the annotations table.
EVAL_TOOLS = frozenset(
    {
        "jev_read_file",
        "jev_scan_repo",
        "jev_compact_context",
        "jev_evaluate_batch",
        "jev_explain_decision",
        "jev_optimize_prompt",
        "jev_determine_best_model",
        "jev_agent_handoff",
        "jev_triage_email",
        "jev_manage_router_config",
        "jev_manage_temperature",
    }
)


async def _run_tool(connection: Any, name: str, arguments: Any) -> str:
    try:
        result = await connection.call_tool(name, arguments)
    except Exception as error:
        return f"Error executing tool {name}: {error}\n{traceback.format_exc()}"
    return _render(result)


def _render(result: Any) -> str:
    """Tool output as text: MCP content blocks by their text, other lists and dicts as JSON."""
    if isinstance(result, list) and result and all(hasattr(b, "text") for b in result):
        return "\n".join(b.text for b in result)
    return json.dumps(result) if isinstance(result, (dict, list)) else str(result)


async def agent_loop(
    client: Any,
    model: str,
    question: str,
    tools: list[dict[str, Any]],
    connection: Any,
    *,
    system: str,
) -> tuple[str | None, dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    metrics: dict[str, Any] = {}
    tools = [tool for tool in tools if tool["name"] in EVAL_TOOLS]

    while True:
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            started = time.time()
            content = await _run_tool(connection, block.name, block.input)
            entry = metrics.setdefault(block.name, {"count": 0, "durations": []})
            entry["count"] += 1
            entry["durations"].append(time.time() - started)
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
        messages.append({"role": "user", "content": results})

    text = next((b.text for b in response.content if hasattr(b, "text")), None)
    return text, metrics
