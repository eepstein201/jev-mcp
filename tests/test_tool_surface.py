"""The tool surface a client sees: which names are registered and what the shipped texts call them."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "src" / "jev_mcp" / "tools"

RENAMED_TOOLS = {
    "read-file": "jev_read_file",
    "compact": "jev_compact_context",
    "scan-repo": "jev_scan_repo",
    "evaluate": "jev_evaluate_batch",
    "explain-decision": "jev_explain_decision",
    "generate-data": "jev_generate_synthetic_dataset",
    "calibrate": "jev_calibrate_threshold",
    "temperature": "jev_manage_temperature",
    "manage-router": "jev_manage_router_config",
    "model-router": "jev_determine_best_model",
    "handoff": "jev_agent_handoff",
    "optimize-prompt": "jev_optimize_prompt",
    "train": "jev_train_lora",
    "run-browser-agent": "jev_run_browser_agent",
    "triage_email_content": "jev_triage_email",
    "configure_triage_labels": "jev_configure_triage_labels",
    "setup_gas_workflow": "jev_setup_gas_workflow",
}
OLD_TOOL_NAMES = frozenset(RENAMED_TOOLS)
EXPECTED_TOOLS = frozenset(RENAMED_TOOLS.values())


def _tool_decorators(tree: ast.AST) -> list[ast.Call]:
    return [
        decorator
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "tool"
    ]


def registered_tools() -> dict[str, dict[str, ast.expr]]:
    """Map each tool name to its decorator's keyword arguments.

    Parsed from the source because conftest stubs the MCP SDK, so the
    registry cannot be read at runtime in this suite.
    """
    tools: dict[str, dict[str, ast.expr]] = {}
    for path in sorted(TOOLS_DIR.glob("*.py")):
        for decorator in _tool_decorators(ast.parse(path.read_text())):
            keywords = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
            name = ast.literal_eval(keywords["name"])
            assert name not in tools, f"{name} is registered twice"
            tools[name] = keywords
    return tools


def test_the_server_registers_exactly_the_seventeen_renamed_tools():
    assert set(registered_tools()) == EXPECTED_TOOLS


def test_every_tool_name_is_snake_case_with_the_service_prefix():
    assert all(
        re.fullmatch(r"jev_[a-z]+(_[a-z]+)+", name) for name in registered_tools()
    )


@pytest.mark.parametrize(
    "path", ["src/jev_mcp/server.py", "README.md", "CLAUDE.md", "AGENTS.md"]
)
def test_no_shipped_text_still_tells_a_model_to_call_an_old_tool_name(path):
    text = (ROOT / path).read_text()
    stale = [
        old
        for old in sorted(OLD_TOOL_NAMES)
        if re.search(
            rf"`{re.escape(old)}`\s+(MCP )?tool|Tool:\*\* `{re.escape(old)}`|`{re.escape(old)}\(",
            text,
        )
    ]
    assert stale == []
