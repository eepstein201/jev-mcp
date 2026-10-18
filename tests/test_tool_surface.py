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
# jev_list_files is newer than the rename, so it has no old name.
EXPECTED_TOOLS = frozenset(RENAMED_TOOLS.values()) | {"jev_list_files"}

# Each tool's behaviour hints in the order read-only, destructive, idempotent, open-world.
# A tool is read-only only if its code path has no config save, file write or process
# launch. Destructive means it can discard something that existed. Open-world means it
# reaches beyond this machine (model downloads, a live website, Google's API).
ANNOTATIONS: dict[str, tuple[bool, bool, bool, bool]] = {
    "jev_read_file": (True, False, True, False),
    "jev_scan_repo": (True, False, True, False),
    "jev_list_files": (True, False, True, False),
    "jev_compact_context": (True, False, True, False),
    "jev_evaluate_batch": (True, False, True, False),
    "jev_explain_decision": (True, False, True, False),
    "jev_generate_synthetic_dataset": (False, False, False, False),
    "jev_optimize_prompt": (True, False, False, False),
    "jev_determine_best_model": (True, False, True, False),
    "jev_agent_handoff": (True, False, True, False),
    "jev_triage_email": (True, False, True, False),
    "jev_calibrate_threshold": (False, False, False, False),
    "jev_manage_temperature": (False, False, True, False),
    "jev_manage_router_config": (False, True, False, False),
    "jev_configure_triage_labels": (False, True, True, False),
    "jev_train_lora": (False, True, False, True),
    "jev_run_browser_agent": (False, True, False, True),
    "jev_setup_gas_workflow": (False, False, False, True),
}

HINT_FIELDS = (
    "read_only_hint",
    "destructive_hint",
    "idempotent_hint",
    "open_world_hint",
)

# A file opened with any of these mode characters can modify the file.
WRITE_MODE_CHARS = frozenset("wax+")


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


def _tool_body(name: str) -> str:
    """The source of the function the named tool decorator is applied to."""
    for path in sorted(TOOLS_DIR.glob("*.py")):
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool"
                ):
                    continue
                keywords = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
                if "name" in keywords and ast.literal_eval(keywords["name"]) == name:
                    return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"no tool named {name} is registered")


def _hints(node: ast.expr) -> tuple[bool, ...]:
    """The four hint keyword values from a ToolAnnotations(...) call node."""
    assert isinstance(node, ast.Call)
    assert isinstance(node.func, ast.Name) and node.func.id == "ToolAnnotations"
    keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
    return tuple(ast.literal_eval(keywords[field]) for field in HINT_FIELDS)


def _body_writes(name: str) -> bool:
    """Whether the tool's body saves config, writes a file, or starts a process."""
    body = _tool_body(name)
    if any(
        pattern in body
        for pattern in ("save_router_config", "write_json_atomic", "subprocess.")
    ):
        return True
    for node in ast.walk(ast.parse(body)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ):
            mode = node.args[1] if len(node.args) >= 2 else None
            mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), mode)
            if (
                isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and WRITE_MODE_CHARS & set(mode.value)
            ):
                return True
    return False


def test_the_server_registers_exactly_the_eighteen_tools():
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


def test_every_tool_has_a_title_and_annotations():
    missing = [
        name
        for name, kw in registered_tools().items()
        if "title" not in kw or "annotations" not in kw
    ]
    assert missing == []


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_each_tools_hints_match_what_its_code_does(name):
    assert _hints(registered_tools()[name]["annotations"]) == ANNOTATIONS[name]


def test_no_read_only_tool_saves_config_writes_a_file_or_starts_a_process():
    for name, (read_only, *_rest) in ANNOTATIONS.items():
        if read_only:
            assert not _body_writes(name), name
