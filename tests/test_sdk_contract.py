"""Pin the SDK contracts the SeedingMiddleware depends on.

Run in a subprocess on purpose: conftest stubs the mcp SDK for the unit
suite, so the REAL installed package is only reachable from a fresh
interpreter. Each test asserts one contract of the installed SDK; if
`uv lock --upgrade` ever moves to a version that breaks one, this file
fails first and names the contract, before the stdio server misbehaves at
a real handshake. (ServerMiddleware is marked provisional upstream — these
are the tripwires.)
"""

import subprocess
import sys

CONTRACTS = {
    "middleware_kwarg": """
import inspect
from mcp.server.mcpserver import MCPServer
assert "middleware" in inspect.signature(MCPServer.__init__).parameters, (
    "MCPServer no longer takes middleware=; app.py wiring needs the new mechanism")
""",
    "snake_case_client_info": """
from mcp import types
assert "client_info" in types.InitializeRequestParams.model_fields, (
    "InitializeRequestParams no longer exposes client_info; _seed reads it")
assert "name" in types.Implementation.model_fields, "Implementation no longer has .name"
""",
    "middleware_runs_on_notifications": """
import inspect
from mcp.server.runner import ServerRunner
assert hasattr(ServerRunner, "_compose_server_middleware")
assert "_compose_server_middleware" in inspect.getsource(ServerRunner._on_notify), (
    "notifications no longer run the middleware chain; initialized would not reach us")
assert "_compose_server_middleware" in inspect.getsource(ServerRunner._on_request)
""",
    "context_has_method_and_session": """
from mcp.server.context import ServerRequestContext
hints = {f for f in ServerRequestContext.__annotations__ if not f.startswith("_")}
assert {"method", "session"} <= hints, "ServerRequestContext no longer carries method/session"
""",
}


def _run_contract(body: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"SDK contract broken:\n{proc.stderr}"


def test_mcpserver_accepts_middleware_kwarg():
    _run_contract(CONTRACTS["middleware_kwarg"])


def test_initialize_params_expose_snake_case_client_info():
    _run_contract(CONTRACTS["snake_case_client_info"])


def test_runner_composes_middleware_for_requests_and_notifications():
    _run_contract(CONTRACTS["middleware_runs_on_notifications"])


def test_request_context_carries_method_and_session():
    _run_contract(CONTRACTS["context_has_method_and_session"])
