import sys
from unittest.mock import MagicMock

def dummy_decorator(*args, **kwargs):
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

class DummyMCPServer:
    def __init__(self, *args, **kwargs):
        pass
    def tool(self, *args, **kwargs):
        return dummy_decorator
    def prompt(self, *args, **kwargs):
        return dummy_decorator
    def run(self, *args, **kwargs):
        pass

mock_mcpserver = MagicMock()
mock_mcpserver.MCPServer = DummyMCPServer

sys.modules['mcp.server.mcpserver'] = mock_mcpserver
sys.modules['mcp.server.fastmcp'] = MagicMock()
sys.modules['mcp.server'] = MagicMock()

import json
import os
from unittest.mock import patch

import pytest

# No test may start the real start-up model sync (a background network call and config write).
os.environ["JEV_SKIP_MODEL_SYNC"] = "1"


@pytest.fixture(autouse=True)
def isolated_jev_files(tmp_path, monkeypatch):
    """Every test gets its own router config and model catalog cache, never the ones in ~/.jev."""
    from jev_mcp import model_catalog, router_config

    monkeypatch.setattr(router_config, "ROUTER_CONFIG_PATH", str(tmp_path / "jev_home" / "router_config.json"))
    monkeypatch.setattr(model_catalog, "CACHE_PATH", str(tmp_path / "jev_home" / "model_catalog.json"))


# Tests that genuinely need real HTTP (they talk to a server they start themselves).
REAL_HTTP_MODULES = {"test_browser_agent_e2e", "test_stdio_e2e", "test_email_triage_auth_jwks"}


@pytest.fixture(autouse=True)
def no_real_daemon_http(request):
    """Make unmocked daemon calls fail as they do in CI (connection refused).

    Without this, a test that forgets to mock urlopen silently talks to whatever
    llama-server is running on the developer's machine, and its result depends on
    what the model answers. Tests that patch urlopen themselves override this.
    """
    if request.module.__name__.rsplit(".", 1)[-1] in REAL_HTTP_MODULES:
        yield
        return
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no daemon in tests")):
        yield


@pytest.fixture
def kev_http():
    """Stand-in for the Kev daemon at the HTTP boundary.

    `with kev_http(0.9) as mock_urlopen:` answers every question in a request
    with that noul score, so no test reaches a real daemon.
    """
    def _install(noul):
        def _urlopen(req, timeout=None):
            keys = json.loads(req.data)["questions"]
            response = MagicMock()
            response.read.return_value = json.dumps({"answers": {k: {"noul": noul} for k in keys}}).encode("utf-8")
            context = MagicMock()
            context.__enter__.return_value = response
            return context
        return patch("urllib.request.urlopen", side_effect=_urlopen)
    return _install
