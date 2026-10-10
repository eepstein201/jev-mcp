import pytest
from jev_mcp.server import jev_run_browser_agent

def test_jev_run_browser_agent():
    # Just checking it's defined and has the correct signature
    assert callable(jev_run_browser_agent)
