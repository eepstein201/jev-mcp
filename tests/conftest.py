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
