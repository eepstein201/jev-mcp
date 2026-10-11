"""MCP tool families, split out of server.py.

Each module registers its tools on the shared `app.mcp` instance at import
time; server.py imports them so `python -m jev_mcp.server` registers
everything before `main()` runs. These modules must never import
`jev_mcp.server` itself.
"""
