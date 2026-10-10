<!-- Generated: 2026-10-10 | Files scanned: pyproject.toml, Makefile, ci.yml, jev_mac_manager.sh | Token estimate: ~450 -->
# Dependencies

## Python (pyproject.toml, py>=3.11)
mcp[cli]>=2.2,<3 · pydantic>=2 · scikit-learn · tree-sitter(+python/js/ts)
browser-harness · h2 · google-api-python-client · google-auth-oauthlib
Imported but not declared: fastapi (email_triage/api.py, auth.py)

## Runtime engines (not imports)
llama.cpp llama-server ×2 (8080 Kev-0.8B, 8081 Qwen-2.5-7B) · mlx_lm (lora/fuse) · launchd
Submodule: jev-ultrafast (github.com/browser-use/jev-ultrafast)

## Env vars
JEV_FAST_PORT JEV_SMART_PORT JEV_DAEMON_PORT JEV_FAST_ENGINE JEV_BATCH_SIZE
JEV_MCP_API_KEY JEV_MCP_WEBHOOK_URL

## Build / CI
make start|stop|test|lint|format|train · ci.yml (macos-latest): mypy src/jev_mcp/ + pytest --cov-fail-under=85
Commit gate: AGENTS.md (95% per file, 85% overall)

## External integrations
Google Apps Script (examples/gas_triage.js) · Slack (HITL) · MCP clients: Antigravity, Claude Code/Desktop, OpenCode
