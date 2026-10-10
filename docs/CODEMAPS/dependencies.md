<!-- Generated: 2026-10-10 | Files scanned: pyproject.toml, requirements.txt, Makefile, ci.yml, jev_mac_manager.sh (as of 559930f) | Token estimate: ~450 -->
# Dependencies

## Python (pyproject.toml, py>=3.11)
mcp[cli]>=2.2,<3 · pydantic>=2 · fastapi>=0.143 · scikit-learn · tree-sitter(+python/js/ts)
browser-harness · h2 · google-api-python-client · google-auth-oauthlib
requirements.txt (pinned, used by CI): mlx · mlx-lm · mcp · pydantic · python-dotenv · kev (git, needs py>=3.12) · dev tools
Mismatch: pyproject says py>=3.11 but requirements.txt's kev needs >=3.12 (CI runs 3.12)

## Runtime engines (not imports)
llama.cpp llama-server ×2 (8080 Kev-0.8B, 8081 Qwen-2.5-7B) · mlx_lm (lora/fuse) · launchd
Submodule: jev-ultrafast (github.com/browser-use/jev-ultrafast)

## Env vars
JEV_FAST_PORT JEV_SMART_PORT JEV_DAEMON_PORT JEV_FAST_ENGINE JEV_BATCH_SIZE
JEV_MCP_API_KEY (webhook bearer token; JEV_MCP_WEBHOOK_URL / JEV_MCP_API_KEY are also constants inside the generated Apps Script, not process env)

## Build / CI
make start|stop|test|lint|format|train · ci.yml (macos-latest, py3.12, checkout with submodules): mypy src/jev_mcp/ + pytest --cov-fail-under=85
Commit gate: AGENTS.md (95% per file, 85% overall)

## External integrations
Google Apps Script (examples/gas_triage.js) · Slack (HITL) · MCP clients: Antigravity, Claude Code/Desktop, OpenCode
