<!-- Generated: 2026-10-10, re-checked 2026-10-11 | Files scanned: pyproject.toml, Makefile, ci.yml, jev_mac_manager.sh (as of eed41ad) | Token estimate: ~450 -->
# Dependencies

## Python (pyproject.toml, py>=3.12)
mcp[cli]>=2.2,<3 · pydantic>=2 · fastapi>=0.143 · scikit-learn · tree-sitter(+python/js/ts)
browser-harness · h2 · google-api-python-client · google-auth-oauthlib · pyjwt[crypto] · uvicorn · python-dotenv
extras: engines = mlx==0.32.2 · mlx-lm==0.31.3 · kev (git, pinned to a commit, needs py>=3.12) | dev = ruff · mypy · pytest · pytest-cov
pyproject.toml is the only dependency list (no requirements.txt); CI and the manager install `-e ".[dev,engines]"` / `-e ".[engines]"`

## Runtime engines (not imports)
llama.cpp llama-server ×2 (8080 Kev-0.8B, 8081 Qwen-2.5-7B) · mlx_lm (lora/fuse) · launchd
launchd agents: com.jev.mlx_server_fast / _smart (hybrid) · com.jev.mlx_server (legacy single-daemon; `hybrid` deletes its plist)
`manage_daemon stop <plist>` force-kills only the engine on that plist's port
Submodule: jev-ultrafast (github.com/browser-use/jev-ultrafast)

## Env vars
JEV_FAST_PORT JEV_SMART_PORT JEV_DAEMON_PORT JEV_FAST_ENGINE JEV_BATCH_SIZE
JEV_MCP_JWKS_URL JEV_MCP_JWT_SECRET JEV_MCP_JWT_AUDIENCE JEV_MCP_JWT_ISSUER (webhook JWTs) · JEV_SKIP_MODEL_SYNC (no models.dev fetch at start-up) · JEV_ULTRAFAST_PATH (run-browser-agent)
JEV_MCP_API_KEY (webhook bearer token; JEV_MCP_WEBHOOK_URL / JEV_MCP_API_KEY are also constants inside the generated Apps Script, not process env)

## Build / CI
make install|update|start|hybrid|stop|clean|test|lint|format|eval|train · ci.yml (macos-latest, py3.12, checkout with submodules): mypy src/jev_mcp/ + pytest --cov-fail-under=85
Commit gate: AGENTS.md (95% per file, 85% overall)

## External integrations
Google Apps Script (examples/gas_triage.js) · Slack (HITL) · MCP clients: Claude Code/Desktop, Antigravity (anthropic tiers), OpenCode/Cursor/Windsurf/Continue/Zed (openai tiers), Gemini CLI (google tiers) — per-client bucket seeding via client_defaults.py
