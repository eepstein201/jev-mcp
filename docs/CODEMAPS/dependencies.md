<!-- Generated: 2026-10-10, re-checked 2026-10-11 | Files scanned: pyproject.toml, Makefile, ci.yml, jev_mac_manager.sh (as of eed41ad) | Token estimate: ~450 -->
# Dependencies

## Python (pyproject.toml, py>=3.12)
mcp[cli]>=2.2,<3 · pydantic>=2 · fastapi>=0.143 · scikit-learn · tree-sitter(+python/js/ts)
browser-harness · h2 · google-api-python-client · google-auth-oauthlib · pyjwt[crypto] · uvicorn · python-dotenv
extras: engines = mlx==0.32.2 · mlx-lm==0.31.3 · torch>=2.14.1 · sentencepiece (torch is for llama.cpp's converter) | kev = kev (git, pinned to a commit, needs py>=3.12; caps torch<2.9 so conflicts with engines, `[tool.uv] conflicts`) | dev = ruff==0.17.0 · mypy==2.4.0 · pytest==9.1.1 · pytest-cov==7.1.0 · pip-audit==2.10.1 (all pinned; `[tool.mypy] warn_unused_ignores`)
pyproject.toml is the only dependency list (no requirements.txt); CI and the manager install `-e ".[dev,engines]"` / `-e ".[engines]"`

## Runtime engines (not imports)
llama.cpp llama-server ×2 (8080 Kev-0.8B, 8081 Qwen-2.5-7B) · mlx_lm (lora/fuse) · launchd
launchd agents: com.jev.mlx_server_fast / _smart (hybrid) · com.jev.mlx_server (legacy single-daemon; `hybrid` deletes its plist)
`manage_daemon stop <plist>` force-kills only the engine on that plist's port
Submodule: jev-ultrafast (github.com/browser-use/jev-ultrafast)

## Env vars
JEV_FAST_PORT JEV_SMART_PORT JEV_DAEMON_PORT JEV_FAST_ENGINE JEV_BATCH_SIZE
JEV_MCP_JWKS_URL JEV_MCP_JWT_SECRET JEV_MCP_JWT_AUDIENCE JEV_MCP_JWT_ISSUER (webhook JWTs) · JEV_MCP_RATE_LIMIT_PER_MINUTE (authenticated webhook requests per client address per 60 s, default 60, 0 disables; 429 + Retry-After after auth; keyed on the server-reported address — X-Forwarded-For honoured from local proxies via uvicorn proxy_headers, one shared address only behind a non-local proxy or one without that header, fix with FORWARDED_ALLOW_IPS) · JEV_TRIAGE_ACTION_THRESHOLD JEV_TRIAGE_IMPORTANT_THRESHOLD (email triage routing, 0..1, default 0.8) · JEV_ALLOWED_ROOTS (extra roots for read-file / scan-repo / train paths; os.pathsep-separated) · JEV_SKIP_MODEL_SYNC (no models.dev fetch at start-up) · JEV_ULTRAFAST_PATH (run-browser-agent)
JEV_MCP_API_KEY (webhook bearer token; the generated Apps Script holds JEV_MCP_WEBHOOK_URL as a constant and reads the key from a Script Property of the same name, never from source)

## Build / CI
make install|update|start|hybrid|stop|clean|test|lint|format|format-check|audit|check|eval|train · ci.yml (macos-latest, py3.12, checkout with submodules; installs with `uv sync --locked --extra dev --extra engines`) · dependabot.yml (weekly: uv lockfile, github-actions): ruff format --check src tests + mypy src/jev_mcp/ + pytest --cov-fail-under=85 + scripts/pip_audit.sh (accepted advisories: .pip-audit-ignore)
Commit gate: AGENTS.md (95% per file, 85% overall)

## External integrations
Google Apps Script (examples/gas_triage.js) · Slack (HITL) · MCP clients: Claude Code/Desktop, Antigravity (anthropic tiers), OpenCode/Cursor/Windsurf/Continue/Zed (openai tiers), Gemini CLI (google tiers) — per-client bucket seeding via client_defaults.py
