<!-- Generated: 2026-10-10 | Files scanned: 19 src + 33 tests | Token estimate: ~600 | As of: 8479025 -->
# Architecture

Type: single Python package `src/jev_mcp/` (hatchling), MCP stdio server. No frontend.
Entry: `jev-mcp = jev_mcp.server:main` (stdio; `setup-gas` subcommand) · `jev-lint = jev_mcp.cli_linter:main`

## Data flow
```
MCP client (Claude Code / Claude Desktop / Antigravity CLI / OpenCode)
   │ JSON-RPC over stdio  (NEVER print() — use logging)
   ▼
server.py  MCPServer  ── tools + prompts
   │  linter.py (DecisionPreflightLinter gate) → call_fast_autofixer → QFE compress (> router max_tokens 8192 = min(kev 65536, daemon 8192))
   ▼
routing_provider.RoutingProvider  (0.85 confidence gate)
   ├─ KevProvider    → llama-server :JEV_FAST_PORT 8080  (Kev-0.8B, fast)
   └─ DaemonProvider → llama-server :JEV_SMART_PORT 8081 (Qwen-2.5-7B, DCPMI math, Platt scaling)
   ▲ managed by jev_mac_manager.sh (launchd)
```

## Subsystems
- Evaluation core: provider.py (ABC + Noul/Choice/Score questions, post_json) → kev/daemon/routing providers
- Safety: security.py (sanitize_payload), linter.py, path checks in jev_read_file
- Repo tools: scanner.py, chunker.py (tree-sitter py/js/ts)
- Router config: router_config.py ↔ ~/.jev/router_config.json
- Tests: tests/conftest.py stubs the MCP SDK; autouse `no_real_daemon_http` fails any unmocked daemon HTTP call
- Email triage: email_triage/* (FastAPI webhook + MCP tools + Slack HITL + Google Apps Script)
- Browser agent: ultrafast_adapter.py ↔ jev-ultrafast submodule (stub dispatch in server)
- LoRA training: jev_train_lora → lora_training.py (mlx_lm.lora → fuse --dequantize → convert_hf_to_gguf.py → llama-quantize → launchctl kickstart of the serving agent; restore action)

## Antigravity touchpoints (takeover checklist)
- jev_mac_manager.sh:134 writes MCP entry to `~/.gemini/config/mcp_config.json`
- README.md:9,167,768,805 name Antigravity as a client
- tests/test_browser_agent_e2e.py references it
- AGENTS.md = commit gate (green tests, ≥95% per modified file, ≥85% overall) — inherited from Antigravity
