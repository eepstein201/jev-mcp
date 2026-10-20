<!-- Generated: 2026-10-10, refreshed 2026-10-20 | Files scanned: 41 src + 73 test modules | Token estimate: ~850 | As of: 70de32f -->
# Architecture

Type: single Python package `src/jev_mcp/` (hatchling), MCP stdio server. No frontend.
Entry: `jev-mcp = jev_mcp.server:main` (stdio by default; subcommands `setup-gas`, `serve-webhook`, `doctor`) · `jev-lint = jev_mcp.cli_linter:main` · `python -m jev_mcp.client_config link-opencode|unlink-opencode` (called by the manager) · `python -m jev_mcp.engine_profiles resolve|plist|activate` (profile resolver; also called by the manager)

## Data flow
```
MCP client (Claude Code / Claude Desktop / Antigravity CLI / OpenCode)
   │ JSON-RPC over stdio  (NEVER print() — use logging)
   ▼
server.py (entry: prompts + main)  ──  app.py MCPServer + tools/* registrations
   │  linter.py (DecisionPreflightLinter gate) → call_fast_autofixer → QFE compress (> router max_tokens 8192 = min(kev 65536, daemon 8192))
   ▼
routing_provider.RoutingProvider  (confidence gate: active profile's gate, 0.85 without one)
   ├─ KevProvider    → llama-server :JEV_FAST_PORT 8080  (profile fast model; default Kev-0.8B)
   └─ DaemonProvider → llama-server :JEV_SMART_PORT 8081 (profile smart model, default Qwen-2.5-7B; DCPMI math, Platt scaling)
   ▲ managed by jev_mac_manager.sh (launchd), which serves the active engine profile
```

## Subsystems
- Evaluation core: provider.py (ABC + Noul/Choice/Score questions, post_json) → kev/daemon/routing providers
- Engine profiles: engine_profiles.py ↔ tracked engine_profiles.toml (8 models, 7 profiles) + ~/.jev/active_profile marker (written only by `activate`); `make start [PROFILE=x]` = resolve → SHA-256-verified downloads → plists (.new, before anything stops) → restart → activate. Server side: served model id from /props model_path (30 s TTL), gate + smart prompt style (plain/sandbox) from the profile; responses carry {profile, fast, smart} via app.engine_identity
- Safety: security.py (sanitize_payload), linter.py, path checks in jev_read_file
- Repo tools: scanner.py, chunker.py (tree-sitter py/js/ts) · compaction.py (jev_compact_context chunking + per-chunk scoring, shared with the eval harness)
- Router config: router_config.py ↔ ~/.jev/router_config.json · model_catalog.py (models.dev catalog, update suggestions) · client_defaults.py (SeedingMiddleware: per-client bucket defaults at the MCP handshake)
- Health: doctor.py (`jev-mcp doctor [--json]`: engine /health + profile-expected vs served model, router-config validity, path roots, auth variables by NAME only, thresholds, versions; exit 0/1) · availability.py (`python -m jev_mcp.availability sample|summary`: probes both engines on an interval into ~/.jev/availability.jsonl; docs/perf/availability.md)
- Client config: client_config.py (writes/removes this server's OpenCode entry; never rewrites an unparseable file)
- Tests: tests/conftest.py stubs the MCP SDK; autouse `no_real_daemon_http` fails any unmocked daemon HTTP call; manager e2e tests/test_mac_manager.py runs the real manager against a temp HOME (stand-in launchctl/pkill) + fake llama server
- Evals (need live daemons, not part of `make test`): tests/run_evals.py (engine benchmark; `--scores-out` per-row dump, `--simulate` offline routing-gate replay) · tests/evals/{browser_eval,triage_eval,hf_email_import,memory,compaction_eval,webhook_latency}.py (compaction_eval/webhook_latency run directly, no make target; triage_eval `--binary-label-key` adds ROC-AUC/AP/best-F1 via metrics.py) + usage/ (tool-usage eval, calls the Anthropic API) — `make eval|eval-browser|eval-triage|eval-memory|eval-usage`
- Email triage: email_triage/* (FastAPI webhook + MCP tools + Slack HITL + Google Apps Script)
- Browser agent: ultrafast_adapter.py ↔ jev-ultrafast submodule (stub dispatch in server)
- LoRA training: jev_train_lora → lora_training.py (mlx_lm.lora → fuse --dequantize → convert_hf_to_gguf.py → llama-quantize → launchctl kickstart of the serving agent; restore action)

## Antigravity touchpoints (takeover checklist)
- jev_mac_manager.sh `link_mcp_configs` writes the MCP entry to `~/.gemini/config/mcp_config.json` (beside Claude Desktop and `~/.claude.json`; OpenCode goes through client_config.py)
- README.md names Antigravity as a client (intro, tools list, "Connecting Jev to Your AI Client", model-routing SOP); client_defaults.py maps it to the anthropic tiers
- AGENTS.md = commit gate (green tests, ≥95% per modified file, ≥85% overall) — inherited from Antigravity
