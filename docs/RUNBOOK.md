# Jev MCP Runbook (local daemons)

Jev MCP is a local tool: an MCP stdio server plus two `llama-server` daemons managed by launchd. There is no hosted deployment, health endpoint, alerting, or escalation path defined in this repository.

## Components

| Piece | Where |
|-------|-------|
| Fast engine (Kev 0.8B) | launchd `com.jev.mlx_server_fast`, port `JEV_FAST_PORT` (8080) |
| Smart engine (Qwen 7B) | launchd `com.jev.mlx_server_smart`, port `JEV_SMART_PORT` (8081) |
| Plists | `~/Library/LaunchAgents/com.jev.mlx_server_{fast,smart}.plist` |
| Legacy single-daemon agent | launchd `com.jev.mlx_server` (`install`/`update` only); `hybrid` stops it and deletes its plist |
| Daemon logs | `~/.jev/mlx_server_fast.log`, `~/.jev/mlx_server_smart.log` |
| MCP server log | `~/.jev/jev.log` |
| Router config | `~/.jev/router_config.json` |
| Email-triage labels | `~/.jev-mcp/triage_configs.json` |
| Models | `models/Kev-0.8B-GGUF/`, `models/Qwen2.5-7B-Instruct-GGUF/` (repo-local) |

## Operations

```bash
make start     # (re)start both daemons; downloads missing model files
make stop      # stop both daemons, keep the install
make update    # re-pull code, update packages, restart
make clean     # uninstall: unload daemons, remove plists, the venv and ALL of ~/.jev
```

`hybrid` stops any existing agents before regenerating plists and starting them, so `make start` is also the restart command. Stopping an agent force-kills only the engine on that agent's port, so other `llama-server` processes on the machine are left alone.

`make clean` runs headless (no confirmation) and deletes `~/.jev`, which holds `router_config.json` (buckets, rules, fitted temperature), LoRA adapters, training artifacts and logs. Copy anything you want to keep first.

## Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Tools return `{"status": "ERROR", "message": "No engine answered: ..."}` (`jev_compact_context`: `ENGINE_UNAVAILABLE`); log says "PLEASE ENSURE THE DAEMON IS RUNNING" | Daemons not running | `make start`, then check `~/.jev/mlx_server_*.log` |
| Port already in use | Another service on 8080/8081 | Set `JEV_FAST_PORT` / `JEV_SMART_PORT` in `.env`, then `make start` |
| Daemon killed or Mac swaps heavily | Too many parallel slots for your RAM | Set a lower `JEV_BATCH_SIZE` in `.env`, then `make start` |
| HTTP 400 on very large emails or states | Request exceeds the per-slot context (16,384 tokens) | Shorten the input; run `jev_compact_context` on it first |
| `jev-ultrafast` import error in tests | Submodule not checked out | `git submodule update --init` |

## Rollback

- **LoRA fusion:** `jev_train_lora` with `target_gguf_path` replaces that model, keeps the original as `<target>.bak`, and restarts only the agent serving it. Roll back with `jev_train_lora` `action="restore"` and the same `target_gguf_path`.
- **Router config:** `~/.jev/router_config.json` is plain JSON (written atomically via a temp file + rename); restore a copy, or delete it to return to defaults — note a Claude-family client will then re-seed provider defaults at its next connect (`buckets_source: "detected"`); any manual bucket choice marks it `"user"` and stops that.
- **Webhook key:** the API key is not part of the deployed Apps Script; the script reads it from the Script Property `JEV_MCP_API_KEY` (Project Settings, Script Properties). To rotate, change `JEV_MCP_API_KEY` on the server and update that property. No re-deploy is needed.
