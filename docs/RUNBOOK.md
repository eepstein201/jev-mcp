# Jev MCP Runbook (local daemons)

Jev MCP is a local tool: an MCP stdio server plus two `llama-server` daemons managed by launchd. There is no hosted deployment, health endpoint, alerting, or escalation path defined in this repository.

## Components

| Piece | Where |
|-------|-------|
| Fast engine (Kev 0.8B) | launchd `com.jev.mlx_server_fast`, port `JEV_FAST_PORT` (8080) |
| Smart engine (Qwen 7B) | launchd `com.jev.mlx_server_smart`, port `JEV_SMART_PORT` (8081) |
| Plists | `~/Library/LaunchAgents/com.jev.mlx_server_{fast,smart}.plist` |
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
make clean     # uninstall: unload daemons, remove the venv
```

`hybrid` stops any existing agents before regenerating plists and starting them, so `make start` is also the restart command.

## Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Tools return empty results; log says "PLEASE ENSURE THE DAEMON IS RUNNING" | Daemons not running | `make start`, then check `~/.jev/mlx_server_*.log` |
| Port already in use | Another service on 8080/8081 | Set `JEV_FAST_PORT` / `JEV_SMART_PORT` in `.env`, then `make start` |
| Daemon killed or Mac swaps heavily | Too many parallel slots for your RAM | Set a lower `JEV_BATCH_SIZE` in `.env`, then `make start` |
| HTTP 400 on very large emails or states | Request exceeds the per-slot context (16,384 tokens) | Shorten the input; `compact` it first |
| `jev-ultrafast` import error in tests | Submodule not checked out | `git submodule update --init` |

## Rollback

- **LoRA fusion:** `train` with `target_gguf_path` replaces that model and backs up the original; restore the backup and run `make start`.
- **Router config:** `~/.jev/router_config.json` is plain JSON; restore a copy or delete it to return to defaults.
- **Webhook key:** the API key is written into the deployed Apps Script by `jev-mcp setup-gas`. To rotate, change `JEV_MCP_API_KEY` and re-run `setup-gas`.
