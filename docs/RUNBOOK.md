# Jev MCP Runbook (local daemons)

Jev MCP is a local tool: an MCP stdio server plus two `llama-server` daemons managed by launchd. There is no hosted deployment, health endpoint, alerting, or escalation path defined in this repository.

## Components

| Piece | Where |
|-------|-------|
| Fast engine (the profile's fast model; `default`: Kev 0.8B) | launchd `com.jev.mlx_server_fast`, port `JEV_FAST_PORT` (8080) |
| Smart engine (the profile's smart model; `default`: Qwen 7B) | launchd `com.jev.mlx_server_smart`, port `JEV_SMART_PORT` (8081) |
| Plists | `~/Library/LaunchAgents/com.jev.mlx_server_{fast,smart}.plist` |
| Legacy single-daemon agent | launchd `com.jev.mlx_server` (`install` only); `hybrid` stops it and deletes its plist |
| Daemon logs | `~/.jev/mlx_server_fast.log`, `~/.jev/mlx_server_smart.log` |
| MCP server log | `~/.jev/jev.log` |
| Router config | `~/.jev/router_config.json` |
| Active engine profile | `~/.jev/active_profile` (one line; written only by `jev_mcp.engine_profiles activate`, which `make start` runs last) |
| Profile table | `engine_profiles.toml` (tracked in the repo; 8 models, 7 profiles) |
| Email-triage labels | `~/.jev-mcp/triage_configs.json` |
| Models | `models/<repo name>/<file>` per `engine_profiles.toml` (repo-local; downloaded with SHA-256 verification) |

## Operations

```bash
make start [PROFILE=name]   # (re)start both daemons on a profile; downloads missing model files
                            # (SHA-256 verified), then activates: writes ~/.jev/active_profile
                            # only once both daemons serve the profile's models
make stop                   # stop both daemons, keep the install
make update                 # re-pull code, update packages, restart both daemons on the active profile
make clean                  # uninstall: unload daemons, remove plists, the venv and ALL of ~/.jev
```

`hybrid` (what `make start` runs) resolves the profile in `engine_profiles.toml`, stages both new plists as `.new` before stopping anything, stops any existing agents, and only then swaps the plists in and starts them — so a failed step leaves the running daemons untouched. This makes `make start` the restart command too. Stopping an agent force-kills only the engine on that agent's port, so other `llama-server` processes on the machine are left alone. Eval-only profiles (`accuracy`, `probe-*`) need `JEV_ALLOW_EVAL_ONLY=1`.

`make clean` runs headless (no confirmation) and deletes `~/.jev`, which holds `router_config.json` (buckets, rules, fitted temperature), LoRA adapters, training artifacts and logs. Copy anything you want to keep first.

## Health check

```bash
.venv/bin/python -m jev_mcp.server doctor   # or: jev-mcp doctor [--json]
```

One command answers "is anything misconfigured?": each engine's answer to `GET /health`, the active engine profile with the model each daemon actually serves next to the one that profile expects (a mismatch is a finding), the router config's JSON validity, the path-security roots (including a working directory ignored for being the home directory), which webhook auth variables are set, the triage thresholds, and the Python / `mcp` / `torch` versions. Exit `0` = healthy, exit `1` = at least one finding, and each finding names its fix (`make start`, `JEV_MCP_JWT_AUDIENCE`, `JEV_ALLOWED_ROOTS`). Auth variables appear by name only; a value is never printed. Run this before anything in "Common issues".

## Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Tools return `{"status": "ERROR", "message": "No engine answered: ..."}` (`jev_compact_context`: `ENGINE_UNAVAILABLE`); log says "PLEASE ENSURE THE DAEMON IS RUNNING" | Daemons not running | `make start`, then check `~/.jev/mlx_server_*.log` |
| Port already in use | Another service on 8080/8081 | Set `JEV_FAST_PORT` / `JEV_SMART_PORT` in `.env`, then `make start` |
| Daemon killed or Mac swaps heavily | The profile's model pair is too large for your RAM | Start a lighter profile (`make start PROFILE=lean`) or hand-edit `parallel` down in `engine_profiles.toml` |
| `make start` ends with "Profile … did not become active" (stderr lines like `smart model_path: expected X, served Y`) | A daemon serves a different model, slot count or context than the profile | Check `~/.jev/mlx_server_*.log`; re-run `make start`; the marker in `~/.jev/active_profile` was left untouched |
| HTTP 400 on very large emails or states | Request exceeds the per-slot context (16,384 tokens) | Shorten the input; run `jev_compact_context` on it first |
| `jev-ultrafast` import error in tests | Submodule not checked out | `git submodule update --init` |

## Rollback

- **LoRA fusion:** `jev_train_lora` with `target_gguf_path` replaces that model, keeps the original as `<target>.bak`, and restarts only the agent serving it. Roll back with `jev_train_lora` `action="restore"` and the same `target_gguf_path`.
- **Router config:** `~/.jev/router_config.json` is plain JSON (written atomically via a temp file + rename); restore a copy, or delete it to return to defaults — note a Claude-family client will then re-seed provider defaults at its next connect (`buckets_source: "detected"`); any manual bucket choice marks it `"user"` and stops that.
- **Webhook key:** the API key is not part of the deployed Apps Script; the script reads it from the Script Property `JEV_MCP_API_KEY` (Project Settings, Script Properties). To rotate, change `JEV_MCP_API_KEY` on the server and update that property. No re-deploy is needed.
