# Jev MCP Instructions

You are an AI pair programmer operating in the Jev MCP repository. The user relies on a local background llama.cpp AI daemon to power various tools here.

## Model Routing

Both engines run side by side (`make start`); requests route between them automatically. To recommend an upstream model for a task, use the `model-router` tool. To restart or repair the daemons, use `make start` / `make stop`, and report which port answered.

## Tech Stack

- Python ≥3.12 (`requires-python`; matches CI and `kev` in the `engines` extra of `pyproject.toml`, the only dependency list), single package `src/jev_mcp/` (hatchling build).
- MCP SDK `mcp` (`>=2.2,<3`; middleware verified identical on 2.2.0 and 2.3.0), pydantic 2.x for question models.
- Runtime engines (not imports): `llama.cpp` (`llama-server`) daemons — Kev-0.8B on `JEV_FAST_PORT` (8080), Qwen-2.5-7B on `JEV_SMART_PORT` (8081), managed by `jev_mac_manager.sh` (launchd).
- scikit-learn (Platt scaling), tree-sitter (AST chunking, py/js/ts).
- Tools: pytest + pytest-mock, mypy (`make lint`, CI gate), ruff format. No mypy/ruff/pytest config files — defaults apply.

## Project Structure

- `src/jev_mcp/app.py` — shared server state (`mcp`, `provider`, `linter`, `logger`, `_chat_completion*`, `call_fast_autofixer`, notices, model sync); tool modules reference `app.X` at call time and must never import `jev_mcp.server`.
- `src/jev_mcp/server.py` — entry point (`python -m jev_mcp.server`): `@mcp.prompt` definitions, `main()`, and re-exports of every tool name (`tools/` registers on `app.mcp` at import).
- `src/jev_mcp/tools/` — tool families: `evaluate` (evaluate/explain-decision/generate-data), `calibrate` (calibrate/temperature), `router` (manage-router/model-router/handoff), `training` (train), `files` (read-file/compact/scan-repo), `prompts` (optimize-prompt), `browser`, `triage`.
- `src/jev_mcp/provider.py` — `JevProvider` ABC + `NoulQuestion`/`ChoiceQuestion`/`ScoreQuestion` + shared `post_json`.
- `src/jev_mcp/kev_provider.py` (fast tier) / `daemon_provider.py` (smart tier, DCPMI math) / `routing_provider.py` (0.85 confidence gate).
- `src/jev_mcp/linter.py`, `cli_linter.py`, `security.py`, `chunker.py`, `scanner.py`.
- `src/jev_mcp/router_config.py` (router config load/save, model update helpers, per-client default seeding), `model_catalog.py` (models.dev catalog fetch/cache, same-family update suggestions), `client_defaults.py` (client-name → bucket-tiers table + `SeedingMiddleware` that seeds defaults at the MCP handshake), `ultrafast_adapter.py` (`local_choose` for the `jev-ultrafast` submodule), `browser_agent.py` (runs that agent on the local engines for `run-browser-agent`), `lora_training.py` (fuse/convert/swap/restore steps for `train`), `prompt_selection.py` (accuracy ranking for `optimize-prompt`), `email_triage/` (FastAPI webhook `api.py`, `auth.py`, `core.py` routing rules, `mcp_tool.py`, `slack.py`, `gas_setup.py`; labels persist in `~/.jev-mcp/triage_configs.json`).
- `docs/CODEMAPS/` — token-lean architecture maps (architecture, backend, data, dependencies).
- `tests/` — pytest suite; `conftest.py` stubs the MCP SDK before server import.
- Global runtime config lives OUTSIDE the repo: `~/.jev/router_config.json` (buckets, rules, fitted temperature, `auto_update`) and `~/.jev/model_catalog.json` (cached model catalog); logs go to `~/.jev/jev.log`.

## Code Rules

- NEVER `print()` in server code — stdout is the JSON-RPC stream. Use `logging`.
- MCP tools return JSON strings (`{"status": ...}`) and never raise across the boundary; wrap in try/except.
- Tests must mock at the HTTP/subprocess boundary (`urllib.request.urlopen`, `subprocess.check_output`) and must NOT touch the real `~/.jev/router_config.json` — monkeypatch `ROUTER_CONFIG_PATH` to a tmp path.
- Prefer immutable updates (no in-place mutation of result dicts).

## Build & Run

- Daemons: `make start` / `./jev_mac_manager.sh hybrid` · stop: `make stop`.
- Tests: `make test` (pytest with coverage) · Type check: `make lint` (mypy) · Format: `make format` (ruff).
- Live tool calls require the daemons running; without them providers return empty results and log "PLEASE ENSURE THE DAEMON IS RUNNING".

## Conventions

- Commits: conventional style (`feat:`, `fix:`, `docs:`, `security:`, `test:`, optional scope) on `main`.
- Before committing: follow AGENTS.md — green test run + coverage thresholds (95% per modified file, 85% overall) observed before `git commit`.
- File naming: snake_case modules, `test_*.py` mirroring the source module.

## LoRA Fine-Tuning 
- The `jev_train_lora` tool supports both `mlx` and `llama.cpp` backends. 
- The default `mlx` option runs `mlx_lm.lora`, then (with `fuse=True`) the steps in `lora_training.py`: `mlx_lm fuse --dequantize` → llama.cpp `convert_hf_to_gguf.py` (auto-fetched into `~/.jev/llama.cpp-src` for the commit in `llama-server --version` unless `JEV_LLAMA_CPP_SRC` is set) → `llama-quantize` Q4_K_M. With `target_gguf_path` it keeps the original as `.bak` and restarts only the launchd agent serving that file (`launchctl kickstart -k`); `action="restore"` puts the `.bak` back.
