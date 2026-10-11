# Jev MCP Instructions

You are an AI pair programmer operating in the Jev MCP repository. The user relies on a local background llama.cpp AI daemon to power various tools here.

## Model Routing

Both engines run side by side (`make start`); requests route between them automatically. To recommend an upstream model for a task, use the `model-router` tool. To restart or repair the daemons, use `make start` / `make stop`, and report which port answered.

## Tech Stack

- Python ≥3.12 (`requires-python`; matches CI and `kev` in `requirements.txt`), single package `src/jev_mcp/` (hatchling build).
- MCP SDK `mcp` 2.2.0 (`>=2.2,<3`), pydantic 2.x for question models.
- Runtime engines (not imports): `llama.cpp` (`llama-server`) daemons — Kev-0.8B on `JEV_FAST_PORT` (8080), Qwen-2.5-7B on `JEV_SMART_PORT` (8081), managed by `jev_mac_manager.sh` (launchd).
- scikit-learn (Platt scaling), tree-sitter (AST chunking, py/js/ts).
- Tools: pytest + pytest-mock, mypy (`make lint`, CI gate), ruff format. No mypy/ruff/pytest config files — defaults apply.

## Project Structure

- `src/jev_mcp/server.py` — all MCP tools & prompts, QFE compression, linter/autofixer gate, router config persistence.
- `src/jev_mcp/provider.py` — `JevProvider` ABC + `NoulQuestion`/`ChoiceQuestion`/`ScoreQuestion` + shared `post_json`.
- `src/jev_mcp/kev_provider.py` (fast tier) / `daemon_provider.py` (smart tier, DCPMI math) / `routing_provider.py` (0.85 confidence gate).
- `src/jev_mcp/linter.py`, `cli_linter.py`, `security.py`, `chunker.py`, `scanner.py`.
- `src/jev_mcp/router_config.py` (router config load/save), `ultrafast_adapter.py` (`local_choose` for the `jev-ultrafast` submodule), `email_triage/` (FastAPI webhook `api.py`, `auth.py`, `core.py` routing rules, `mcp_tool.py`, `slack.py`, `gas_setup.py`; labels persist in `~/.jev-mcp/triage_configs.json`).
- `docs/CODEMAPS/` — token-lean architecture maps (architecture, backend, data, dependencies).
- `tests/` — pytest suite; `conftest.py` stubs the MCP SDK before server import.
- Global runtime config lives OUTSIDE the repo: `~/.jev/router_config.json` (buckets, rules, fitted temperature); logs go to `~/.jev/jev.log`.

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
- The default `mlx` option leverages `mlx_lm.lora` and dynamically automates `mlx_lm.fuse` followed by an auto-restart of the active inference daemon (`llama-server`) for zero-downtime weight updates.
