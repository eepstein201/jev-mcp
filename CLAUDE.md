# Jev MCP Instructions

You are an AI pair programmer operating in the Jev MCP repository. The user relies on a local background llama.cpp AI daemon to power various tools here.

## Model Routing

Both engines run side by side (`make start`); requests route between them automatically. To recommend an upstream model for a task, use the `jev_determine_best_model` tool. To restart or repair the daemons, use `make start` / `make stop`, and report which port answered.

## Tech Stack

- Python ≥3.12 (`requires-python`; matches CI and the `kev` extra of `pyproject.toml`, the only dependency list; `kev` is the alternate `kev.serve` engine and caps `torch<2.9`, so it is its own extra, not part of `engines`), single package `src/jev_mcp/` (hatchling build).
- MCP SDK `mcp` (`>=2.2,<3`; middleware verified identical on 2.2.0 and 2.3.0), pydantic 2.x for question models.
- Runtime engines (not imports): `llama.cpp` (`llama-server`) daemons — the active engine profile's fast and smart models on `JEV_FAST_PORT` (8080) and `JEV_SMART_PORT` (8081) (`default` profile: Kev-0.8B + Qwen2.5-7B), managed by `jev_mac_manager.sh` (launchd).
- scikit-learn (Platt scaling), tree-sitter (AST chunking, py/js/ts).
- Tools: pytest + pytest-mock, mypy (`make lint`, CI gate), ruff format (`make format`; `ruff format --check src tests` is a CI gate). The only tool config is `[tool.mypy] warn_unused_ignores` and `[tool.pytest.ini_options] pythonpath = ["."]` (so tests can import `tests.evals`) in `pyproject.toml`; the dev tools are pinned in the `dev` extra. `make audit` (`scripts/pip_audit.sh`, pip-audit) is a CI gate: advisories accepted with a reason live in `.pip-audit-ignore`, anything new fails.

## Project Structure

- `src/jev_mcp/app.py` — shared server state (`mcp`, `provider`, `linter`, `logger`, `_chat_completion*`, `call_fast_autofixer`, notices, model sync); tool modules reference `app.X` at call time and must never import `jev_mcp.server`.
- `src/jev_mcp/server.py` — entry point (`python -m jev_mcp.server`): `@mcp.prompt` definitions, `main()`, and re-exports of every tool name (`tools/` registers on `app.mcp` at import).
- `src/jev_mcp/tools/` — tool families: `evaluate` (jev_evaluate_batch/jev_explain_decision/jev_generate_synthetic_dataset), `calibrate` (jev_calibrate_threshold/jev_manage_temperature), `router` (jev_manage_router_config/jev_determine_best_model/jev_agent_handoff), `training` (jev_train_lora), `files` (jev_read_file/jev_compact_context/jev_scan_repo), `prompts` (jev_optimize_prompt), `browser`, `triage`.
- `src/jev_mcp/provider.py` — `JevProvider` ABC + `NoulQuestion`/`ChoiceQuestion`/`ScoreQuestion` + shared `post_json`.
- `src/jev_mcp/kev_provider.py` (fast tier) / `daemon_provider.py` (smart tier, DCPMI math; served-model identity from `/props` `model_path` with a 30 s TTL, prompt style plain/sandbox and the routing gate from the active profile) / `routing_provider.py` (confidence gate: the active profile's `gate`, `DEFAULT_GATE` 0.85 without one).
- `src/jev_mcp/engine_profiles.py` + tracked `engine_profiles.toml` (8 models, 7 profiles: default/lean/balanced + eval-only accuracy/probe-*; schema in the phase-1 contract) — the resolver CLI `python -m jev_mcp.engine_profiles resolve|plist|activate` and the server-side `active_profile()`; the marker `~/.jev/active_profile` is written only by `activate`.
- `src/jev_mcp/linter.py`, `cli_linter.py`, `security.py`, `chunker.py`, `scanner.py`.
- `src/jev_mcp/router_config.py` (router config load/save, model update helpers, per-client default seeding), `model_catalog.py` (models.dev catalog fetch/cache, same-family update suggestions), `client_defaults.py` (client-name → bucket-tiers table + `SeedingMiddleware` that seeds defaults at the MCP handshake), `ultrafast_adapter.py` (`local_choose` for the `jev-ultrafast` submodule), `browser_agent.py` (runs that agent on the local engines for `jev_run_browser_agent`), `lora_training.py` (fuse/convert/swap/restore steps for `jev_train_lora`), `prompt_selection.py` (accuracy ranking for `jev_optimize_prompt`), `doctor.py` (`jev-mcp doctor`: engine health, config, path roots, auth variable names, thresholds; prints only from its own `main()`), `client_config.py` (writes/removes the OpenCode entry for the manager), `email_triage/` (FastAPI webhook `api.py`, `auth.py`, `core.py` routing rules, `mcp_tool.py`, `slack.py`, `gas_setup.py`, `rate_limit.py` per-client webhook rate limit; labels persist in `~/.jev-mcp/triage_configs.json`).
- `docs/CODEMAPS/` — token-lean architecture maps (architecture, backend, data, dependencies).
- `tests/` — pytest suite; `conftest.py` stubs the MCP SDK before server import.
- Global runtime config lives OUTSIDE the repo: `~/.jev/router_config.json` (buckets, rules, fitted temperature, `auto_update`), `~/.jev/model_catalog.json` (cached model catalog) and `~/.jev/active_profile` (one-line engine-profile marker, written only by `activate`); logs go to `~/.jev/jev.log`.

## Code Rules

- NEVER `print()` in server code — stdout is the JSON-RPC stream. Use `logging`.
- MCP tools return JSON strings (`{"status": ...}`) and never raise across the boundary; wrap in try/except.
- Tests must mock at the HTTP/subprocess boundary (`urllib.request.urlopen`, `subprocess.check_output`) and must NOT touch the real `~/.jev/router_config.json` — monkeypatch `ROUTER_CONFIG_PATH` to a tmp path.
- Prefer immutable updates (no in-place mutation of result dicts).

## Build & Run

- Daemons: `make start [PROFILE=x]` / `./jev_mac_manager.sh hybrid [x]` — the profile flow: resolve in `engine_profiles.toml`, SHA-256-verified downloads of missing models, plists staged as `.new` before anything stops, restart both agents, then `activate` (writes `~/.jev/active_profile` only when both daemons serve the profile's models; a failed activation exits non-zero). Legacy `./jev_mac_manager.sh start <0.5b|7b>` exits 2. · stop: `make stop`.
- Tests: `make test` (pytest with coverage; the manager e2e `tests/test_mac_manager.py` runs the real manager against a temp HOME with stand-in launchctl/pkill, and the profile lanes live in `tests/test_engine_profiles.py` and `tests/test_engine_profile_server.py`) · Type check: `make lint` (mypy) · Format: `make format` (ruff).
- Live tool calls require the daemons running; without them the router raises `EngineUnavailableError`, tools answer `{"status": "ERROR"}` (`jev_compact_context` answers `ENGINE_UNAVAILABLE` and keeps every chunk), and the log says "PLEASE ENSURE THE DAEMON IS RUNNING".

## Conventions

- Commits: conventional style (`feat:`, `fix:`, `docs:`, `security:`, `test:`, optional scope) on `main`.
- Before committing: follow AGENTS.md — green test run + coverage thresholds (95% per modified file, 85% overall) observed before `git commit`.
- File naming: snake_case modules, `test_*.py` mirroring the source module.

## LoRA Fine-Tuning 
- The `jev_train_lora` tool supports both `mlx` and `llama.cpp` backends. 
- The default `mlx` option runs `mlx_lm.lora`, then (with `fuse=True`) the steps in `lora_training.py`: `mlx_lm fuse --dequantize` → llama.cpp `convert_hf_to_gguf.py` (auto-fetched into `~/.jev/llama.cpp-src` for the commit in `llama-server --version` unless `JEV_LLAMA_CPP_SRC` is set) → `llama-quantize` Q4_K_M. With `target_gguf_path` it keeps the original as `.bak` and restarts only the launchd agent serving that file (`launchctl kickstart -k`); `action="restore"` puts the `.bak` back.
