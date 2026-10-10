# Contributing to Jev MCP

## Prerequisites

- Apple Silicon Mac. `jev_mac_manager.sh` exits on anything other than `arm64`.
- `llama-server` (llama.cpp); the manager checks for it before starting daemons.
- Python ≥3.11 (declared in `pyproject.toml`). CI runs 3.12 because `kev` in `requirements.txt` needs ≥3.12, so use 3.12 locally.
- Git with submodules: `git clone --recurse-submodules https://github.com/eepstein201/jev-mcp.git`. The `jev-ultrafast` submodule is required by `tests/test_browser_agent_e2e.py`.

## Setup

```bash
make install     # venv + models + both daemons (see README for details)
```

Copy `.env.example` to `.env` if you need non-default ports or batch size.

## Commands

<!-- AUTO-GENERATED from Makefile -->
| Command | Description |
|---------|-------------|
| `make install` | `jev_mac_manager.sh install`, then `hybrid` (download both models, start both daemons) |
| `make update` | `jev_mac_manager.sh update` |
| `make start` / `make hybrid` | `jev_mac_manager.sh hybrid` |
| `make stop` | `jev_mac_manager.sh stop` |
| `make clean` | `jev_mac_manager.sh uninstall --headless` |
| `make format` | `ruff format .` in `.venv` |
| `make lint` | `mypy src/jev_mcp/` in `.venv` |
| `make test` | `pytest --cov=src --cov-report=term-missing --cov-fail-under=85 tests/` in `.venv` |
| `make train` | Prints a pointer; training runs through the `train` MCP tool |
<!-- /AUTO-GENERATED -->

## Tests

- Framework: pytest with pytest-cov. Test files mirror source modules (`tests/test_<module>.py`).
- Mock at the boundary: patch `urllib.request.urlopen` and `subprocess.check_output`. Never touch the real `~/.jev/router_config.json`; monkeypatch `ROUTER_CONFIG_PATH` to a tmp path.
- `tests/conftest.py` stubs the MCP SDK before `server` is imported.
- Integrate new tests into the suite itself; a scratch script does not count toward the gate.

## Commit gate (from `AGENTS.md`)

Do not commit unless:

1. The test command exits `0` with no failures.
2. Every modified or added file has ≥95% coverage.
3. Overall coverage is ≥85%.

CI enforces only the 85% overall threshold; the 95% per-file rule is a local gate.

## Code rules

- Never `print()` in server code; stdout is the JSON-RPC stream. Use `logging`.
- MCP tools return JSON strings (`{"status": ...}`) and never raise across the boundary.
- Prefer immutable updates to result dicts.
- Modules are `snake_case`; commits use conventional prefixes (`feat:`, `fix:`, `docs:`, `security:`, `test:`).

## CI

`.github/workflows/ci.yml` runs on pushes and PRs to `main` (macos-latest, Python 3.12, submodules checked out):

1. `pip install -r requirements.txt`, `pip install mypy pytest`, `pip install -e .`
2. `mypy src/jev_mcp/`
3. `pytest --cov=src --cov-report=term --cov-fail-under=85 tests/`

## PR checklist

- [ ] `make lint` and `make test` pass locally
- [ ] Per-file coverage ≥95% for every file you touched
- [ ] No `print()` in server code; no secrets committed
- [ ] Docs updated if a tool signature, env var, or command changed (README, `.env.example`, `docs/CODEMAPS/`)
