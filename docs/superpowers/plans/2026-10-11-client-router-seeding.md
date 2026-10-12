# Per-Client Router Default Seeding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the connecting MCP client is an Anthropic-family client and the user has not chosen router buckets, seed b1–b4 with Anthropic tiers and record provenance so user choices are never overwritten; unknown clients keep today's behavior.

**Architecture:** An SDK `ServerMiddleware` (`mcp.server.context.ServerMiddleware`, protocol `async __call__(ctx, call_next)`) observes `notifications/initialized`, reads `ctx.session.client_params.clientInfo.name`, and asks `router_config` to seed if eligible. Provenance lives in the config itself (`buckets_source: "detected" | "user"`, `detected_client`). Tools that mutate buckets on user consent stamp `"user"`.

**Tech Stack:** Python ≥3.12, mcp SDK 2.2.0 (`MCPServer(..., middleware=[...])`), pytest + pytest-cov, mypy.

**Spec:** the approved in-chat design from this session (bounded path — summarized in **Design invariants** below) and `docs/TODO.md`'s "Per-client router defaults" entry (the **Decision (2026-10-10)** paragraph).

## Design invariants (from the approved design)

- v1 client table is Anthropic-only: patterns `("claude code", "claude desktop", "claude", "antigravity")`, substring match on lowercased `clientInfo.name`; tiers identical to `router_config.FALLBACK_BUCKETS` (kept current by the existing catalog machinery). OpenAI/Google rows are future work, not YAGNI'd speculation.
- Seed **only** when: config file missing, or `buckets_source == "detected"`.
- Existing configs without `buckets_source` are treated as `"user"` — never re-seeded (conservative migration; the user's live config must stay untouched).
- `set_bucket`, `apply_updates`, `set_auto_update(enabled=true)` stamp `buckets_source: "user"`. The background same-family auto-update (`app._sync_models`) leaves `buckets_source` unchanged.
- The middleware never issues server-to-client requests (deadlock risk during `initialize` — SDK docstring warning), never raises across the handshake, and seeds at most once per server process.

## Global Constraints

- Run tests as `.venv/bin/python -m pytest tests -q` (bare `pytest` collects the `jev-ultrafast` submodule and fails).
- Commit gate: green suite, ≥95% coverage per modified file, ≥85% overall, `mypy` clean (`.venv/bin/mypy src/jev_mcp`).
- NEVER `print()` in server code (stdout is the JSON-RPC stream) — use `app.logger`/module logger.
- Tests must not touch the real `~/.jev/`: conftest already gives unit tests a temp `router_config.ROUTER_CONFIG_PATH`; subprocess e2e tests must set `HOME` to a tmp dir in the child env.
- Conventional commits (`feat:`, `test:`, `docs:`), no Co-Authored-By; work on a feature branch off `main`.
- Idiom per `/ecc:python-patterns`: explicit over implicit, specific exceptions (no bare `except:`), type hints on every new signature, no mutable default args.

## Review Focus

The input classes the design implies but the happy path doesn't exercise — each pinned by a test in the owning task:

1. **Middleware deadlocks if it awaits anything on `ctx.session`** (SDK: awaiting server-to-client requests inside the chain deadlocks `initialize`). — Task 3: unit test asserts the middleware touches only `call_next` and reads `client_params` (fake session fails the test if `send_request`/`send_ping` is called).
2. **Existing installs (config without `buckets_source`) re-seeded by a Claude client.** — Task 1: fieldless config on disk + Claude client → file byte-identical after seeding attempt.
3. **Seeding error breaks the MCP handshake.** — Task 3: `seed_defaults_for_client` monkeypatched to raise → middleware still returns `call_next`'s result, logs a warning.
4. **Repeated `notifications/initialized` (misbehaving client) seeds twice.** — Task 3: two invocations → `seed_defaults_for_client` called exactly once.
5. **`client_params` is `None` or `clientInfo.name` empty/unreadable.** — Task 3: `client_params=None` → no seeding, no exception. (Task 1 covers the empty-name case at the mapping layer.)

---

### Task 1: Client mapping and eligibility-checked seeding

**Files:**
- Create: `src/jev_mcp/client_defaults.py`
- Modify: `src/jev_mcp/router_config.py`
- Test: `tests/test_client_defaults.py`

**Interfaces:**
- Consumes: `router_config.load_router_config()`, `router_config.save_router_config(config)`, `router_config.FALLBACK_BUCKETS`, `router_config.ROUTER_CONFIG_PATH` (all existing).
- Produces:
  - `client_defaults.ANTHROPIC_CLIENT_PATTERNS: tuple[str, ...]`
  - `client_defaults.default_buckets_for_client(client_name: str) -> dict[str, str] | None`
  - `router_config.seed_defaults_for_client(client_name: str) -> dict[str, Any] | None` — the written config, or `None` when not eligible.

- [ ] **Step 1: Write the failing tests** in `tests/test_client_defaults.py`

```python
# All run under conftest's temp ROUTER_CONFIG_PATH (autouse fixture).
def test_claude_code_name_maps_to_anthropic_tiers():
    buckets = client_defaults.default_buckets_for_client("Claude Code v2.1")
    assert buckets == dict(router_config.FALLBACK_BUCKETS)


def test_unknown_client_maps_to_none():
    assert client_defaults.default_buckets_for_client("opencode") is None
    assert client_defaults.default_buckets_for_client("") is None


def test_seeds_when_no_config_file_exists(monkeypatch):
    # no config file in the tmp home
    written = router_config.seed_defaults_for_client("Claude Code")
    assert written is not None
    on_disk = router_config.load_router_config()
    assert on_disk["buckets"] == dict(router_config.FALLBACK_BUCKETS)
    assert on_disk["rules"] == []
    assert on_disk["buckets_source"] == "detected"
    assert on_disk["detected_client"] == "Claude Code"


def test_fieldless_existing_config_is_never_reseeded():
    router_config.save_router_config({"buckets": {"b1": "my-model"}, "rules": []})
    before = Path(router_config.ROUTER_CONFIG_PATH).read_text()
    assert router_config.seed_defaults_for_client("Claude Code") is None
    assert Path(router_config.ROUTER_CONFIG_PATH).read_text() == before


def test_detected_config_reseeds_to_current_table():
    router_config.save_router_config(
        {
            "buckets": {"b1": "old"},
            "rules": [],
            "buckets_source": "detected",
            "detected_client": "Claude Code",
        }
    )
    written = router_config.seed_defaults_for_client("Claude Code")
    assert written["buckets"] == dict(router_config.FALLBACK_BUCKETS)
    # non-bucket fields survive
    assert router_config.load_router_config()["detected_client"] == "Claude Code"


def test_unknown_client_never_writes():
    assert router_config.seed_defaults_for_client("opencode") is None
    assert not os.path.exists(router_config.ROUTER_CONFIG_PATH)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_client_defaults.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jev_mcp.client_defaults'`

- [ ] **Step 3: Implement `src/jev_mcp/client_defaults.py`**

`ANTHROPIC_CLIENT_PATTERNS = ("claude code", "claude desktop", "claude", "antigravity")` and `default_buckets_for_client(client_name: str) -> dict[str, str] | None`: return `dict(FALLBACK_BUCKETS)` when any pattern is a substring of `client_name.lower()`, else `None`. No I/O in this module's mapping function.

- [ ] **Step 4: Implement `seed_defaults_for_client` in `router_config.py`**

Guard order: (1) `default_buckets_for_client` returns `None` → return `None`; (2) file missing → write fresh `{"buckets": tiers, "rules": [], "buckets_source": "detected", "detected_client": name}`; (3) file exists and `buckets_source == "detected"` → immutable update (`{**config, "buckets": tiers, "detected_client": name}`), preserving every other field; (4) anything else → return `None`. Use specific exception handling (`OSError`, `json.JSONDecodeError`) — a corrupt file is treated as not-eligible (return `None`), never raised.

- [ ] **Step 5: Run tests to verify pass, then full gate**

Run: `.venv/bin/python -m pytest tests -q && .venv/bin/mypy src/jev_mcp`
Expected: all pass, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/jev_mcp/client_defaults.py src/jev_mcp/router_config.py tests/test_client_defaults.py
git commit -m "feat: map connecting-client name to router default buckets with provenance"
```

---

### Task 2: User actions stamp `buckets_source: "user"`

**Files:**
- Modify: `src/jev_mcp/tools/router.py` (`jev_manage_router_config`)
- Modify: `src/jev_mcp/app.py` (`_sync_models` — no code change expected, test pins the invariant)
- Test: `tests/test_client_defaults.py`

**Interfaces:**
- Consumes: Task 1's `buckets_source` field semantics.
- Produces: no new names; behavior contract consumed by Task 4's e2e.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_client_defaults.py`)

```python
def test_set_bucket_stamps_user_provenance():
    router_config.save_router_config(
        {
            "buckets": dict(router_config.FALLBACK_BUCKETS),
            "rules": [],
            "buckets_source": "detected",
            "detected_client": "Claude Code",
        }
    )
    server.jev_manage_router_config(action="set_bucket", bucket_id="b1", model_name="m")
    assert router_config.load_router_config()["buckets_source"] == "user"


def test_apply_updates_stamps_user_provenance(cached_catalog):
    # STALE buckets from tests/test_router_updates.py style fixture + catalog with newer same-family models
    ...
    server.jev_manage_router_config(action="apply_updates")
    assert router_config.load_router_config()["buckets_source"] == "user"


def test_background_auto_update_preserves_detected(monkeypatch, cached_catalog):
    # auto_update=True + STALE buckets + buckets_source "detected"; run app._sync_models()
    assert router_config.load_router_config()["buckets_source"] == "detected"
```

Complete the `...` bodies by reusing the `CATALOG`/`STALE` fixtures from `tests/test_router_updates.py` (import or copy — prefer moving them to a small local helper in this file; do not edit the other test file).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_client_defaults.py -q`
Expected: FAIL — `test_set_bucket_stamps_user_provenance` (field stays `"detected"`).

- [ ] **Step 3: Implement in `tools/router.py`**

In `jev_manage_router_config`, the three consent paths (`set_bucket`, `apply_updates`, `set_auto_update` with truthy `enabled`) stamp `buckets_source: "user"` on the config they save (immutable update — add the key into the dict passed to `app.save_router_config`). No other action writes the field.

- [ ] **Step 4: Run full gate**

Run: `.venv/bin/python -m pytest tests -q && .venv/bin/mypy src/jev_mcp`
Expected: all pass (Task 1 tests + existing 362), mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/jev_mcp/tools/router.py tests/test_client_defaults.py
git commit -m "feat: manage-router consent actions mark bucket provenance as user-chosen"
```

---

### Task 3: `SeedingMiddleware` wired into the server

**Files:**
- Modify: `src/jev_mcp/client_defaults.py` (add the middleware class)
- Modify: `src/jev_mcp/app.py:74` (`MCPServer("jev-mcp")` → pass `middleware=[...]`)
- Test: `tests/test_client_defaults.py`

**Interfaces:**
- Consumes: `router_config.seed_defaults_for_client` (Task 1), `app.logger`.
- Produces: `client_defaults.SeedingMiddleware` with `async def __call__(self, ctx: Any, call_next: Any) -> Any`.

- [ ] **Step 1: Write the failing tests** (append; async via `pytest.mark.asyncio` is NOT configured — drive coroutines with `asyncio.run` inside sync tests)

```python
def _fake_ctx(method, client_name):
    # ctx.method: str; ctx.session.client_params.clientInfo.name — build with
    # SimpleNamespace; session also carries send_request/send_ping sentinels
    # that raise AssertionError("must not be awaited") if touched.
    ...

def test_middleware_seeds_on_initialized_notification():
    # call_next returns a sentinel; ctx method "notifications/initialized", name "Claude Code"
    # assert result is sentinel and seed happened (config file now exists)

def test_middleware_ignores_other_methods():
    # method "tools/list" → no config file written, sentinel returned

def test_middleware_never_uses_session_requests():
    # fake session whose send_request/send_ping raise; run middleware; passes iff untouched

def test_middleware_seeds_at_most_once():
    # two invocations, then assert router_config.seed_defaults_for_client
    # (monkeypatched counting stub) called exactly once

def test_middleware_survives_seeding_failure(monkeypatch):
    # seed stub raises RuntimeError; middleware returns call_next's sentinel, no raise

def test_middleware_tolerates_missing_client_params():
    # client_params None → sentinel returned, no seeding, no exception
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_client_defaults.py -q`
Expected: FAIL — `AttributeError: module 'jev_mcp.client_defaults' has no attribute 'SeedingMiddleware'`

- [ ] **Step 3: Implement `SeedingMiddleware` in `client_defaults.py`**

Instance holds a `threading.Event` as the one-shot guard. `__call__`: if `ctx.method == "notifications/initialized"` and the event can be set (first time only), read `client_params` defensively (`getattr` chains, `None` at any hop → skip) and call `router_config.seed_defaults_for_client(name)` inside `try/except Exception` logging a warning — then **always** `return await call_next(ctx)` on the single original path (seed before or after, but never between a conditional return). Import `router_config` lazily inside the method to keep module import order simple.

- [ ] **Step 4: Wire into `app.py`**

`from jev_mcp.client_defaults import SeedingMiddleware` at top; `mcp = MCPServer("jev-mcp", middleware=[SeedingMiddleware()])`. The conftest `DummyMCPServer` accepts kwargs, so unit tests are unaffected.

- [ ] **Step 5: Run full gate**

Run: `.venv/bin/python -m pytest tests -q && .venv/bin/mypy src/jev_mcp`
Expected: all pass, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/jev_mcp/client_defaults.py src/jev_mcp/app.py tests/test_client_defaults.py
git commit -m "feat: seed router defaults from the connecting client at handshake"
```

---

### Task 4: Subprocess e2e — real handshake, isolated HOME

**Files:**
- Modify: `tests/test_stdio_e2e.py`
- Docs: `docs/TODO.md` (tick the per-client seeding decision line), `CLAUDE.md` (add `client_defaults.py` to Project Structure), `docs/CODEMAPS/backend.md` (one line under Layout)

**Interfaces:**
- Consumes: everything above; the existing `_start_server`/`send`/`read_rpc` helpers in `tests/test_stdio_e2e.py`.

Per `/ecc:e2e-testing` discipline translated to stdio: isolated state (`HOME=tmp` in the child env — the real `~/.jev` must never be written), wait-for-condition not fixed timeouts (poll for the config file with a deadline), deterministic env (`JEV_SKIP_MODEL_SYNC=1` already inherited; seeding is local-only so no daemon needed).

- [ ] **Step 1: Write the failing e2e tests**

Two tests, reusing the module's JSON-RPC helpers with a `_start_server(home=tmp_path)` variant that overrides `env["HOME"]`:

```python
def test_stdio_handshake_seeds_anthropic_defaults_for_claude_client(tmp_path):
    # initialize with clientInfo {"name": "claude-code-e2e", "version": "0.0.1"},
    # send notifications/initialized, then tools/list (keeps the session alive).
    # Poll (deadline loop, 0.1s steps, ~10s max) for tmp_path/.jev/router_config.json.
    # Assert: buckets == FALLBACK_BUCKETS, buckets_source == "detected",
    # detected_client == "claude-code-e2e", "rules": [].

def test_stdio_handshake_does_not_seed_unknown_client(tmp_path):
    # same flow with clientInfo name "opencode-e2e";
    # after tools/list completes, assert the config file does NOT exist.
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_stdio_e2e.py -q`
Expected: FAIL — `FileNotFoundError` on the polled config path (first test); second may pass already (acceptable — it pins the invariant).

- [ ] **Step 3: Make it pass**

No product code expected — if the e2e fails, the bug is in Tasks 1–3; fix there, not in the test. Then update the three docs (TODO tick with a one-line *(fixed: …)* note; CLAUDE.md structure line; backend codemap line).

- [ ] **Step 4: Run the full gate with coverage**

Run: `.venv/bin/python -m pytest tests -q --cov=src/jev_mcp --cov-report=term && .venv/bin/mypy src/jev_mcp`
Expected: all pass; `client_defaults.py` ≥95%, overall ≥85%; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_stdio_e2e.py docs/TODO.md CLAUDE.md docs/CODEMAPS/backend.md
git commit -m "test: e2e handshake seeds per-client defaults; docs for client_defaults"
```

---

### Task 5: PR

- [ ] Push the branch, open a PR to `main` (body ends with the Claude Code line), verify CI green on the PR head, and merge only on the user's confirmation.
