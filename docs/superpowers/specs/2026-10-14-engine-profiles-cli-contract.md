# Engine profiles: phase-1 interface contract

Frozen before lanes P1-a (resolver), P1-b (manager) and P1-c (server side) start.
Design: `2026-10-14-engine-profiles-design.md` (this folder). A lane that finds this
contract wrong stops and reports; it does not change the interface on its own.

## 1. Ownership

| Lane | Owns (only these files) |
|---|---|
| P1-a | `engine_profiles.toml`, `src/jev_mcp/engine_profiles.py` (fills the committed skeleton), `tests/test_engine_profiles.py` |
| P1-b | `jev_mac_manager.sh`, `Makefile`, `tests/test_mac_manager.py` |
| P1-c | `kev_provider.py`, `daemon_provider.py`, `routing_provider.py`, `tools/evaluate.py`, `tools/router.py`, `tools/training.py`, `lora_training.py`, `doctor.py`, `app.py`, `tests/fake_llama_server.py`, `tests/fixtures/llama/*`, and their tests |
| P1-docs | `README.md` (incl. the legacy `start` text at `:150`, `:774`), `docs/`, `CLAUDE.md`, `.env.example`, RUNBOOK, CONTRIBUTING |

P1-c imports `jev_mcp.engine_profiles` only through the names in section 4. P1-b calls it
only through the CLI in section 5.

## 2. Files and paths

| What | Where | Written by |
|---|---|---|
| Profile table | `<repo>/engine_profiles.toml` (tracked) | hand-edited |
| Model file | `<models_dir>/<repo name after the "/">/<filename>`, e.g. `models/Kev-0.8B-GGUF/Kev-0.8B-Q8_0.gguf` (the existing layout, so today's files are reused) | manager download |
| Active marker | `~/.jev/active_profile`: one line, `<profile name>\n` | `activate` only |

`models_dir` defaults to `<repo>/models`, where `<repo>` is three directories above
`engine_profiles.py` (as in `client_config.REPO_ROOT`). The manager always passes
`--config` and `--models-dir` explicitly, so an editable install pointing at another
checkout never resolves the wrong tree.

## 3. `engine_profiles.toml` schema

```toml
[models."kev-0.8b"]
tier = "fast"                       # "fast" | "smart"
repo = "ggml-org/Kev-0.8B-GGUF"     # ^[\w.-]+/[\w.-]+$
revision = "e551e319d483ff57e1ff208924b349d397cffc1c"   # ^[0-9a-f]{40}$, never a branch
filename = "Kev-0.8B-Q8_0.gguf"     # ^[\w.-]+\.gguf$
sha256 = "27278f34…"                # ^[0-9a-f]{64}$
parallel = 4                        # int 1..16 (bool rejected)
ctx_per_slot = 16384                # int 512..131072
reasoning = "auto"                  # "auto" (no flag) | "off" | "on"
style = "plain"                     # "plain" | "sandbox" (smart-tier prompt style)

[profiles.default]
fast = "kev-0.8b"                   # must name a model with tier = "fast"
smart = "qwen2.5-7b"                # must name a model with tier = "smart"
gate = 0.85                         # optional, float in (0, 1), default 0.85
eval_only = false                   # optional, default false
```

- Model and profile names: `^[a-z0-9][a-z0-9._-]{0,31}$`. The served alias is the model
  name (no separate `alias` field).
- Unknown keys at any level, a missing required key, a wrong type or an out-of-range
  value is a `ProfileError` naming the key. The only top-level tables are `models`
  and `profiles`.
- Contents: models `kev-0.8b`, `kev-4b`, `kev-4b-q8`, `qwen2.5-7b`, `qwen3.5-4b`,
  `qwen3.5-4b-q6`, `qwen3.5-4b-q8`, `qwen3.5-9b`, using the revision and sha256 values
  from `pins.tsv`. Every model uses `ctx_per_slot = 16384`; fast models use
  `parallel = 4` and smart models use `parallel = 2` (the phase-0 settings, so
  profiles are compared at the same context). Kev models: `reasoning = "auto"`,
  `style = "plain"`. `qwen2.5-7b`: `"auto"`, `"sandbox"`. Qwen3.5 models: `"off"`,
  `"sandbox"`.
- Profiles: `default` (kev-0.8b + qwen2.5-7b), `lean` (kev-0.8b + qwen3.5-4b) and
  `balanced` (kev-4b + qwen3.5-4b). The eval-only profiles are `accuracy`
  (kev-4b + qwen3.5-9b), `probe-kev4b-q8` (kev-4b-q8 + qwen3.5-4b), `probe-q35-q6`
  (kev-4b + qwen3.5-4b-q6) and `probe-q35-q8` (kev-4b + qwen3.5-4b-q8).

## 4. Python API (`jev_mcp.engine_profiles`)

The skeleton is committed with this contract. P1-a fills in the bodies and must not
change the signatures.

```python
DEFAULT_PROFILE = "default"
DEFAULT_GATE = 0.85
DEFAULT_CONFIG_PATH: str          # <repo>/engine_profiles.toml
DEFAULT_MODELS_DIR: str           # <repo>/models
ACTIVE_PROFILE_PATH: str          # ~/.jev/active_profile

class ProfileError(ValueError): ...

@dataclass(frozen=True)
class ModelSpec:
    name: str; tier: Literal["fast", "smart"]; repo: str; revision: str
    filename: str; sha256: str; parallel: int; ctx_per_slot: int
    reasoning: Literal["auto", "off", "on"]; style: Literal["plain", "sandbox"]
    def path(self, models_dir: str) -> str      # contained in realpath(models_dir), else ProfileError
    def url(self) -> str                        # https://huggingface.co/<repo>/resolve/<revision>/<filename>

@dataclass(frozen=True)
class Profile:
    name: str; fast: ModelSpec; smart: ModelSpec; gate: float; eval_only: bool

@dataclass(frozen=True)
class ProfileTable:
    models: Mapping[str, ModelSpec]; profiles: Mapping[str, Profile]
    def get(self, name: str) -> Profile         # ProfileError if unknown or invalid

def load_profiles(path: str = DEFAULT_CONFIG_PATH) -> ProfileTable                  # raises ProfileError
def read_active_name(path: str = ACTIVE_PROFILE_PATH) -> str                       # missing file -> DEFAULT_PROFILE; invalid -> ProfileError
def active_profile(config_path: str = DEFAULT_CONFIG_PATH,
                   marker_path: str = ACTIVE_PROFILE_PATH) -> Profile | None       # server side: never raises; logs and returns None
def launchd_plist(model: ModelSpec, *, models_dir: str, port: int, label: str,
                  log_path: str, llama_server: str, workdir: str,
                  sleep_idle_seconds: int | None = None) -> bytes                  # plistlib bytes
```

Server rule: when `active_profile()` returns None, the gate is `DEFAULT_GATE` and
responses report `"profile": null`. A tool never fails because the marker is missing.

## 5. CLI: `python -m jev_mcp.engine_profiles`

The CLI uses argparse and prints only from `main()`. Errors go to stderr as one line.
Every subcommand accepts `--config PATH` and `--models-dir PATH`.

**Exit codes:**
- `0`: success.
- `1`: a runtime failure: the daemon is unreachable, its `/props` don't match, a
  timeout, or an I/O error.
- `2`: invalid input: a bad name, a bad TOML file, an unknown profile, a failed
  validation, an eval-only profile without `--allow-eval-only`, or an argparse
  usage error.

### `resolve [PROFILE] [--allow-eval-only]`
If PROFILE is omitted, the name comes from `read_active_name()`. On stdout, one
`key=value` per line, in exactly this order and with no other lines:

```
profile=<name>
fast_model=<name>
fast_path=<absolute path>
fast_url=<https url>
fast_sha256=<hex>
smart_model=<name>
smart_path=<absolute path>
smart_url=<https url>
smart_sha256=<hex>
```

Values never contain a newline or control character. Paths are rejected with exit 2
if they contain one.

### `plist PROFILE --tier fast|smart --port N --label L --log PATH --llama-server PATH --workdir PATH --out PATH [--sleep-idle-seconds N] [--allow-eval-only]`
Writes the plist to `--out` atomically (a temp file in the same directory, then
`os.replace`) and prints nothing. `--label` must match `^[A-Za-z0-9._-]+$`; `--log`,
`--llama-server`, `--workdir` and `--out` must be absolute paths. The plist has
these keys:
- `Label`
- `ProgramArguments`
- `RunAtLoad` and `KeepAlive`, both `true`
- `StandardOutPath` and `StandardErrorPath`, both set to `--log`
- `WorkingDirectory`, set to `--workdir`

`ProgramArguments` has exactly this order:

```
<llama-server> -m <path> --host 127.0.0.1 --port <N> --alias <model name>
--parallel <parallel> -c <parallel * ctx_per_slot> --log-verbosity 3
[--reasoning off|on]            # omitted when reasoning = "auto"
[--sleep-idle-seconds <N>]      # only when given
```

`--log-verbosity 3` is the ledger ruling: it puts the model, KV and compute buffer
lines in the startup log. The first P1-M start checks that they appear. Each
argument is its own `<string>`, so the manager's `plist_port` awk and
`lora_training.find_launchd_label` keep working; P1-b and P1-c each test their reader
against a generated plist.

### `activate PROFILE --fast-port N --smart-port N [--timeout SECONDS=120] [--marker PATH] [--allow-eval-only]`
For each tier, it polls `GET http://127.0.0.1:<port>/props` once a second until it
gets a 200 response or the timeout runs out; a 503 means the model is still
loading. It then checks:
- `realpath(model_path)` equals `realpath(ModelSpec.path(models_dir))`
- `total_slots` equals `parallel`
- `default_generation_settings.n_ctx` (falling back to a top-level `n_ctx`) equals
  `ctx_per_slot`. The live daemon reports the context per slot: `-c 65536` with 4
  slots shows 16384.

If every check passes, it writes the marker atomically and prints
`active profile: <name>`. Otherwise it prints one stderr line per mismatching field
(`<tier> <field>: expected X, served Y`), exits 1 and leaves the marker untouched.

## 6. Manager flow (P1-b)

`make start [PROFILE=x]` runs `./jev_mac_manager.sh hybrid [x]`:

1. If x is given, it must match `^[a-z0-9][a-z0-9._-]{0,31}$`; otherwise exit 2.
2. Run `resolve [x] --config "$PWD/engine_profiles.toml" --models-dir "$PWD/models"`.
   Read the output with `while IFS== read -r k v` and a `case` allow-list of the
   section-5 keys, never with `eval` or `source`. A non-zero exit from the resolver
   stops the manager with that code. Print `Profile: <profile>`.
3. For each tier whose path is missing: run
   `curl --fail --location --proto '=https' --proto-redir '=https' -o <path>.part <url>`,
   then `shasum -a 256` and compare the result with `<tier>_sha256`. On a match, run
   `mv` to the final path. On any failure, run `rm -f` on the `.part` file and exit 1;
   no `.gguf` is left behind.
4. Run `check_llama_cpp`. Run the `plist` subcommand for each tier, using the
   current labels, logs and ports, with `--out <live plist path>.new`, **before**
   stopping anything. A failed `plist` removes any `.new` files and leaves the
   running daemons and live plists untouched. Then stop all three agents; `stop`
   still reads the OLD live plists, so the old ports are the ones freed. Remove
   the single-daemon plist (as today), `mv` each `.new` onto its live path, and
   start both agents. (Amended twice after the P1-b reviews.)
5. Run `activate <profile> --fast-port $JEV_FAST_PORT --smart-port $JEV_SMART_PORT`.
   A non-zero exit makes `make start` exit non-zero.

**Python:** the CLI runs as `"${JEV_PROFILES_PYTHON:-$PWD/.venv/bin/python}" -m jev_mcp.engine_profiles`
with `PYTHONPATH="$PWD/src"` prepended. Tests use `JEV_PROFILES_PYTHON` as their shim
point. `JEV_ALLOW_EVAL_ONLY=1` adds `--allow-eval-only` to `resolve`, `plist` and
`activate`.

**Legacy:** `install` keeps its hard-coded Kev-0.8B/Qwen2.5-7B single-daemon path
(`make install` runs `hybrid` right after, so the dual-daemon setup wins). `update`
restarts both daemons through the profile flow, on the active profile (default without a
marker). `start <0.5b|7b>` exits 2 and points to `make start PROFILE=…`.

## 7. Cross-lane invariants (P1-c)

- **Model identity is `/props` `model_path`.** It is cached with a TTL; a TTL of 30
  seconds is suggested. Temperature keys stay `model_key(model_path)`, the lower-case
  file name, so the existing calibration entries and the legacy `qwen2.5-7b` prefix
  keep matching. With `--alias`, a chat reply's `model` field becomes the alias, so
  that field must **not** overwrite the provider's model id any more. This closes the
  phase-0 deferred minor "chat-reply path stores any non-UNKNOWN id verbatim".
- **Responses** add `{"profile": <name|null>, "fast": <model_key|null>, "smart": <model_key|null>}`.
- **Plain or sandbox prompt:** the choice comes from the served model's `style`. To
  find it, take the active profile's smart `ModelSpec` whose `filename` matches the
  served `model_path`. With no match, or with no profile, use `"sandbox"`: untrusted
  state is never sent without the sandbox by default.
- **Gate:** `routing_provider.py:108`, `tools/evaluate.py:111` and `tools/router.py:195`
  read `profile.gate`. When no profile is active they use `DEFAULT_GATE`.
- **Fake server:** P1-c adds `default_generation_settings.n_ctx` to the recorded
  `/props` responses, keeping the top-level `n_ctx` so phase-0 tests stay green. It
  also adds a scenario where the served model differs from the requested one.
  P1-a's own tests mock `urllib.request.urlopen` and do not use the fake server.
