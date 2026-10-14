# Engine profiles: swap the fast and smart models safely, by restart then live

Revision 2: folds in four independent reviews (architecture, ML measurement,
security, silent failures) of revision 1.

## Context

jev-mcp runs two local `llama-server` daemons: Kev-0.8B (fast, 8080) and
Qwen2.5-7B (smart, 8081), both hard-coded in `jev_mac_manager.sh`. The goal is
**better accuracy and lower memory** on an M2 Pro 16 GB, with model pairs
switchable by restart (phase 1) and later live (phase 2).

The reviews changed the plan's shape. A swap is only safe if three things are
true first, and the current code does not guarantee any of them:

1. a model that answers badly is **noticed** (today an all-zero smart answer, a
   failed escalation, or a stale `unknown_model` all pass as SUCCESS);
2. calibration is **fitted on the smart engine and keyed to the model that
   produced it** (today it is fitted through the router, so the fast engine's
   scores contaminate it, and one global temperature serves every model);
3. a comparison **measures the routed pair, with real memory** (today
   `make eval` scores each engine alone and nothing measures resident memory).

So the work is **Phase 0 (harden + measure, current models) → Phase 1
(profiles + restart) → Phase 2 (live swap)**. Phase 0 improves the system even
if no model is ever swapped.

### Candidates (verified on Hugging Face)

| Model | Repo | Q4_K_M | Notes |
|---|---|---|---|
| Kev-0.8B (today, fast, Q8_0) | `ggml-org/Kev-0.8B-GGUF` | 0.78 GB | baseline |
| Kev-4B | `ggml-org/Kev-4B-GGUF` | 3.03 GB (Q8_0 4.48) | official GGUF |
| Qwen2.5-7B (today, smart) | `bartowski/Qwen2.5-7B-Instruct-GGUF` | 4.4 GB | baseline |
| Qwen3.5-4B | `bartowski/Qwen_Qwen3.5-4B-GGUF` | 3.0 GB (Q6 3.8, Q8 4.6) | thinking model |
| Qwen3.5-9B | `bartowski/Qwen_Qwen3.5-9B-GGUF` | 6.2 GB | thinking model; eval-only until phase 2 |

`Qwen3-7B-Instruct` and Kev-0.5B from the original suggestion are not
candidates (the first does not exist; the second is an unofficial prototype).
Architectures (and so KV-cache cost) of Kev-4B and Qwen3.5 are **unverified**;
they are measured from the `llama-server` startup log, not guessed.

## Phase 0: harden and measure (no model change; own PR)

### 0a. Make bad answers visible
- `daemon_provider._normalize_logprobs` (`:174-176`): an all-zero result on the
  smart tier is "no answer", not a score. Return `{}` so the existing
  unavailable/degraded paths fire; log the true+false mass.
- `routing_provider.py:105-111`: a question that was escalated but not answered
  by smart is marked `escalation_failed`; results carry `served_by`; the tool
  status is not plain `SUCCESS` when any escalation failed.
- `provider.post_json` (`:138`): keep the HTTP status and body in the log.
- `kev_provider._parse_answer` (`:44-48`): a missing `noul` key returns `{}`,
  not `0.0`. Contract test against a recorded real response.
- `daemon_provider.py:364-367`: take the model id from both the real and the
  prior call; do not cache on mismatch or under `unknown_model`.

### 0b. One temperature accessor, fitted on the smart engine
- `router_config.py`: `get_temperature(config, model_id)` and
  `with_temperature(config, model_id, value)`. Entries keyed by served model id
  plus file identity (size). The legacy `fitted_temperature` applies **only**
  to `qwen2.5-7b`; every other model starts at 1.0 and is flagged
  `uncalibrated`.
- Route every reader/writer through it: `daemon_provider._saved_temperature`,
  `tools/calibrate.py:222,402-444`, `app.py:268-272`, `server.py:49` prompt,
  `tests/evals/usage/seed_home.py:26`.
- `jev_calibrate_threshold` fits through `smart_provider`, not the router
  (`calibrate.py:347-354`).

### 0c. Measurement foundation
- `tests/run_evals.py`: dump per-row scores for both engines (JSONL), label
  results from the served model id (not the hard-coded strings at `:173-180`),
  and add an **offline routing simulation**: routed accuracy and escalation
  rate as a curve over the gate. No extra inference.
- Deterministic salt option for evals (`daemon_provider.py:309`) so reruns
  differ only by the model.
- `tests/evals/memory.py` (or a `make eval-memory` target): record the
  `model buffer` / `KV buffer` / `compute buffer` lines from the llama-server
  logs, `footprint -p <pid>` after a max-size request, and swap/pageout deltas
  (`sysctl vm.swapusage`, `vm_stat`). Not `ps -o rss`.
- Baseline: run the full table for the current pair and commit it.

### 0d. Shared test fixture (built first; phase 0 tests need it)
- `tests/fake_llama_server.py`: a loopback HTTP server that replays recorded
  responses on `/v1/chat/completions`, `/v1/systemone`, `/health` and
  `/props`: think-first token, 400 context overflow, missing `model`, wrong
  Kev shape, slow/503 load, and a swap-mid-batch model change. It is a
  fixture, not product code, and is reused by phases 1 and 2.
- It exists to fix the pattern in `docs/TODO.md` where a mock "echoes the
  caller's model id" and lets a real bug pass; recorded real responses are
  captured once from the live daemons and checked in.

## Phase 1: profiles + restart (own PR)

### 1. Profiles file and resolver
- `engine_profiles.toml` (repo root). Per model: `tier`, `repo`, **`revision`
  (40-hex commit, never `main`)**, `filename`, **`sha256` (mandatory)**,
  `parallel`, `ctx_per_slot`, `reasoning` (`off|on`), `alias`. No `params_b`,
  no free-form `extra_args` (typed fields only; unknown keys rejected).
  Per profile: `fast`, `smart`, `gate` (default 0.85).
- Profiles: `default` (kev-0.8b + qwen2.5-7b), `lean` (kev-0.8b +
  qwen3.5-4b), `balanced` (kev-4b + qwen3.5-4b). `accuracy` (kev-4b +
  qwen3.5-9b) is **eval-only until phase 2**. Extra eval-only probes:
  kev-4b Q8_0, qwen3.5-4b Q6_K/Q8_0 (quantization confound).
- `src/jev_mcp/engine_profiles.py`: frozen dataclasses; stdlib `tomllib`;
  allow-list validation (names `^[a-z0-9][a-z0-9._-]{0,31}$`, repo
  `^[\w.-]+/[\w.-]+$`, filename `^[\w.-]+\.gguf$`, bounded ints); path
  containment under the models dir via `realpath`; CLI
  (`python -m jev_mcp.engine_profiles resolve|plist|activate`) following
  `client_config.py` (argparse, prints only from `main()`).
- **The plist is generated in Python with `plistlib`**, not the bash heredoc;
  it always includes `--host 127.0.0.1`, `--alias <model>`, `--reasoning off`
  where set, the TOML's `--parallel` / `-c`, and (if the phase-1 spike below
  passes) `--sleep-idle-seconds`.

### 2. Manager
- `hybrid [profile]` uses the resolver (the venv exists by then). Legacy
  `install`/`update` keep their hard-coded Kev-0.8B/Qwen2.5-7B paths: the
  resolver is not available before the venv. The legacy `start <0.5b|7b>`
  path is already broken (`:462`, unbound `$2` under `set -u`; `install 7b`
  puts Qwen on the fast port): freeze it with an explicit error and fix
  README `:150,774`. No "legacy keeps working" promise.
- `PROFILE` is validated with the same regex in the shell and in Python;
  resolver output is read with `while IFS== read -r k v` and a `case`
  allow-list, never `eval`/`source`.
- Downloads: `curl --fail --proto '=https' --proto-redir '=https'` to a
  `.part` file, verify SHA-256, then `mv`. A partial or error-page file can no
  longer be trusted by the `[ ! -f ]` check.
- `make start PROFILE=<name>`; **no argument = the active profile** (sticky,
  printed at start), a fresh install = `default`. An invalid
  `~/.jev/active_profile` is an error, not a silent fallback.
- `active_profile` is written by the Python CLI **only after** `GET /props`
  on each daemon shows the expected model path, `n_ctx` and `total_slots`;
  `make start` exits non-zero on mismatch. It is a convenience marker; the
  running daemon's reported model is the truth.

### 3. Server side
- `KevProvider.current_model_id` and the daemon's model come from `/props`
  with a TTL, not constants; tool responses include
  `{profile, fast, smart}`.
- `SMALL_MODEL_MARKER` (`daemon_provider.py:15,298`): replace the filename
  heuristic with the model's declared `style` (`plain|sandbox`).
- Gate: `routing_provider.py:90` and the two other 0.85 sites read the
  profile's `gate`.
- `doctor.py`: served model vs profile (from `/props`), a one-question
  scoring probe (first token is a target, mass > 0.9), the loading (503)
  state shown as "loading", exit 1 on mismatch.
- `jev_train_lora`: refuses a base model that does not match the active smart
  model; training data must include the empty think block when reasoning is
  off. Qwen3.5 converter/mlx_lm support is **unverified and out of scope**;
  documented, not built.
- Spike (throwaway, 15 min): does `--sleep-idle-seconds` release memory on a
  plain (non-router) daemon? If yes, a large part of the memory win ships in
  phase 1 as one flag.

### 4. Tests (written first)
- `test_engine_profiles.py`: resolve; hostile names (`../x`, `a;b`, newline,
  `&`, `<`) rejected; unknown key; wrong tier; sha/revision required.
- Plist round-trips through `plistlib` with special characters; asserts
  `--host 127.0.0.1`, `--alias`, `--reasoning off`, slots/ctx.
- Download: HTTP error and hash mismatch delete the `.part` and leave no
  `.gguf`.
- The **loopback fake `llama-server`** fixture (built in 0d) replays recorded responses: think-first
  token, 400 context overflow, missing `model`, wrong Kev shape, slow load.
  Each must surface as a degraded/error result, never as `SUCCESS` with
  zeros.
- Temperature accessor: legacy applies only to qwen2.5-7b; per-site tests for
  all five call sites; swap-mid-batch prior-cache test.
- `test_mac_manager.py`: add shims for `curl`, `llama-server`, `sysctl` and an
  env override for the resolver/repo root (the existing sandbox has none).
- Gate: `make check`; 95% per touched file, 85% overall.

### 5. Decision (live daemons, not CI)
For each profile: `make start PROFILE=x` → calibrate through the smart engine
→ score dump → offline gate sweep → triage evals → browser eval → memory
record. One table in `docs/`.

- **Primary metric:** routed accuracy at a fixed escalation/latency budget.
- **Test:** paired (McNemar or paired bootstrap on per-row differences) with a
  non-inferiority margin **declared before running**; each run repeated 3x.
- **Data hygiene:** HF seed 0 and the synthetic 30 are *development* sets
  (the triage prompt was tuned on them against the incumbents). Report on
  fresh seeds plus a small human-labelled slice that includes positives (the
  HF sets measure only false escalation).
- **Reasoning-off proof per Qwen3.5 profile:** `POST /apply-template` ends in
  an empty think block; per-row true+false mass median > 0.9 and < 1% of rows
  under 0.05; prior mass logged.
- **Default changes only** to a profile that is non-inferior on the primary
  metric and triage agreement **and** uses less resident memory (footprint +
  swap), at the same fixed context. Otherwise `default` stays and the table
  goes to you.

### 6. Docs
README (CLI, env, profiles), `docs/CODEMAPS/*`, `CLAUDE.md`, RUNBOOK,
CONTRIBUTING, `.env.example`, and the remaining "0.5B/7B" strings
(`routing_provider.py:22,63,70,101`, `tools/evaluate.py:425-531`,
`Makefile:43,78`).

## Phase 2: live swap via router mode (after phase 1 merges)

Spike first (throwaway), answering: (1) Kev's `POST /v1/systemone` through the
router with a `model` field; (2) logprobs parity under the router; (3) load
time for 3 GB / 6 GB models vs the 30 s Kev / 60 s smart timeouts; (4)
`--no-models-autoload` behaviour and that the response `model` equals the
request; (5) whether `/health` reflects loaded state; (6) auth on `/models/*`.

If it passes (own detailed plan then):
- Presets INI generated from `engine_profiles.toml` into `~/.jev/` (0600,
  absolute validated paths, `--models-max 1`, `--no-models-autoload`, no
  `--models-dir`/URLs/hf ids).
- `--api-key-file ~/.jev/api_key` (0600) and a Bearer header in `post_json`.
- Tool `jev_manage_engine_profile` (`set`; `view` as a separate read-only
  tool): name must be in the loaded profile table, never a path/URL/id;
  `set` loads from the table, waits for `GET /models` = `loaded`, verifies
  the served model, then activates. Annotations `(read-only False,
  destructive True, idempotent True, open-world False)`; rate-limited;
  added to `tests/test_tool_surface.py`.
- Known breakages to handle: `lora_training.find_launchd_label` (matches the
  model path in the plist), `pkill --port` and `_get_daemon_pid` (router
  child processes), prior cache across swaps where the pid is constant.
- If spike question 1 fails: only the smart tier uses the router; the fast
  tier keeps restart-swap.

## Execution strategy (parallel lanes, subagent-driven)

### Rules
- **Dependency graph first.** Work is split into lanes; a lane may run in
  parallel only if its write surface (files) is disjoint from every other
  running lane's. Reads, searches, status and metadata queries are always
  batched.
- **Ruling (two skills disagree):** subagent-driven-development says never
  run implementers in parallel; parallel-execution-optimizer allows it in
  isolated worktrees. Ruling: parallel implementers only in **separate
  worktrees on separate branches with disjoint files**; anything that
  shares a file runs sequentially. I merge lanes and re-run the integrated
  suite before the next gate. Cost if wrong: a merge conflict, caught by
  the integration run.
- **Fresh implementer per task**, given only its task brief, the interfaces
  it touches and the global constraints; never the session history.
  Implementers never dispatch subagents.
- **Reviews:** per task, one reviewer checks spec compliance and quality
  from a diff file; fixes loop up to 5 rounds (rounds 1-3 resume the
  implementer, 4-5 a fresh one on a stronger model) with a scoped
  re-review each round. One whole-branch review per phase on the strongest
  model, plus read-only specialist lanes run in parallel on the phase diff:
  `ecc:security-reviewer`, `ecc:silent-failure-hunter`,
  `ecc:python-reviewer`, and `ecc:mle-reviewer` for phase 0c and phase 1 §5.
- **Model tier per role (always set explicitly):** plan text that contains
  the code to write (TOML, fixtures, mechanical call-site rewires) → cheapest
  tier; multi-file integration (manager bash, providers, doctor) → standard;
  design judgment and the final review → most capable; reviewers never below
  the mid tier.
- **Ledger.** Each phase keeps a progress ledger in its plan workspace
  (`<repo>/.superpowers/sdd/<plan>/progress.md`, git-ignored): baseline and
  rulings written once, then one line per dispatched wave and per completed
  task, including commit ranges. After a compaction the ledger and `git log`
  are the truth. Every ruling is listed in the final message.
- **Stops** (only these): an irreversible or destructive step; a
  security-sensitive action; a side effect outside the worktree (a merge or
  a push to `main` is the user's: PRs are opened, not merged); a plan so
  broken every path is a guess.
- **Exclusive lanes.** Live-daemon work (calibration, evals, latency and
  memory measurement, the `--sleep-idle-seconds` spike) runs **alone and
  one profile at a time**: other lanes' test runs would distort latency and
  memory, and two profiles cannot share 16 GB. No parallel lane runs
  during a measurement.

### Lane matrix

| Lane | Parallel? | Write surface | Needs | Verification |
|---|---|---|---|---|
| **P0-D** fake llama-server fixture (0d) | first, alone (tiny) | `tests/fake_llama_server.py`, recorded responses | live daemons once, to record | fixture self-tests |
| **P0-A** make bad answers visible (0a) | yes, own worktree | `daemon_provider.py` (`:174`, `:364`), `routing_provider.py`, `provider.py`, `kev_provider.py`, their tests | P0-D | covering tests; per-file 95% |
| **P0-B** temperature accessor (0b), non-provider parts | yes, own worktree | `router_config.py`, `tools/calibrate.py`, `app.py`, `server.py`, `tests/evals/usage/seed_home.py` | none | covering tests |
| **P0-B2** the one `daemon_provider._saved_temperature` rewire | after P0-A merges | `daemon_provider.py` | P0-A, P0-B | covering tests |
| **P0-C** score dump, routing sim, memory recorder (0c) | yes, own worktree | `tests/run_evals.py`, `tests/evals/memory.py`, `Makefile` target | none | unit tests; one dry run |
| **P0-M** baseline measurement | **exclusive, serial** | `docs/` table | P0-A/B/B2/C merged, live daemons | table committed |
| **P0-R** phase review | after merge; reviewers parallel | none (read-only) | merged phase 0 | findings adjudicated |
| **P1-pins** look up `revision` + `sha256` for every candidate (HF API) | yes, read-only | none (feeds TOML) | none | values cross-checked against the file listing |
| **P1-a** resolver, TOML, plist generator | yes, own worktree | `engine_profiles.toml`, `src/jev_mcp/engine_profiles.py`, `tests/test_engine_profiles.py` | P1-pins; **CLI contract frozen first** | tests incl. hostile names, plist round-trip |
| **P1-b** manager and download hardening | yes, own worktree, **against the frozen CLI contract** | `jev_mac_manager.sh`, `Makefile`, `tests/test_mac_manager.py` | P1-a contract | shimmed sandbox tests |
| **P1-c** server side (`/props` ids, gate, style, doctor) | yes, own worktree | `kev_provider.py`, `daemon_provider.py`, `routing_provider.py`, `doctor.py`, tests | P0 merged | covering tests + fake server |
| **P1-int** integrate a, b, c | sequential | none (merge) | all three | `make check`; manager end-to-end against the fake server |
| **P1-docs** docs and codemaps | after P1-int, parallel with reviews | `README.md`, `docs/`, `CLAUDE.md`, `.env.example` | P1-int | link and command accuracy |
| **P1-R** phase review | after P1-int; reviewers parallel | none | merged phase 1 | findings adjudicated |
| **P1-M** decision measurement | **exclusive, serial** | `docs/` table | P1-int, models downloaded | table + decision rule applied |
| **P2-spike** router-mode spike | **exclusive** | none (throwaway) | P1 merged | answers to the six questions |

### Waves
1. P0-D, in parallel with P1-pins (read-only).
2. P0-A, P0-B, P0-C together (three worktrees, disjoint files), then P0-B2.
3. Merge, run the full suite, then P0-M alone, then P0-R reviewers together.
   Open the phase-0 PR.
4. P1-a, P1-b, P1-c together once the CLI contract is written down; then
   P1-int; then P1-docs and P1-R together.
5. P1-M alone. Open the phase-1 PR.
6. P2-spike alone; its own plan follows.

### Background and gating
- Large model downloads (3 to 6 GB each) can run in the background during
  waves 2 to 4 because they are network-bound and touch no code, but only
  after `df` shows room and the SHA-256 pins exist; each verifies its hash
  before use. They are not started during phase 0 unless you say so.
- A background download or test session is polled deliberately and never
  left running past the turn that started it.

## Execution notes
- Three PRs (phase 0, phase 1, phase 2), each in a worktree on a feature
  branch; `main` requires `build-and-test`. Worktree setup per the saved note
  (submodule init, `.venv` symlink, `PYTHONPATH=src`). Parallel lanes get
  their own worktree each, branched from the same base, and are merged into
  the phase branch by me, never by a lane.
- First commit: this design as
  `docs/superpowers/specs/2026-10-14-engine-profiles-design.md`.
- Model downloads are large (3-6 GB each) and happen only when a profile is
  first started; none during phase 0.

## Verification
1. Phase 0: `make check` green; a loopback fake daemon returning think-first
   tokens yields a degraded result, not zeros; `jev_manage_temperature`
   `set`/`reset` still change scores; baseline table committed.
2. Phase 1: `make start PROFILE=default` → `jev-mcp doctor` exit 0, served
   model matches; `make eval` numbers within noise of the phase-0 baseline.
3. `make start PROFILE=lean` → doctor probe passes; a live
   `jev_evaluate_batch` shows non-flat scores; calibration writes an entry
   for qwen3.5-4b and leaves qwen2.5-7b's untouched; killing the daemon or
   corrupting the GGUF makes `make start` fail non-zero and doctor exit 1.
4. Comparison table exists; the decision rule has been applied and its
   verdict recorded.
5. `make start` with no argument restarts the active profile and prints it.
6. Parallel execution: after every merge of lanes, the integrated branch
   passes `make check` (not just each lane's tests); no lane touched a file
   outside its row in the lane matrix (`git diff --stat` per lane); the
   ledger names every dispatched wave and completed task; no measurement ran
   while another lane was executing.
