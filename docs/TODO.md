# TODO: root causes to fix

Found during the 2026-10-10 documentation and benchmark review. Each item lists the root cause, the evidence, and a fix direction. "Verified" means reproduced against the live daemons; "from source" means read in the code but not yet exercised.

## P0: tools that return wrong results in the default dual-engine setup

- [x] **Module-level provider points at the Kev daemon on the wrong path.** *(fixed: now `RoutingProvider()`; router `max_tokens` is the most restrictive engine, 8192; verified live for `compact` and `calibrate`)*
  `server.py:117` sets `provider = DaemonProvider()`, which falls back to `JEV_DAEMON_PORT` (default `8080`). In the default setup 8080 is the Kev daemon, which does not answer meaningfully on `/v1/chat/completions`, so scores come back near 0.
  - Affected tools (they use the module-level `provider`): `evaluate` (`server.py:170`), `optimize-prompt` (`:474`), `calibrate` (`:547`), `compact` (`:1492`).
  - Verified: `calibrate` reports AUC 0.500 and 0% recall on the 88-row golden set; `compact` drops every chunk, even an obviously relevant three-sentence state.
  - Not yet tested directly: `evaluate`, `optimize-prompt` (same provider, same expected failure).
  - Unaffected (they build `RoutingProvider()` locally): `model-router`, `handoff`, `read-file`, `scan-repo`.
  - Fix: use `RoutingProvider()` (or an explicit smart/fast provider) instead of the bare `DaemonProvider()`; add a test that fails when the module-level provider resolves to the fast port.

- [x] **Email-triage compaction is a no-op (key mismatch).** *(fixed: reads `compacted_state`; test uses the real return shape)*
  `email_triage/mcp_tool.py:50-51` reads `compressed_context`, but `jev_compact_context` returns `compacted_state` (`server.py:1522`). Long bodies are never compacted.
  - The unit test passes only because its mock returns the caller's key (`tests/test_email_triage_mcp.py:110`).
  - Fix: read `compacted_state`; make the test use the real return shape; fix the provider item above first, otherwise the result would be an empty string.

- [x] **`calibrate` reports "Optimal" at 0% recall.** *(fixed: marked `❌ Unusable` when decisions are automated but no positive is caught)*
  In `evaluate_thresholds`, precision defaults to 1.0 when nothing is predicted positive, so a model that never says True is labelled `✅ Optimal` / `✅ Safe`.
  - Fix: require a minimum recall (or at least one true positive) before recommending a threshold.

- [x] **`run-browser-agent` termination is not evidence-gated: `done` arrives both prematurely and never.** Measured 2026-10-11 on self-verifying local pages (`.reports/browser-agent-eval/results.md`): 0/8 runs terminated correctly. With `MISSION_COMPLETE` visible in the page title and text, the smart engine chose CLICK over DONE in 7/7 decisions (instrumented replay, `probe_d2.py`); conversely, a two-field goal ended `done` after one field was filled, with no success marker anywhere on the page. The harness accepts DONE purely on the model's word. *(fixed same night: `browser_agent.choose_locally` forces DONE when the goal's stated success marker is in page text/title and demotes a premature DONE to the runner-up operation; the criterion is also stated in the prompts. Re-measured across three rounds: termination 0/8 → 7/8 — never-DONE, premature-DONE, and the select thrash all eliminated. Rare residual: the engine sometimes answers BLOCKED on an opening decision (one T1 flake, clean on retry) — engine-level nondeterminism, not the gate.)*
  - Fix directions: gate DONE on the goal's stated success marker appearing in page text/title; surface the marker and the last few identical actions in the operation prompt. *(done)*
- [ ] **`run-browser-agent` repeats its last action instead of progressing.** Same measurement: single-field typing goals re-filled the same field 4× and never clicked Submit; a 14-option select thrashed between two Lisboa variants to the 20-step limit with success visible ~9 times; the only anti-repeat is the loop's 3×-no-page-change auto-block, which text-changing repeats defeat (and which masks repeats that change nothing as `blocked`). Dense-click discrimination itself: 2/4 (one one-word miss by the 7B among 50 labeled elements after correct escalation; one below-fold target never found despite scrolling). *(mostly fixed same night: tried-element flags in target listings, alternating/no-change/repeated-pick prompt notes, two-stage narrowing over the top-5 for listings >12 candidates (shortlist answers expanded back to full-candidate probabilities — the submodule's `validate_choice` requires exactly the offered id set; the live eval caught the first version violating it), a scroll hint on large listings, a post-fill progression note, and an early exit from triple no-effect WAITs. Re-measured: dense 3/4 — the steel-vs-ceramic case now recovers to correct; typing cases complete the fill→submit flow, including both fields. Remaining: D4-style deep rows — the engine scrolls once, samples wrong near-miss elements, and blocks before reaching the target row; needs sustained scrolling or deeper-row discrimination.)*
  - Fix directions: exclude or down-weight the immediately-previous action in target options; detect alternating-pick loops; two-stage narrowing (top-k) for ~50-element listings. *(all three done; sustained-scroll behavior for below-fold targets remains open)*

## P1: correctness and safety of training and daemon control

- [x] **`train` fuse step cannot work for the default model.** *(fixed: `lora_training.build_fuse_script` runs fuse `--dequantize` → `convert_hf_to_gguf.py` → `llama-quantize`; run end to end on Qwen 7B on 2026-10-10. The converter is fetched automatically for the installed llama.cpp commit and refetched after an upgrade, see README. The full `train` tool path was then run end to end on 2026-10-11: one call (120 iters, fuse) trained, fused, converted, and quantized Q4_K_M; swapped the live-served GGUF (md5 changed, original kept as `.bak`); the smart agent restarted and served the fine-tuned file (a direct `:8081` completion named the swapped path, and an ambiguous evaluate escalated to it); `action="restore"` put the original back byte-identically. Found en route and fixed: dataset staging was skipped whenever the dataset was already named `train.jsonl`, so a stale or dangling `~/.jev/training/train.jsonl` was trusted — staging now always runs unless the dataset is the staged file itself)*
  It runs `mlx_lm.fuse --export-gguf`, which `mlx_lm` 0.31.3 limits to unquantized `llama`/`mistral`/`mixtral`. The default base is a quantized `qwen2` model.
  - Fix: `mlx_lm.fuse --dequantize --save-path <dir>`, then llama.cpp `convert_hf_to_gguf.py`, then `llama-quantize`. The converter is not shipped by Homebrew's `llama.cpp`; pin it to the installed build's commit.

- [x] **`train` restart logic targets the wrong process.** *(fixed: restarts only the launchd agent whose plist serves the replaced file, with `launchctl kickstart -k`)*
  It kills the first `llama-server` PID it finds and relaunches that command with `nohup`. With two daemons this may be Kev rather than the model being replaced, and the launchd agents use `KeepAlive`, so launchd also respawns it.
  - Fix: restart the specific launchd agent (`launchctl kickstart -k gui/$UID/<label>`); never `nohup` a second copy.

- [x] **`train` has a backup but no restore.** *(fixed: `action="restore"`; the first `.bak` is never overwritten)*
  `mv <target> <target>.bak` exists (`server.py:1191`), but nothing restores it.
  - Fix: add a restore action (move `.bak` back, restart the agent) and document it.

- [x] **`train` non-fuse path downloads and runs a script from `master`.** *(fixed: `fuse=False` now only reports the adapter path)*
  It `curl`s `convert-lora-to-ggml.py` from llama.cpp `master` and executes it.
  - Fix: remove, or pin to a commit and verify a checksum.

- [x] **`train` uses fixed `--iters 500`.** *(fixed: `iters` argument, default 500; batch size and layers still fixed)*
  No way to scale iterations to dataset size; overfits small sets.
  - Fix: expose iterations (and batch size / layers) as arguments.

- [x] **`manage_daemon stop` kills every `llama-server`.** *(fixed: the force-kill is scoped to the port in the agent's plist; `uninstall` now unloads all three agents; covered by `tests/test_mac_manager.py`)*
  `jev_mac_manager.sh:312-313` runs `pkill -9 -f llama-server` after booting out one plist, so stopping one engine stops both.
  - Fix: stop only the targeted agent.

- [x] **Legacy launchd agent is still loaded and failing.** *(script fixed: `hybrid` now deletes the legacy plist; removed on this machine on 2026-10-10. Was still to do: its plist runs `kev.serve --port 8080`, competes with the fast engine, and has crash-looped 855 times; boot it out and delete `~/Library/LaunchAgents/com.jev.mlx_server.plist`.)*
  `com.jev.mlx_server` shows a changing PID with exit status 3 alongside the `_fast` and `_smart` agents.
  - Fix: boot it out and remove its plist during `hybrid`/`install`.

## P1: scoring behaviour

- [x] **XML sandbox prompt is never used for Qwen under `llama-server`.** *(fixed: the smart tier always uses the sandbox; only a model the daemon itself reports as 0.5B gets the plain prompt, and no process list is consulted. README numbers re-measured.)*
  **Decision (2026-10-10): turn it on.** Select the prompt from the configured engine, not `pgrep`. This changes every Qwen score, so re-run `make eval` and the calibration table and update the README numbers in the same change.
  - Follow-ups found while measuring: warm latency rose from about 244 ms to about 662 ms per sample (longer prompt), and the random per-question salt makes scores vary slightly between runs (84 vs 85 of 88 correct). Options: move the static instructions ahead of the salted tag so `llama-server` can reuse the prompt prefix; derive the salt from a hash of the payload so a given input always scores the same.
  `daemon_provider.py:204-208` decides `is_small_model` from `pgrep -fl mlx_lm`; with no `mlx_lm` process the call fails and it defaults to `True`, selecting the simple prompt. The README says the 7B profile is sandboxed. The prompt also silently switches whenever any `mlx_lm` process (e.g. training) is running.
  - Fix: decide from the configured engine/model, not from `pgrep`.

- [x] **Qwen scores are saturated.** (verified) *(fixed in two parts: censored tokens, and `calibrate` now returns `fitted_temperature` and saves it only with `save_to_config=True`. About half of Qwen's scores are still near 0 or 1.)*
  **Decision (2026-10-10): do both, in the form that is best for now.**
  (a) Missing tokens: treat as censored. When `true`/`false` is absent from the top-11 log-probs, use the lowest log-prob in that list as an upper bound instead of the `-9999.0` sentinel, so scores are small but never a false exact 0.0. *(done: with the sandbox prompt, saturation fell from 80% to 49%, exact 0.0 scores from 26 to 0, AUC rose from 0.977 to 0.997 and accuracy at 0.5 from 89.8% to 95.5%.)*
  (b) Temperature: have `calibrate` return the fitted temperature and persist it only when asked (an explicit flag), not silently. A temperature fitted on one question's dataset is global and would change every other question's scores.
  80% of scores on the 88-row set are within 1e-6 of 0 or 1; 26 are exactly 0.0 because the `true` token fell outside `top_logprobs=11` and got the `-9999.0` sentinel. Thresholds between 0.50 and 0.90 behave identically.
  - Fix options: apply the fitted temperature by default after calibration; treat missing tokens as censored rather than zero; report saturation in tool output.

- [x] **`JEV_DAEMON_PORT` default collides with the fast port.** *(fixed: when unset it follows `JEV_SMART_PORT`, default 8081)*
  Both default to `8080`. Fix together with the provider item above (drop the variable or default it to the smart port).

- [x] **`compact` chunks JSON states blindly.** *(fixed: dict states are chunked one top-level key at a time)*
  A JSON state has no blank lines, so it is treated as one blob and cut every 2000 characters (`server.py:1459-1474`), splitting mid-sentence.
  - Fix: chunk by top-level key / list item for dict states.

- [x] **`compact` relevance threshold drops useful context.** *(fixed: default and email-triage threshold are now 0.3, from a 24-pair labelled check where the fast engine reached AUC 0.986 with relevant chunks at 0.32-0.90 and irrelevant at 0.04-0.35; `chunk_scores` are returned. The check set is small and hand-written; re-measure on real content.)*
  With goal "Fine-tune a LoRA adapter on the Qwen 7B model" and `confidence_threshold=0.5`, it kept the training sentence, correctly dropped an unrelated one, but also dropped "After training, fuse the LoRA adapter into the Qwen model with mlx_lm.fuse --dequantize". The default threshold is higher still (0.85), and email triage uses 0.6.
  - Chunks are scored by the fast engine only (`provider.fast_provider`), with the question "Does this text snippet contain information strictly relevant to achieving the goal".
  - Fix options: measure keep/drop accuracy on a labelled set and pick the threshold from that (the golden-set tooling in `tests/run_evals.py` can score it); soften "strictly relevant" in the prompt; escalate borderline chunks to the smart engine instead of dropping them; return the per-chunk scores so callers can see what was close.

- [x] **Tried: a dynamic `optimize-prompt` step inside `compact`.** *(result: not adopted)* On the same 24 pairs, `optimize-prompt` as shipped did not help (margin between lowest relevant and highest irrelevant score -0.09 vs -0.03 baseline) and added about 7 s per call. Choosing the rephrasing with the widest score spread did help (margin +0.08, AUC 1.000), but a static wording change, "directly relevant" instead of "strictly relevant", gave nearly the same result (margin +0.06, AUC 1.000) at no cost, so that was adopted instead.

- [x] **`optimize-prompt` selects the most confident phrasing, not the most accurate.** *(fixed: with a labelled `dataset` it selects by accuracy (`prompt_selection.py`); the single-state mode still selects by confidence but now says so. Live check on 8 password rows: old mode picked a 62.5% phrasing, dataset mode picked 100% vs 87.5% for the original)*
  It scores each variation on one sample state and keeps the one with the highest `max(p, 1-p)`; it never sees labels. In the `compact` experiment this raised relevant and irrelevant scores together.
  - Fix options: accept a small labelled set and pick by accuracy or AUC; or, when unlabelled, pick by score spread across several states.

- [x] **`calibrate` is single-question and in-sample.** *(fixed: rows may carry their own `"question"`; Platt scaling is 5-fold cross-validated when each class has at least 5 rows, and the report states which was used. Persisting the temperature is tracked under the saturation item.)*
  One `question` per call, and the Platt fit is evaluated on the rows it was fitted on.
  - Fix: accept per-row questions; report cross-validated numbers; optionally persist the fitted temperature (the tool currently never writes it).

## P2: smaller issues

- [x] `email_triage/mcp_tool.py`: `except Exception: pass` swallows compaction errors; log a warning instead. The `2000` and `0.6` thresholds are magic numbers. *(fixed)*
- [x] `email_triage/auth.py`: `jwt_decode` is a stub that always raises, so only the static `JEV_MCP_API_KEY` works. No launcher ships for the webhook app. *(fixed: JWTs are verified with PyJWT against a JWKS URL or an HS256 secret, with required expiry and optional audience/issuer; `jev-mcp serve-webhook` launches the app on localhost. Not yet exercised against a real identity provider.)*
- [x] `run-browser-agent`: only checks that `./jev-ultrafast` exists (CWD-relative) and returns a dispatch message. *(fixed: `browser_agent.py` runs the `jev-ultrafast` agent loop with choices from the local router and field text from the smart engine. Verified 2026-10-11 against real Chrome (`chrome://inspect/#remote-debugging` enabled): a goal-completion run on example.com (21.8s) and a three-step click run — "More information…" → `iana.org/help/example-domains`, then two more correct element picks to `iana.org/about` (26.6s) — so real-DOM element picking by the local dual-engine works. Measured 2026-10-11 on self-verifying dense/typing/select pages (`.reports/browser-agent-eval/`, gitignored): mechanics all work and typing text was exact 3/3 into the right fields, dense-click picking 2/4, select first-pick 1/1 — but termination was wrong in 8/8 runs; see the two P0 items below.)*
  - Found on the way: `ultrafast_adapter.local_choose` never showed the model the element labels (options were bare indices) and read page text from a key the agent does not pass, so its choices were blind. Labels are now in the target prompt and the page is passed correctly. The fast engine still sees no per-element state (value, checked, role); add it if element picking proves weak.
- [x] Two unrelated `NoulQuestion`/`ChoiceQuestion` classes (`provider.py` vs `email_triage/core.py`) with the same names. *(fixed: `email_triage/core.py` builds the provider's classes directly)*
- [x] `router_config.py` defaults reference old model names (`claude-3-5-*`, `claude-fable`). *(fixed: `model_catalog.py` fetches models.dev at start-up; buckets keep the user's model and are only moved within the same family, by alert + `apply_updates` or by opt-in `set_auto_update`. Not done: seeding a new install from the connecting client's provider; new installs get Anthropic tiers. Also fixed: `tests/test_routing.py` was writing to the real `~/.jev/router_config.json`; every test now gets a temp config.)*
  **Decision (2026-10-10): discover defaults per client, with a fallback table.** *(fixed 2026-10-11: a `SeedingMiddleware` on the MCP server reads the connecting client from the handshake (`client_info.name`) and, when no user-chosen buckets exist, seeds `b1`-`b4` from a per-client table (`client_defaults.py`, v1 Anthropic-only). Provenance lives in the config: `buckets_source: "detected"|"user"` + `detected_client`; consent actions (`set_bucket`, `apply_updates`, `set_auto_update` on) stamp `"user"`, fieldless existing configs count as user-chosen and are never touched, and background same-family auto-updates preserve `"detected"`. Adding another provider = adding rows to the client table. Note: any tool call that materializes a config file before a Claude-family handshake — e.g. `add_rule` from a non-Anthropic client on a fresh install — makes that fieldless file count as user-chosen, and seeding will never apply afterwards.)*
- [x] `server.py` was 1933 lines; split by tool family. *(fixed: shared state (`mcp`, `provider`, `linter`, `logger`, chat helpers, notices) moved to `app.py`; the eight tool families live in `tools/{evaluate,calibrate,router,training,files,prompts,browser,triage}.py` and reference `app.X` at call time so tests patch `jev_mcp.app`; `server.py` keeps the prompts, `main()` and re-exports (~175 lines). Also fixed on the way: the browser/triage tools sat after the `__main__` guard so the real stdio server never registered them, and `email_triage/mcp_tool.py` lazily imported `jev_mcp.server`, double-importing the `__main__` module at request time.)*
- [x] *(fixed: `requirements.txt` is retired; `pyproject.toml` has `engines` and `dev` extras, CI and the manager install from it, `uv.lock` refreshed.)* `requirements.txt` and `pyproject.toml` list different dependency sets (e.g. `fastapi`, `tree-sitter`, Google clients only in `pyproject`; `mlx`, `kev` only in `requirements.txt`).
  **Decision (2026-10-10): `pyproject.toml` is the source of truth.** Move the engine and dev packages into optional dependency groups there (e.g. `engines`: `mlx`, `mlx-lm`, `kev`; `dev`: `ruff`, `mypy`, `pytest`, `pytest-cov`), switch CI to `pip install -e ".[dev,engines]"`, relock, and retire `requirements.txt`.
- [x] CI: `actions/checkout@v4` and `actions/setup-python@v5` trigger a Node 20 deprecation warning. *(fixed: both on v7; the warning is gone from the main run)*
- [x] Some tests reached a real daemon when one was running (two `compact` tests, and two linter tests whose auto-fixer made a real generation call, taking 9-13 s with model-dependent results). *(fixed: an autouse fixture in `tests/conftest.py` makes unmocked daemon calls fail as in CI; the suite runs in about 6 s instead of 35 s)*
- [ ] Golden dataset: 80 of 88 rows were written in one pass by an LLM and frontier models score 100%; add real, hard cases.

## Review follow-ups: 2026-10-11 security and Python review

The high and medium findings landed in #19 (`c40a822`, `409cbd9`). These are what remains. Items under "Needs a decision" are behaviour changes, so they wait for a call.

### Needs a decision

- [x] **`is_safe_path` was a denylist.** *(fixed: allows the server's working directory, `~/.jev`, the temp dir and `JEV_ALLOWED_ROOTS`; the credential-name denylist still wins inside every root, and the filesystem root never counts as a root. BREAKING: other paths are refused with a message naming the roots. A working directory that is `$HOME` or above it is not treated as a root, so a server started there allows only `~/.jev`, the temp dir, the repo's `models/` and `JEV_ALLOWED_ROOTS`; the repo's `models/` is always allowed so `train` works wherever the server starts.)*
- [x] **Webhook key sat in the Apps Script source.** *(fixed: the template reads it from the Script Property `JEV_MCP_API_KEY`; `setup_gas_workflow` and `jev-mcp setup-gas` no longer take, prompt for or upload the key, and the tool's result carries a `next_step` telling the user to set the property. BREAKING: `api_key` and `--api-key` are gone. Script properties cannot be set through `projects.updateContent`, so setting it is a one-time manual step.)*
- [ ] **`is_important_score` is computed and never used.** `email_triage/core.py::decide_routing` takes it as a parameter and ignores it. Either let it affect routing or remove the parameter.
- [ ] **JWKS mode now needs an audience.** A deployed webhook that sets `JEV_MCP_JWKS_URL` without `JEV_MCP_JWT_AUDIENCE` rejects every JWT. Check what is deployed.

### Security lows

- [x] `email_triage/auth.py`: `auth_header.replace("Bearer ", "")` replaced the prefix anywhere in the header. *(fixed: sliced after the `startswith` check)*
- [x] `email_triage/auth.py`: the HS256 shared secret had no minimum length. *(fixed: under 32 bytes is rejected and logged; BREAKING for a deployed shorter secret)*
- [x] `read-file` had no size cap. *(fixed: reads at most 1,048,576 characters, then returns `ERROR`; the file is never fully loaded)*
- [x] `tools/training.py`: the generated run script was `chmod 0o755`. *(fixed: `0o600`; it is launched as `bash <path>`)*
- [x] Webhook: no request-size cap. *(fixed: `BodySizeLimitMiddleware` answers `413` past 8 MiB, declared or streamed)*
- [ ] Webhook: no rate limiting. `serve-webhook` binds localhost, so this can wait unless it is exposed.
- [ ] `lora_training.py` fetches the llama.cpp converter from a GitHub archive by commit SHA with no hash check. GitHub does not promise byte-stable archives, so a pinned hash would be brittle; candidate to accept as a known risk.

### Correctness and quality

- [ ] Choice questions with more than 26 options run out of letter labels; check what happens and fail with a clear error.
- [ ] `linter.py` (~line 139) warns "missing fallback" when one is present.
- [ ] Functions over 50 lines: `jev_calibrate_threshold` 274, `jev_evaluate_batch` 255, `jev_optimize_prompt` 204, `jev_train_lora` 203, `lint` 177, `local_choose` 145. Split the first two behind the existing tests.
- [ ] Config loaders (`router_config.py`, `model_catalog.py`) return untyped dicts.
- [ ] `run-browser-agent`: deep or below-fold rows are still missed (see the dense-click item above).

### Repo hygiene

- [ ] No dependency audit: `pip-audit` is not installed and there is no `.github/dependabot.yml`.
- [ ] Four `# type: ignore` comments are unused according to `mypy --warn-unused-ignores`: `tools/evaluate.py:209,217` and `tools/calibrate.py:40,44`. Likely relocated by the ruff pass in #20. Remove them and run that flag in CI.
- [ ] CI installs `mypy`, `pytest` and `pytest-cov` unpinned and does not install from `uv.lock`, so a new release can break the gate with no code change (ruff was pinned in #23 for the same reason).
- [ ] Branch protection: make `build-and-test` a required status check so CI is enforced by rule.
