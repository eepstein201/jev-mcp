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

## P1: correctness and safety of training and daemon control

- [ ] **`train` fuse step cannot work for the default model.** (from source)
  It runs `mlx_lm.fuse --export-gguf`, which `mlx_lm` 0.31.3 limits to unquantized `llama`/`mistral`/`mixtral`. The default base is a quantized `qwen2` model.
  - Fix: `mlx_lm.fuse --dequantize --save-path <dir>`, then llama.cpp `convert_hf_to_gguf.py`, then `llama-quantize`. The converter is not shipped by Homebrew's `llama.cpp`; pin it to the installed build's commit.

- [ ] **`train` restart logic targets the wrong process.** (from source)
  It kills the first `llama-server` PID it finds and relaunches that command with `nohup`. With two daemons this may be Kev rather than the model being replaced, and the launchd agents use `KeepAlive`, so launchd also respawns it.
  - Fix: restart the specific launchd agent (`launchctl kickstart -k gui/$UID/<label>`); never `nohup` a second copy.

- [ ] **`train` has a backup but no restore.**
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

- [ ] **Legacy launchd agent is still loaded and failing.** *(script fixed: `hybrid` now deletes the legacy plist. Still to do on this machine: its plist runs `kev.serve --port 8080`, competes with the fast engine, and has crash-looped 855 times; boot it out and delete `~/Library/LaunchAgents/com.jev.mlx_server.plist`.)*
  `com.jev.mlx_server` shows a changing PID with exit status 3 alongside the `_fast` and `_smart` agents.
  - Fix: boot it out and remove its plist during `hybrid`/`install`.

## P1: scoring behaviour

- [ ] **XML sandbox prompt is never used for Qwen under `llama-server`.** (from source)
  **Decision (2026-10-10): turn it on.** Select the prompt from the configured engine, not `pgrep`. This changes every Qwen score, so re-run `make eval` and the calibration table and update the README numbers in the same change.
  `daemon_provider.py:204-208` decides `is_small_model` from `pgrep -fl mlx_lm`; with no `mlx_lm` process the call fails and it defaults to `True`, selecting the simple prompt. The README says the 7B profile is sandboxed. The prompt also silently switches whenever any `mlx_lm` process (e.g. training) is running.
  - Fix: decide from the configured engine/model, not from `pgrep`.

- [ ] **Qwen scores are saturated.** (verified)
  **Decision (2026-10-10): do both, in the form that is best for now.**
  (a) Missing tokens: treat as censored. When `true`/`false` is absent from the top-11 log-probs, use the lowest log-prob in that list as an upper bound instead of the `-9999.0` sentinel, so scores are small but never a false exact 0.0.
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

- [ ] **`optimize-prompt` selects the most confident phrasing, not the most accurate.**
  It scores each variation on one sample state and keeps the one with the highest `max(p, 1-p)`; it never sees labels. In the `compact` experiment this raised relevant and irrelevant scores together.
  - Fix options: accept a small labelled set and pick by accuracy or AUC; or, when unlabelled, pick by score spread across several states.

- [x] **`calibrate` is single-question and in-sample.** *(fixed: rows may carry their own `"question"`; Platt scaling is 5-fold cross-validated when each class has at least 5 rows, and the report states which was used. Persisting the temperature is tracked under the saturation item.)*
  One `question` per call, and the Platt fit is evaluated on the rows it was fitted on.
  - Fix: accept per-row questions; report cross-validated numbers; optionally persist the fitted temperature (the tool currently never writes it).

## P2: smaller issues

- [x] `email_triage/mcp_tool.py`: `except Exception: pass` swallows compaction errors; log a warning instead. The `2000` and `0.6` thresholds are magic numbers. *(fixed)*
- [ ] `email_triage/auth.py`: `jwt_decode` is a stub that always raises, so only the static `JEV_MCP_API_KEY` works. No launcher ships for the webhook app.
- [ ] `run-browser-agent`: only checks that `./jev-ultrafast` exists (CWD-relative) and returns a dispatch message. *(unused imports removed; the stub behaviour remains)*
- [ ] Two unrelated `NoulQuestion`/`ChoiceQuestion` classes (`provider.py` vs `email_triage/core.py`) with the same names.
- [ ] `router_config.py` defaults reference old model names (`claude-3-5-*`, `claude-fable`).
  **Decision (2026-10-10): discover defaults per client, with a fallback table.** Ideally the MCP layer detects the connecting CLI on each load (Claude Code, Antigravity, OpenCode, ...) and seeds buckets `b1`-`b4` with that client's current default models; the user can overwrite them, and a built-in default is used when detection fails. Needs design first: what each client exposes (MCP `clientInfo` name/version, the client's own config files), and how user overrides are kept separate from detected defaults.
- [ ] `server.py` is 1748 lines; split by tool family.
- [ ] `requirements.txt` and `pyproject.toml` list different dependency sets (e.g. `fastapi`, `tree-sitter`, Google clients only in `pyproject`; `mlx`, `kev` only in `requirements.txt`).
  **Decision (2026-10-10): `pyproject.toml` is the source of truth.** Move the engine and dev packages into optional dependency groups there (e.g. `engines`: `mlx`, `mlx-lm`, `kev`; `dev`: `ruff`, `mypy`, `pytest`, `pytest-cov`), switch CI to `pip install -e ".[dev,engines]"`, relock, and retire `requirements.txt`.
- [x] CI: `actions/checkout@v4` and `actions/setup-python@v5` trigger a Node 20 deprecation warning. *(fixed: both on v7; the warning is gone from the main run)*
- [x] Some tests reached a real daemon when one was running (two `compact` tests, and two linter tests whose auto-fixer made a real generation call, taking 9-13 s with model-dependent results). *(fixed: an autouse fixture in `tests/conftest.py` makes unmocked daemon calls fail as in CI; the suite runs in about 6 s instead of 35 s)*
- [ ] Golden dataset: 80 of 88 rows were written in one pass by an LLM and frontier models score 100%; add real, hard cases.
