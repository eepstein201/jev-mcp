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

- [ ] **`train` non-fuse path downloads and runs a script from `master`.** (from source)
  It `curl`s `convert-lora-to-ggml.py` from llama.cpp `master` and executes it.
  - Fix: remove, or pin to a commit and verify a checksum.

- [ ] **`train` uses fixed `--iters 500`.**
  No way to scale iterations to dataset size; overfits small sets.
  - Fix: expose iterations (and batch size / layers) as arguments.

- [ ] **`manage_daemon stop` kills every `llama-server`.**
  `jev_mac_manager.sh:312-313` runs `pkill -9 -f llama-server` after booting out one plist, so stopping one engine stops both.
  - Fix: stop only the targeted agent.

- [ ] **Legacy launchd agent is still loaded and failing.**
  `com.jev.mlx_server` shows a changing PID with exit status 3 alongside the `_fast` and `_smart` agents.
  - Fix: boot it out and remove its plist during `hybrid`/`install`.

## P1: scoring behaviour

- [ ] **XML sandbox prompt is never used for Qwen under `llama-server`.** (from source)
  `daemon_provider.py:204-208` decides `is_small_model` from `pgrep -fl mlx_lm`; with no `mlx_lm` process the call fails and it defaults to `True`, selecting the simple prompt. The README says the 7B profile is sandboxed. The prompt also silently switches whenever any `mlx_lm` process (e.g. training) is running.
  - Fix: decide from the configured engine/model, not from `pgrep`.

- [ ] **Qwen scores are saturated.** (verified)
  80% of scores on the 88-row set are within 1e-6 of 0 or 1; 26 are exactly 0.0 because the `true` token fell outside `top_logprobs=11` and got the `-9999.0` sentinel. Thresholds between 0.50 and 0.90 behave identically.
  - Fix options: apply the fitted temperature by default after calibration; treat missing tokens as censored rather than zero; report saturation in tool output.

- [ ] **`JEV_DAEMON_PORT` default collides with the fast port.**
  Both default to `8080`. Fix together with the provider item above (drop the variable or default it to the smart port).

- [ ] **`compact` chunks JSON states blindly.** (verified)
  A JSON state has no blank lines, so it is treated as one blob and cut every 2000 characters (`server.py:1459-1474`), splitting mid-sentence.
  - Fix: chunk by top-level key / list item for dict states.

- [ ] **`calibrate` is single-question and in-sample.**
  One `question` per call, and the Platt fit is evaluated on the rows it was fitted on.
  - Fix: accept per-row questions; report cross-validated numbers; optionally persist the fitted temperature (the tool currently never writes it).

## P2: smaller issues

- [ ] `email_triage/mcp_tool.py`: `except Exception: pass` swallows compaction errors; log a warning instead. The `2000` and `0.6` thresholds are magic numbers.
- [ ] `email_triage/auth.py`: `jwt_decode` is a stub that always raises, so only the static `JEV_MCP_API_KEY` works. No launcher ships for the webhook app.
- [ ] `run-browser-agent`: only checks that `./jev-ultrafast` exists (CWD-relative) and returns a dispatch message; unused `subprocess`/`tempfile` imports.
- [ ] Two unrelated `NoulQuestion`/`ChoiceQuestion` classes (`provider.py` vs `email_triage/core.py`) with the same names.
- [ ] `router_config.py` defaults reference old model names (`claude-3-5-*`, `claude-fable`).
- [ ] `server.py` is 1748 lines; split by tool family.
- [ ] `requirements.txt` and `pyproject.toml` list different dependency sets (e.g. `fastapi`, `tree-sitter`, Google clients only in `pyproject`; `mlx`, `kev` only in `requirements.txt`).
- [ ] CI: `actions/checkout@v4` and `actions/setup-python@v5` trigger a Node 20 deprecation warning.
- [ ] Golden dataset: 80 of 88 rows were written in one pass by an LLM and frontier models score 100%; add real, hard cases.
