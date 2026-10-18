<!-- Generated: 2026-10-11, refreshed 2026-10-14 | server.py split into app.py + tools/ (see git history) | Token estimate: ~1100 | As of: 43fb0dd -->
# Backend (MCP tools)

## Layout
server.py (~200 lines) — entry point: `python -m jev_mcp.server` → main() (`setup-gas` / `serve-webhook` / `doctor` / stdio; only the stdio path runs `start_model_sync`), 13 `@mcp.prompt` definitions, re-exports of every tool name.
doctor.py (~420 lines) — `jev-mcp doctor [--json]`; prints only from its own main(); reads engine `/health` (2 s timeout), the active profile's expected vs `/props`-served model per engine, router config, `security` roots, auth variable NAMES, triage thresholds, versions.
engine_profiles.py (~700 lines) — profile table + resolver CLI (`python -m jev_mcp.engine_profiles resolve|plist|activate`; contract: docs/superpowers/specs/2026-10-14-engine-profiles-cli-contract.md §5); reads tracked engine_profiles.toml, validates every field against allow-lists; `active_profile()` never raises (None + WARNING). Stdlib only.
app.py (~360 lines) — shared state: `mcp` (MCPServer), `provider` (RoutingProvider), `linter`, `logger`, `_clean_json`/`_safe_truncate`/`_chat_completion(_json)`, `call_fast_autofixer`, `get_temp_notice`/`get_model_notice`, `start_model_sync`/`_sync_models`, `active_engine_profile`/`engine_identity` ({profile, fast, smart} spread into evaluation and routing results). Tool modules reference `app.X` AT CALL TIME (never import `jev_mcp.server`). MCPServer carries `client_defaults.SeedingMiddleware` (handshake-time per-client bucket seeding; config provenance `buckets_source`/`detected_client`).

## Tools (tools/) → impl
18 tools. Tool name = function name, except the three triage tools (function in brackets). Renamed 2026-10-12, no aliases; old → new table in README "Renamed tools". Every tool carries a title and behaviour hints (annotations), pinned by tests/test_tool_surface.py.
jev_evaluate_batch             → tools/evaluate.py (state + QuestionType[] → RoutingProvider)
jev_explain_decision           → tools/evaluate.py
jev_generate_synthetic_dataset → tools/evaluate.py
jev_optimize_prompt            → tools/prompts.py (dataset → prompt_selection.py picks by accuracy; single state → by confidence)
jev_calibrate_threshold        → tools/calibrate.py (golden set, Platt; 5-fold StratifiedKFold when each class ≥ PLATT_CV_FOLDS=5)
jev_manage_temperature         → tools/calibrate.py (view|set|reset)
jev_manage_router_config       → tools/router.py (~/.jev/router_config.json; apply_updates / set_auto_update via model_catalog.py; main() starts start_model_sync)
jev_determine_best_model       → tools/router.py
jev_agent_handoff              → tools/router.py
jev_train_lora                 → tools/training.py (engine mlx|llama.cpp, fuse, target_gguf_path, action train|restore; steps in lora_training.py; ensure_converter syncs ~/.jev/llama.cpp-src to the llama-server commit)
jev_read_file                  → tools/files.py (path-secured, chunked)
jev_compact_context            → tools/files.py (confidence_threshold default 0.2)
jev_scan_repo                  → tools/files.py
jev_list_files                 → tools/files.py → listing.py (read-only; is_safe_path, security.is_sensitive_entry, no symlink follow, caps depth 5 / entries 1000; docs/perf/list-files.md)
jev_run_browser_agent          → tools/browser.py → browser_agent.run_browser_agent (jev-ultrafast Agent loop; choose → ultrafast_adapter.local_choose, field text → smart daemon; http/https only, max_steps)
jev_triage_email [tool_triage_email_content] / jev_configure_triage_labels [tool_configure_triage_labels] / jev_setup_gas_workflow [tool_setup_gas_workflow] → tools/triage.py → email_triage/*

## Prompts (server.py)
Prompt (slash-command) names kept their old spelling; each prompt text names the renamed tool.
calibrate compact evaluate explain-decision generate-data handoff model-router manage-router
optimize-prompt temperature train scan-repo read-file

## Email triage (email_triage/)
`jev-mcp serve-webhook` (server.main → uvicorn, 127.0.0.1:8090 by default; refuses to start with no credential)
POST /api/v1/triage/email → api.triage_email (Depends get_auth ← auth.verify_auth_token: static key, else auth.jwt_decode via JWKS or HS256 secret)
  api.enforce_rate_limit after auth (route dep; api.limiter ← rate_limit.py keyed on request.client.host; JEV_MCP_RATE_LIMIT_PER_MINUTE per 60 s → 429 + Retry-After; counts authenticated requests only)
  → mcp_tool.triage_email_content → evaluate_email_with_mlx → core.build_triage_questions
      body > 2000 chars (COMPACTION_MIN_BODY_CHARS) → server.jev_compact_context(goal=labels, conf COMPACTION_CONFIDENCE=0.3) → body = compressed_context
      (lazy import of tools.files inside fn; any failure/empty result silently keeps original body)
  → core.decide_routing → slack.build_escalation_message / build_hitl_message
api.triage_email: degraded result + X-Triage-Attempt < MAX_DEGRADED_ATTEMPTS (3) → 503 + Retry-After, rate-limit count refunded (limiter.refund), no Slack; else 200 with an "Unconfirmed" Slack notice; mcp_tool.fail_toward_attention turns a degraded archive_and_label into hitl_slack_block
gas_setup.setup_gas_workflow → examples/gas_triage.js (Google Apps Script)

## Linter thresholds (linter.py)
MAX_STATE_TOKENS=3500 · MIN_OPTIONS=2 · MAX_OPTIONS=10 · MAX_OPTION_LENGTH_RATIO=3.5

## Observed smells (for review, not fixed here)
- Largest modules: daemon_provider.py (~720), engine_profiles.py (~700), tools/evaluate.py (~580), tools/calibrate.py (~530) — all under the 800-line cap; tools/files.py `jev_read_file`/`jev_scan_repo` withhold irrelevant content without naming it (see docs/TODO.md)
