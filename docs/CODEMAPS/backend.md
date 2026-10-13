<!-- Generated: 2026-10-11 | server.py split into app.py + tools/ (see git history) | Token estimate: ~900 -->
# Backend (MCP tools)

## Layout
server.py (~175 lines) — entry point: `python -m jev_mcp.server` → main() (`setup-gas` / `serve-webhook` / stdio), `@mcp.prompt` definitions, re-exports.
app.py — shared state: `mcp` (MCPServer), `provider` (RoutingProvider), `linter`, `logger`, `_clean_json`/`_safe_truncate`/`_chat_completion(_json)`, `call_fast_autofixer`, `get_temp_notice`/`get_model_notice`, `start_model_sync`/`_sync_models`. Tool modules reference `app.X` AT CALL TIME (never import `jev_mcp.server`). MCPServer carries `client_defaults.SeedingMiddleware` (handshake-time per-client bucket seeding; config provenance `buckets_source`/`detected_client`).

## Tools (tools/) → impl
Tool name = function name, except the three triage tools (function in brackets). Renamed 2026-10-12, no aliases; old → new table in README "Renamed tools".
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
jev_compact_context            → tools/files.py (confidence_threshold default 0.3)
jev_scan_repo                  → tools/files.py
jev_run_browser_agent          → tools/browser.py → browser_agent.run_browser_agent (jev-ultrafast Agent loop; choose → ultrafast_adapter.local_choose, field text → smart daemon; http/https only, max_steps)
jev_triage_email [tool_triage_email_content] / jev_configure_triage_labels [tool_configure_triage_labels] / jev_setup_gas_workflow [tool_setup_gas_workflow] → tools/triage.py → email_triage/*

## Prompts (server.py)
Prompt (slash-command) names kept their old spelling; each prompt text names the renamed tool.
calibrate compact evaluate explain-decision generate-data handoff model-router manage-router
optimize-prompt temperature train scan-repo read-file

## Email triage (email_triage/)
`jev-mcp serve-webhook` (server.main → uvicorn, 127.0.0.1:8090 by default; refuses to start with no credential)
POST /api/v1/triage/email → api.triage_email (Depends get_auth ← auth.verify_auth_token: static key, else auth.jwt_decode via JWKS or HS256 secret)
  → mcp_tool.triage_email_content → evaluate_email_with_mlx → core.build_triage_questions
      body > 2000 chars (COMPACTION_MIN_BODY_CHARS) → server.jev_compact_context(goal=labels, conf COMPACTION_CONFIDENCE=0.3) → body = compressed_context
      (lazy import of tools.files inside fn; any failure/empty result silently keeps original body)
  → core.decide_routing → slack.build_escalation_message / build_hitl_message
gas_setup.setup_gas_workflow → examples/gas_triage.js (Google Apps Script)

## Linter thresholds (linter.py)
MAX_STATE_TOKENS=3500 · MIN_OPTIONS=2 · MAX_OPTIONS=10 · MAX_OPTION_LENGTH_RATIO=3.5

## Observed smells (for review, not fixed here)
- (resolved) server.py was 1772 lines; now app.py + tools/ families + a ~175-line entry point
