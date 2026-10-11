<!-- Generated: 2026-10-10 | Files scanned: server.py 1772 lines + 18 modules | Token estimate: ~900 | As of: 8479025 -->
# Backend (MCP tools)

## Tools (server.py) → impl
evaluate           → jev_evaluate_batch L159   (state + QuestionType[] → RoutingProvider)
optimize-prompt    → jev_optimize_prompt L406 (dataset → prompt_selection.py picks by accuracy; single state → by confidence)
calibrate          → jev_calibrate_threshold L526 (golden set, Platt; 5-fold StratifiedKFold when each class ≥ PLATT_CV_FOLDS=5)
explain-decision   → jev_explain_decision L735
generate-data      → jev_generate_synthetic_dataset L801
manage-router      → jev_manage_router_config L952 (~/.jev/router_config.json)
model-router       → jev_determine_best_model L991
handoff            → jev_agent_handoff L1121
train              → jev_train_lora L1154 (engine mlx|llama.cpp, fuse, target_gguf_path, action train|restore; steps in lora_training.py; ensure_converter syncs ~/.jev/llama.cpp-src to the llama-server commit)
temperature        → jev_manage_temperature L1315 (view|set|reset)
read-file          → jev_read_file L1350 (path-secured, chunked)
compact            → jev_compact_context L1461 (confidence_threshold default 0.3)
scan-repo          → jev_scan_repo L1555
run-browser-agent  → jev_run_browser_agent → browser_agent.run_browser_agent (jev-ultrafast Agent loop; choose → ultrafast_adapter.local_choose, field text → smart daemon; http/https only, max_steps)
triage_email_content / configure_triage_labels / setup_gas_workflow → email_triage/* L1738-1772

## Prompts (server.py L1054-1114, 1297-1308, 1672-1683)
calibrate compact evaluate explain-decision generate-data handoff model-router manage-router
optimize-prompt temperature train scan-repo read-file

## Helpers
_clean_json L37 · _safe_truncate L66 (24k chars) · _chat_completion(_json) L74/109
call_fast_autofixer L124 · get_temp_notice L932 · main L1686

## Email triage (email_triage/)
`jev-mcp serve-webhook` (server.main → uvicorn, 127.0.0.1:8090 by default; refuses to start with no credential)
POST /api/v1/triage/email → api.triage_email (Depends get_auth ← auth.verify_auth_token: static key, else auth.jwt_decode via JWKS or HS256 secret)
  → mcp_tool.triage_email_content → evaluate_email_with_mlx → core.build_triage_questions
      body > 2000 chars (COMPACTION_MIN_BODY_CHARS) → server.jev_compact_context(goal=labels, conf COMPACTION_CONFIDENCE=0.3) → body = compressed_context
      (lazy import of server inside fn; any failure/empty result silently keeps original body)
  → core.decide_routing → slack.build_escalation_message / build_hitl_message
gas_setup.setup_gas_workflow → examples/gas_triage.js (Google Apps Script)

## Linter thresholds (linter.py)
MAX_STATE_TOKENS=3500 · MIN_OPTIONS=2 · MAX_OPTIONS=10 · MAX_OPTION_LENGTH_RATIO=3.5

## Observed smells (for review, not fixed here)
- server.py is 1772 lines (>800 guideline)
- ROUTER_CONFIG_PATH is defined twice (router_config.py L4, server.py L927)
- router_config defaults reference stale model names (claude-3-5-*, claude-3-opus, claude-fable)
