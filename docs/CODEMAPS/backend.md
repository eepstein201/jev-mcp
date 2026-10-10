<!-- Generated: 2026-10-10 | Files scanned: server.py 1748 lines + 18 modules | Token estimate: ~900 -->
# Backend (MCP tools)

## Tools (server.py) → impl
evaluate           → jev_evaluate_batch L155   (state + QuestionType[] → RoutingProvider)
optimize-prompt    → jev_optimize_prompt L402
calibrate          → jev_calibrate_threshold L522 (golden set, Platt)
explain-decision   → jev_explain_decision L713
generate-data      → jev_generate_synthetic_dataset L779
manage-router      → jev_manage_router_config L930 (~/.jev/router_config.json)
model-router       → jev_determine_best_model L969
handoff            → jev_agent_handoff L1099
train              → jev_train_lora L1132 (engine mlx|llama.cpp, fuse, target_gguf_path)
temperature        → jev_manage_temperature L1297 (view|set|reset)
read-file          → jev_read_file L1332 (path-secured, chunked)
compact            → jev_compact_context L1443
scan-repo          → jev_scan_repo L1529
run-browser-agent  → jev_run_browser_agent L1685 (checks ./jev-ultrafast; returns dispatch msg only)
triage_email_content / configure_triage_labels / setup_gas_workflow → email_triage/* L1714-1744

## Prompts (server.py L1032-1290, 1646-1657)
calibrate compact evaluate explain-decision generate-data handoff model-router manage-router
optimize-prompt temperature train scan-repo read-file

## Helpers
_clean_json L37 · _safe_truncate L66 (24k chars) · _chat_completion(_json) L74/109
call_fast_autofixer L120 · get_temp_notice L910 · main L1660

## Email triage (email_triage/)
POST /api/v1/triage/email → api.triage_email (Depends get_auth ← auth.verify_auth_token)
  → mcp_tool.triage_email_content → evaluate_email_with_mlx → core.build_triage_questions
  → core.decide_routing → slack.build_escalation_message / build_hitl_message
gas_setup.setup_gas_workflow → examples/gas_triage.js (Google Apps Script)

## Linter thresholds (linter.py)
MAX_STATE_TOKENS=3500 · MAX_OPTIONS=10 · MAX_OPTION_LENGTH_RATIO=3.5

## Observed smells (for review, not fixed here)
- server.py is 1748 lines (>800 guideline)
- ultrafast_adapter builds ChoiceQuestion(key=, prompt=, options=) but provider.ChoiceQuestion uses question/instruction/choices
- jev_run_browser_agent: unused subprocess/tempfile imports; "jev-ultrafast" path is CWD-relative; only returns a dispatch message
- router_config defaults reference stale model names (claude-3-5-*, claude-fable)
