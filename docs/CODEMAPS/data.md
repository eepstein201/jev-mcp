<!-- Generated: 2026-10-10 | Files scanned: 6 | Token estimate: ~350 -->
# Data & State (no database)

| Path | Owner | Contents |
|------|-------|----------|
| ~/.jev/router_config.json | router_config.py | buckets b1-b4 → model names, rules[], fitted temperature |
| ~/.jev/jev.log | server logging | runtime log |
| ~/.jev/adapters/{run_id} | jev_train_lora | LoRA adapters |
| ~/.jev/training | jev_train_lora | training data/artifacts |
| ~/.jev-mcp/triage_configs.json | email_triage/mcp_tool.py | {context_id: [labels]} |
| tests/golden_dataset.json | tests/run_evals.py | calibration/eval set |

## Models (pydantic, provider.py / email_triage/core.py)
NoulQuestion{question,instruction,type="noul"} · ChoiceQuestion{...,choices} · ScoreQuestion
CategoryConfig.categories = Follow Up|Pending|Receipts|Newsletter|Notifications|Review
JevMailConfig{action_threshold=0.8, important_threshold=0.8}
