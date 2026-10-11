<!-- Generated: 2026-10-10 | Files scanned: 6 | Token estimate: ~350 -->
# Data & State (no database)

| Path | Owner | Contents |
|------|-------|----------|
| ~/.jev/router_config.json | router_config.py | buckets b1-b4 → model names, rules[], fitted temperature, auto_update, buckets_source ("detected"/"user"), detected_client |
| ~/.jev/model_catalog.json | model_catalog.py | reduced models.dev catalog (id, family, release date, price), refreshed daily at start-up |
| ~/.jev/jev.log | server logging | runtime log |
| ~/.jev/adapters/{run_id} | jev_train_lora | LoRA adapters |
| ~/.jev/training | jev_train_lora | training data/artifacts |
| ~/.jev/mlx_server_{fast,smart}.log | jev_mac_manager.sh (launchd) | daemon logs |
| ~/.jev-mcp/triage_configs.json | email_triage/mcp_tool.py | {context_id: [labels]} |
| ~/.jev-mcp/client_secrets.json | email_triage/gas_setup.py | Google OAuth client (user-supplied, read only) |
| tests/golden_dataset.json | tests/run_evals.py | calibration/eval set |

## Models (pydantic)
provider.py: NoulQuestion{prompt,key} · ChoiceQuestion{prompt,options,key} · ScoreQuestion{prompt,labels,key} · QuestionType=Union
email_triage/core.py builds jev_mcp.provider's question classes (no separate duplicates)
CategoryConfig.categories = Follow Up|Pending|Receipts|Newsletter|Notifications|Review
JevMailConfig{action_threshold=0.8, important_threshold=0.8}

## Lifetime
`make clean` (uninstall) runs `rm -rf ~/.jev`: router config, adapters, training artifacts and logs are all deleted. `~/.jev-mcp/` is left in place.
