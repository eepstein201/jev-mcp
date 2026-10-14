<!-- Generated: 2026-10-10, refreshed 2026-10-14 | Files scanned: 9 | Token estimate: ~500 | As of: 43fb0dd -->
# Data & State (no database)

| Path | Owner | Contents |
|------|-------|----------|
| engine_profiles.toml (tracked) | hand-edited | engine profiles: 8 models (tier, repo, pinned revision + sha256, parallel, ctx_per_slot, reasoning, style) and 7 profiles (fast + smart pair; gate default 0.85; eval_only flags) |
| ~/.jev/active_profile | engine_profiles `activate` (via `make start`) | one line: the active profile name; missing file = `default` |
| models/<repo name>/<file> (repo-local) | jev_mac_manager.sh | GGUF weights downloaded per engine_profiles.toml (SHA-256 verified, staged as `.part`) |
| ~/.jev/router_config.json | router_config.py | buckets b1-b4 → model names, rules[], temperatures{} (per smart model; legacy global fitted_temperature applies to qwen2.5-7b only), auto_update, buckets_source ("detected"/"user"), detected_client |
| ~/.jev/model_catalog.json | model_catalog.py | reduced models.dev catalog (id, family, release date, price), refreshed daily at start-up |
| ~/.jev/jev.log | server logging | runtime log |
| ~/.jev/adapters/{run_id} | jev_train_lora | LoRA adapters |
| ~/.jev/training | jev_train_lora | training data/artifacts |
| ~/.jev/llama.cpp-src | lora_training.py | llama.cpp converter fetched for the running `llama-server` commit (unless JEV_LLAMA_CPP_SRC names your own) |
| ~/.jev/datasets/hf-email-classifier.jsonl | tests/evals/hf_email_import.py | 120-email triage eval set built from the gated HF dataset (outside the repo) |
| ~/.jev/usage-eval-venv | `make eval-usage` | separate venv for the tool-usage eval runner (`anthropic` + `mcp` 1.x) |
| /tmp/jev-usage-eval-home | tests/evals/usage/seed_home.py | seeded HOME + fixture dir for `make eval-usage` |
| ~/.jev/mlx_server_{fast,smart}.log | jev_mac_manager.sh (launchd) | daemon logs |
| ~/.jev-mcp/triage_configs.json | email_triage/mcp_tool.py | {context_id: [labels]} |
| ~/.jev-mcp/client_secrets.json | email_triage/gas_setup.py | Google OAuth client (user-supplied, read only) |
| tests/golden_dataset.json | tests/run_evals.py | calibration/eval set |
| tests/evals/triage_synthetic.jsonl | tests/evals/triage_eval.py | 30 labelled synthetic emails with expected routing |

## Models (pydantic)
provider.py: NoulQuestion{prompt,key} · ChoiceQuestion{prompt,options,key} · ScoreQuestion{prompt,labels,key} · QuestionType=Union
email_triage/core.py builds jev_mcp.provider's question classes (no separate duplicates)
CategoryConfig.categories = Follow Up|Pending|Receipts|Newsletter|Notifications|Review
JevMailConfig{action_threshold=0.8, important_threshold=0.8}

## Lifetime
`make clean` (uninstall) runs `rm -rf ~/.jev`: router config, adapters, training artifacts, the llama.cpp converter, eval datasets and logs are all deleted. `~/.jev-mcp/` is left in place.
