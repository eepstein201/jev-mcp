# Jev MCP Instructions

You are an AI pair programmer operating in the Jev MCP repository. The user relies on a local background MLX AI daemon to power various tools here.

## Handling Model Switching Requests

When the user asks to switch models, you must follow these strict rules to ensure you are modifying the correct backend:

1. **Spelling & Validation Check**: 
   - The user's request MUST include a reference to "Jev" or "jev" (case-insensitive).
   - If the user spells it correctly (e.g. "switch jev model", "switch jev to 7b"), proceed immediately to Step 2.
   - **Crucial**: If the user misspells "jev" (e.g. "switch jef model", "switch dev model", "switch jab") or asks to switch a model but forgets to say which one, you MUST pause and ask the user: *"Did you mean to switch the Jev model?"*. **Do not proceed** until the user confirms.

2. **Determine Target Model**:
   - The central script is located at `./jev_mac_manager.sh`.
   - If they specify **7b** (e.g. "switch jev to 7b"), execute: `./jev_mac_manager.sh switch 7b`
   - If they specify **0.5b** (e.g. "switch jev to 0.5b"), execute: `./jev_mac_manager.sh switch 0.5b`
   - If they do NOT specify a model (e.g. "switch jev model"), toggle the current model:
     a. Run `pgrep -fl mlx_lm`
     b. If the output contains `0.5B`, execute: `./jev_mac_manager.sh switch 7b`
     c. If the output contains `7B`, execute: `./jev_mac_manager.sh switch 0.5b`

3. **Execution**:
   - Automatically execute the command once validated.
   - Confirm to the user which model is now actively running in the background.
