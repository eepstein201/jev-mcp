import os
import json
from datetime import datetime
from typing import Any

from jev_mcp import model_catalog

ROUTER_CONFIG_PATH = os.path.expanduser("~/.jev/router_config.json")

# Used only when no catalog has been fetched yet (first run offline).
FALLBACK_BUCKETS = {
    "b1": "claude-haiku-5-5",
    "b2": "claude-sonnet-5-5",
    "b3": "claude-opus-5-5",
    "b4": "claude-fable-5-1",
}

def load_router_config():
    if not os.path.exists(ROUTER_CONFIG_PATH):
        return {
            "buckets": model_catalog.default_buckets(model_catalog.load_cached()) or dict(FALLBACK_BUCKETS),
            "rules": []
        }
    with open(ROUTER_CONFIG_PATH, "r") as f:
        return json.load(f)

def save_router_config(config):
    os.makedirs(os.path.dirname(ROUTER_CONFIG_PATH), exist_ok=True)
    with open(ROUTER_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def pending_model_updates(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Buckets whose model has a newer release in the same family, per the cached catalog."""
    return model_catalog.suggest_updates(config.get("buckets", {}), model_catalog.load_cached())

def apply_model_updates(config: dict[str, Any], updates: dict[str, dict[str, str]]) -> dict[str, Any]:
    """A copy of config with each updated bucket moved to its newer model."""
    return {
        **config,
        "buckets": {**config.get("buckets", {}), **{bucket: u["newer"] for bucket, u in updates.items()}},
        "models_updated_at": datetime.now().isoformat(timespec="seconds"),
    }
