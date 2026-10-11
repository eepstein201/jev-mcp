import contextlib
import os
import json
import tempfile
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

def write_json_atomic(path: str, data: Any) -> None:
    # Write to a sibling temp file and rename over the target, so a write that
    # dies partway through never leaves a truncated file behind. Each write gets
    # its own temp file, so two writers cannot interleave into one.
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=folder, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def save_router_config(config):
    write_json_atomic(ROUTER_CONFIG_PATH, config)

def seed_defaults_for_client(client_name: str) -> dict[str, Any] | None:
    """Seed router buckets from the connecting client when the user hasn't chosen them.

    Eligible only when no config file exists yet, or when the buckets were
    previously seeded by detection (``buckets_source == "detected"``); a config
    without the field counts as user-chosen and is never touched. Returns the
    written config, or None when nothing was seeded.
    """
    from jev_mcp.client_defaults import default_buckets_for_client

    tiers = default_buckets_for_client(client_name)
    if tiers is None:
        return None
    try:
        if not os.path.exists(ROUTER_CONFIG_PATH):
            seeded = {
                "buckets": tiers,
                "rules": [],
                "buckets_source": "detected",
                "detected_client": client_name,
            }
        else:
            with open(ROUTER_CONFIG_PATH, "r") as f:
                config = json.load(f)
            if config.get("buckets_source") != "detected":
                return None
            seeded = {**config, "buckets": tiers, "detected_client": client_name}
    except (OSError, json.JSONDecodeError):
        # An unreadable or corrupt config is never a reason to seed over it.
        return None
    save_router_config(seeded)
    return seeded

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
