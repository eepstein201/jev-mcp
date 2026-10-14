import contextlib
import os
import json
import math
import tempfile
from datetime import datetime
from typing import Any, TypedDict

from jev_mcp import model_catalog

ROUTER_CONFIG_PATH = os.path.expanduser("~/.jev/router_config.json")

DEFAULT_TEMPERATURE = 1.0
# The legacy global `fitted_temperature` was only ever fitted on this model, so it applies to it alone.
LEGACY_TEMPERATURE_MODEL_PREFIX = "qwen2.5-7b"
UNREPORTED_MODEL_IDS = ("", "unknown_model")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Used only when no catalog has been fetched yet (first run offline).
FALLBACK_BUCKETS = {
    "b1": "claude-haiku-5-5",
    "b2": "claude-sonnet-5-5",
    "b3": "claude-opus-5-5",
    "b4": "claude-fable-5-1",
}


class TemperatureEntry(TypedDict):
    """One smart model's calibration temperature, keyed by `model_key` in `temperatures`."""

    temperature: float
    model_size: (
        int | None
    )  # file size in bytes when known; guards against a swapped file
    updated_at: str


class RouterConfig(TypedDict, total=False):
    """The persisted shape of the router config file. total=False: keys appear
    only once written (a fresh config carries just buckets and rules)."""

    buckets: dict[str, str]
    rules: list[dict[str, str]]
    buckets_source: str  # "detected" | "user"
    detected_client: str
    models_updated_at: str
    auto_update: bool
    fitted_temperature: float  # legacy global; see get_temperature
    temperature_updated_at: str
    temperatures: dict[str, TemperatureEntry]


def load_router_config() -> RouterConfig:
    if not os.path.exists(ROUTER_CONFIG_PATH):
        return {
            "buckets": model_catalog.default_buckets(model_catalog.load_cached())
            or dict(FALLBACK_BUCKETS),
            "rules": [],
            "temperatures": {},
        }
    with open(ROUTER_CONFIG_PATH, "r") as f:
        return json.load(f)


def write_json_atomic(path: str, data: Any) -> None:
    # Write to a sibling temp file and rename over the target, so a write that
    # dies partway through never leaves a truncated file behind. Each write gets
    # its own temp file, so two writers cannot interleave into one.
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=folder, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def save_router_config(config: RouterConfig) -> None:
    write_json_atomic(ROUTER_CONFIG_PATH, config)


def seed_defaults_for_client(client_name: str) -> RouterConfig | None:
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
            seeded: RouterConfig = {
                "buckets": tiers,
                "rules": [],
                "buckets_source": "detected",
                "detected_client": client_name,
            }
        else:
            with open(ROUTER_CONFIG_PATH, "r") as f:
                config: RouterConfig = json.load(f)
            if config.get("buckets_source") != "detected":
                return None
            seeded = {**config, "buckets": tiers, "detected_client": client_name}
    except (OSError, json.JSONDecodeError):
        # An unreadable or corrupt config is never a reason to seed over it.
        return None
    save_router_config(seeded)
    return seeded


def pending_model_updates(config: RouterConfig) -> dict[str, dict[str, str]]:
    """Buckets whose model has a newer release in the same family, per the cached catalog."""
    return model_catalog.suggest_updates(
        config.get("buckets", {}), model_catalog.load_cached()
    )


def apply_model_updates(
    config: RouterConfig, updates: dict[str, dict[str, str]]
) -> RouterConfig:
    """A copy of config with each updated bucket moved to its newer model."""
    return {
        **config,
        "buckets": {
            **config.get("buckets", {}),
            **{bucket: u["newer"] for bucket, u in updates.items()},
        },
        "models_updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def model_key(model_id: str) -> str:
    """The lower-cased file name of a model id (the daemon reports a full file path)."""
    return os.path.basename(model_id.strip().replace("\\", "/")).lower()


def _is_temperature(value: Any) -> bool:
    """A usable scoring temperature: a finite, positive number (a boolean is not one)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _is_well_formed(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    size = entry.get("model_size")
    size_ok = size is None or (isinstance(size, int) and not isinstance(size, bool))
    return (
        _is_temperature(entry.get("temperature"))
        and isinstance(entry.get("updated_at"), str)
        and size_ok
    )


def valid_temperatures(config: RouterConfig) -> dict[str, TemperatureEntry]:
    """The well-formed entries of `temperatures`. The file is hand-editable, so a section
    or entry of the wrong shape counts as missing instead of raising."""
    raw = config.get("temperatures")
    if not isinstance(raw, dict):
        return {}
    return {key: entry for key, entry in raw.items() if _is_well_formed(entry)}


def temperature_entry(
    config: RouterConfig, model_id: str, model_size: int | None
) -> TemperatureEntry | None:
    """The model's entry, unless both sizes are known and differ (the file was swapped)."""
    entry = valid_temperatures(config).get(model_key(model_id))
    if entry is None:
        return None
    stored_size = entry.get("model_size")
    if stored_size is not None and model_size is not None and stored_size != model_size:
        return None
    return entry


def legacy_temperature(config: RouterConfig) -> float | None:
    """The legacy global fitted temperature when it is usable, else None.

    The file is hand-editable, so a raw value of the wrong shape (a string, a
    boolean, zero, non-finite) counts as absent rather than being echoed."""
    value = config.get("fitted_temperature")
    return (
        float(value)
        if isinstance(value, (int, float)) and _is_temperature(value)
        else None
    )


def _legacy_temperature(config: RouterConfig, model_id: str) -> float | None:
    """The legacy global, when it is usable and the model is the one it was fitted on."""
    if model_key(model_id).startswith(LEGACY_TEMPERATURE_MODEL_PREFIX):
        return legacy_temperature(config)
    return None


def get_temperature(
    config: RouterConfig, model_id: str, model_size: int | None = None
) -> float:
    """The temperature to apply to this smart model: its own entry, else the legacy
    global (qwen2.5-7b only), else DEFAULT_TEMPERATURE."""
    entry = temperature_entry(config, model_id, model_size)
    if entry is not None:
        return float(entry["temperature"])
    legacy = _legacy_temperature(config, model_id)
    return DEFAULT_TEMPERATURE if legacy is None else legacy


def is_calibrated(
    config: RouterConfig, model_id: str, model_size: int | None = None
) -> bool:
    """Whether get_temperature answers from a stored value rather than the default."""
    return (
        temperature_entry(config, model_id, model_size) is not None
        or _legacy_temperature(config, model_id) is not None
    )


def _raw_temperatures(config: RouterConfig) -> dict[str, Any]:
    raw = config.get("temperatures")
    return raw if isinstance(raw, dict) else {}


def with_temperature(
    config: RouterConfig,
    model_id: str,
    value: float,
    model_size: int | None = None,
    now: datetime | None = None,
) -> RouterConfig:
    """A copy of config with this model's temperature set. Raises ValueError for a model
    id the engine never reported, since that entry could never be read back."""
    key = model_key(model_id)
    if key in UNREPORTED_MODEL_IDS:
        raise ValueError(
            "The smart engine did not report a model, so no temperature can be stored."
        )
    stamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    entry: TemperatureEntry = {
        "temperature": value,
        "model_size": model_size,
        "updated_at": stamp,
    }
    return {
        **config,
        "temperatures": {**_raw_temperatures(config), key: entry},
        "temperature_updated_at": stamp,
    }
