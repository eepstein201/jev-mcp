"""Current model names for the router, taken from the public models.dev catalog.

Only a reduced copy is cached in ~/.jev. Nothing about the user or their prompts is sent:
refresh() is a plain GET of a public file.
"""

import json
import logging
import os
import re
import time
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

CATALOG_URL = "https://models.dev/api.json"
CACHE_PATH = os.path.expanduser("~/.jev/model_catalog.json")
CACHE_TTL_SECONDS = 24 * 60 * 60
FETCH_TIMEOUT_SECONDS = 10
USER_AGENT = "jev-mcp"
# First-party providers only: resellers repeat the same models under other ids.
PROVIDERS = ("anthropic", "openai", "google")
DEFAULT_PROVIDER = "anthropic"
BUCKETS = ("b1", "b2", "b3", "b4")
# Not something to route everyday work to: previews, moving aliases, research modes, dated snapshots.
_SKIPPED_ID = re.compile(r"preview|latest|research|\d{8}$")

Catalog = dict[str, list[dict[str, Any]]]


def slim_catalog(raw: dict[str, Any]) -> Catalog:
    """Keep the routable chat models of the first-party providers, with only the fields the router uses."""
    catalog: Catalog = {}
    for provider in PROVIDERS:
        models = (raw.get(provider) or {}).get("models") or {}
        kept = [
            {
                "id": model_id,
                "family": m["family"],
                "release_date": m["release_date"],
                "cost": m["cost"]["input"],
            }
            for model_id, m in models.items()
            if m.get("family")
            and m.get("release_date")
            and m.get("tool_call")
            and (m.get("modalities") or {}).get("output") == ["text"]
            and isinstance((m.get("cost") or {}).get("input"), (int, float))
            and not _SKIPPED_ID.search(model_id)
        ]
        if kept:
            catalog[provider] = kept
    return catalog


def save_cache(catalog: Catalog) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"fetched_at": time.time(), "providers": catalog}, f)


def _read_cache() -> dict[str, Any]:
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_cached() -> Catalog:
    """The cached catalog, or {} when there is none."""
    providers = _read_cache().get("providers")
    return providers if isinstance(providers, dict) else {}


def refresh(force: bool = False) -> bool:
    """Download the catalog unless the cache is under a day old. True when the cache is fresh afterwards."""
    fetched_at = _read_cache().get("fetched_at", 0)
    if not force and time.time() - fetched_at < CACHE_TTL_SECONDS:
        return True
    try:
        request = urllib.request.Request(
            CATALOG_URL, headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            catalog = slim_catalog(json.loads(response.read()))
        if not catalog:
            raise ValueError("the catalog had no usable models")
        save_cache(catalog)
        return True
    except Exception as e:
        logger.warning(f"Could not refresh the model catalog from {CATALOG_URL}: {e}")
        return False


def family_of(
    catalog: Catalog, model_id: str
) -> Optional[tuple[str, str, Optional[str]]]:
    """(provider, family, release date) for a model. A model no longer listed is placed by its name."""
    for provider, models in catalog.items():
        for m in models:
            if m["id"] == model_id:
                return provider, m["family"], m["release_date"]

    tokens = set(re.split(r"[-._]", model_id))
    best: Optional[tuple[str, str]] = None
    for provider, models in catalog.items():
        for family in {m["family"] for m in models}:
            parts = family.split("-")
            if set(parts) <= tokens and (
                best is None or len(parts) > len(best[1].split("-"))
            ):
                best = (provider, family)
    return (best[0], best[1], None) if best else None


def newest_in_family(
    catalog: Catalog, provider: str, family: str
) -> Optional[dict[str, Any]]:
    members = [m for m in catalog.get(provider, []) if m["family"] == family]
    # Same day: prefer the plain name over a longer variant.
    return (
        max(members, key=lambda m: (m["release_date"], -len(m["id"])))
        if members
        else None
    )


def suggest_updates(
    buckets: dict[str, str], catalog: Catalog
) -> dict[str, dict[str, str]]:
    """Buckets whose model has a newer release in the same family. Never suggests a different tier."""
    updates: dict[str, dict[str, str]] = {}
    for bucket, model_id in buckets.items():
        placed = family_of(catalog, model_id)
        if not placed:
            continue
        provider, family, released = placed
        newest = newest_in_family(catalog, provider, family)
        if (
            newest
            and newest["id"] != model_id
            and (released is None or newest["release_date"] > released)
        ):
            updates[bucket] = {
                "current": model_id,
                "newer": newest["id"],
                "released": newest["release_date"],
            }
    return updates


def default_buckets(catalog: Catalog) -> Optional[dict[str, str]]:
    """b1..b4 for a new install: the newest model of each tier, cheapest first. None if there are not four tiers."""
    families = {m["family"] for m in catalog.get(DEFAULT_PROVIDER, [])}
    newest = [
        m
        for m in (newest_in_family(catalog, DEFAULT_PROVIDER, f) for f in families)
        if m
    ]
    if len(newest) < len(BUCKETS):
        return None
    current = sorted(newest, key=lambda m: m["release_date"], reverse=True)[
        : len(BUCKETS)
    ]
    return {
        bucket: m["id"]
        for bucket, m in zip(BUCKETS, sorted(current, key=lambda m: m["cost"]))
    }
