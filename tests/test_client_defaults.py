"""Tests for per-client router default seeding (client_defaults + router_config)."""
import os
from pathlib import Path

import pytest

from jev_mcp import app, client_defaults, model_catalog, router_config, server


def _entry(model_id, family, released, cost):
    return {"id": model_id, "family": family, "release_date": released, "cost": cost}


CATALOG = {
    "anthropic": [
        _entry("claude-haiku-5-5", "claude-haiku", "2026-10-07", 0.1),
        _entry("claude-sonnet-5", "claude-sonnet", "2026-06-29", 2),
        _entry("claude-sonnet-5-5", "claude-sonnet", "2026-09-28", 2),
        _entry("claude-opus-5-5", "claude-opus", "2026-09-22", 4),
        _entry("claude-fable-5-1", "claude-fable", "2026-09-01", 10),
    ]
}

STALE = {"b1": "claude-3-5-haiku", "b2": "claude-sonnet-5", "b3": "claude-opus-5-5", "b4": "my-local-llama"}


@pytest.fixture
def cached_catalog():
    model_catalog.save_cache(CATALOG)


def _write_config(**extra):
    router_config.save_router_config({
        "buckets": dict(STALE),
        "rules": [{"condition": "c", "target": "b3"}],
        "buckets_source": "detected",
        "detected_client": "Claude Code",
        **extra,
    })


# ---------------------------------------------------------
# Task 2: user consent stamps buckets_source "user"
# ---------------------------------------------------------

def test_set_bucket_stamps_user_provenance():
    _write_config()
    server.jev_manage_router_config(action="set_bucket", bucket_id="b1", model_name="m")
    assert router_config.load_router_config()["buckets_source"] == "user"


def test_apply_updates_stamps_user_provenance(cached_catalog):
    _write_config()
    server.jev_manage_router_config(action="apply_updates")
    on_disk = router_config.load_router_config()
    assert on_disk["buckets_source"] == "user"
    assert on_disk["buckets"]["b1"] == "claude-haiku-5-5"


def test_background_auto_update_preserves_detected(cached_catalog, monkeypatch):
    _write_config(auto_update=True)
    monkeypatch.setattr(model_catalog, "refresh", lambda: None)
    app._sync_models()
    on_disk = router_config.load_router_config()
    assert on_disk["buckets_source"] == "detected"
    assert on_disk["buckets"]["b1"] == "claude-haiku-5-5"


def test_claude_code_name_maps_to_anthropic_tiers():
    buckets = client_defaults.default_buckets_for_client("Claude Code v2.1")
    assert buckets == dict(router_config.FALLBACK_BUCKETS)


def test_unknown_client_maps_to_none():
    assert client_defaults.default_buckets_for_client("opencode") is None
    assert client_defaults.default_buckets_for_client("") is None


def test_seeds_when_no_config_file_exists():
    # No config file in the conftest-provided tmp home.
    written = router_config.seed_defaults_for_client("Claude Code")
    assert written is not None
    on_disk = router_config.load_router_config()
    assert on_disk["buckets"] == dict(router_config.FALLBACK_BUCKETS)
    assert on_disk["rules"] == []
    assert on_disk["buckets_source"] == "detected"
    assert on_disk["detected_client"] == "Claude Code"


def test_fieldless_existing_config_is_never_reseeded():
    router_config.save_router_config({"buckets": {"b1": "my-model"}, "rules": []})
    before = Path(router_config.ROUTER_CONFIG_PATH).read_text()
    assert router_config.seed_defaults_for_client("Claude Code") is None
    assert Path(router_config.ROUTER_CONFIG_PATH).read_text() == before


def test_detected_config_reseeds_to_current_table():
    router_config.save_router_config({
        "buckets": {"b1": "old"},
        "rules": [{"condition": "c", "target": "b3"}],
        "auto_update": True,
        "buckets_source": "detected",
        "detected_client": "Claude Code",
    })
    written = router_config.seed_defaults_for_client("Claude Code")
    assert written["buckets"] == dict(router_config.FALLBACK_BUCKETS)
    # Non-bucket fields survive the immutable update.
    on_disk = router_config.load_router_config()
    assert on_disk["rules"] == [{"condition": "c", "target": "b3"}]
    assert on_disk["auto_update"] is True
    assert on_disk["detected_client"] == "Claude Code"


def test_unknown_client_never_writes():
    assert router_config.seed_defaults_for_client("opencode") is None
    assert not os.path.exists(router_config.ROUTER_CONFIG_PATH)
