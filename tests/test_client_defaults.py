"""Tests for per-client router default seeding (client_defaults + router_config)."""
import os
from pathlib import Path

from jev_mcp import client_defaults, router_config


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
