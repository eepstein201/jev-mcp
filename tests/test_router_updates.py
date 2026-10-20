import json
import os
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import app, model_catalog, router_config, server


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

STALE = {
    "b1": "claude-3-5-haiku",
    "b2": "claude-sonnet-5",
    "b3": "claude-opus-5-5",
    "b4": "my-local-llama",
}


@pytest.fixture
def cached_catalog():
    model_catalog.save_cache(CATALOG)


@pytest.fixture(autouse=True)
def fresh_notice_state(monkeypatch):
    monkeypatch.setattr(app, "_has_notified_models", False)


def _write_config(**extra):
    router_config.save_router_config(
        {"buckets": dict(STALE), "rules": [{"condition": "c", "target": "b3"}], **extra}
    )


def _read_config():
    with open(router_config.ROUTER_CONFIG_PATH) as f:
        return json.load(f)


# --- defaults for a new install ---


def test_a_new_install_gets_current_models_even_with_no_catalog():
    buckets = router_config.load_router_config()["buckets"]

    assert buckets == router_config.FALLBACK_BUCKETS
    assert not any("claude-3" in model for model in buckets.values())


def test_a_new_install_takes_its_buckets_from_the_catalog_when_one_is_cached(
    cached_catalog,
):
    assert router_config.load_router_config()["buckets"] == {
        "b1": "claude-haiku-5-5",
        "b2": "claude-sonnet-5-5",
        "b3": "claude-opus-5-5",
        "b4": "claude-fable-5-1",
    }


# --- applying updates to a config ---


def test_pending_updates_cover_only_buckets_with_a_newer_model_in_their_family(
    cached_catalog,
):
    updates = router_config.pending_model_updates({"buckets": dict(STALE)})

    assert {bucket: u["newer"] for bucket, u in updates.items()} == {
        "b1": "claude-haiku-5-5",
        "b2": "claude-sonnet-5-5",
    }


def test_applying_updates_returns_a_new_config_and_leaves_the_rest_alone(
    cached_catalog,
):
    config = {
        "buckets": dict(STALE),
        "rules": [{"condition": "c", "target": "b3"}],
        "fitted_temperature": 1.4,
    }

    updated = router_config.apply_model_updates(
        config, router_config.pending_model_updates(config)
    )

    assert updated["buckets"] == {
        **STALE,
        "b1": "claude-haiku-5-5",
        "b2": "claude-sonnet-5-5",
    }
    assert updated["rules"] == config["rules"]
    assert updated["fitted_temperature"] == 1.4
    assert updated["models_updated_at"]
    assert config["buckets"] == STALE  # the caller's config is not mutated


# --- telling the user ---


def test_view_all_lists_newer_models_and_asks_about_automatic_updates(cached_catalog):
    _write_config()

    out = json.loads(server.jev_manage_router_config(action="view_all"))

    assert (
        out["buckets"] == STALE
    )  # the user's models are kept until they say otherwise
    assert out["available_model_updates"]["b1"]["newer"] == "claude-haiku-5-5"
    assert "apply_updates" in out["_model_update_notice"]
    assert "set_auto_update" in out["_model_update_notice"]


def test_the_automatic_update_question_is_not_repeated_once_answered(cached_catalog):
    _write_config(auto_update=False)

    out = json.loads(server.jev_manage_router_config(action="view_all"))

    assert "apply_updates" in out["_model_update_notice"]
    assert "set_auto_update" not in out["_model_update_notice"]


def test_view_all_is_unchanged_when_every_bucket_is_current(cached_catalog):
    router_config.save_router_config(
        {"buckets": {"b1": "claude-haiku-5-5"}, "rules": []}
    )

    out = json.loads(server.jev_manage_router_config(action="view_all"))

    assert out == {"buckets": {"b1": "claude-haiku-5-5"}, "rules": []}


def test_model_router_mentions_newer_models_once_per_session(cached_catalog):
    _write_config()
    answer = {"complexity": {"score": "2", "probabilities": {}}}

    with patch.object(app, "RoutingProvider") as provider_cls:
        provider_cls.return_value.evaluate_batch.return_value = answer
        first = json.loads(
            server.jev_determine_best_model(
                task_description="rename a variable", estimated_tokens=0
            )
        )
        second = json.loads(
            server.jev_determine_best_model(
                task_description="rename a variable", estimated_tokens=0
            )
        )

    assert "claude-sonnet-5-5" in first["_model_update_notice"]
    assert "_model_update_notice" not in second


# --- acting on the answer ---


def test_apply_updates_moves_buckets_to_the_newer_models(cached_catalog):
    _write_config()

    out = json.loads(server.jev_manage_router_config(action="apply_updates"))

    assert out["status"] == "UPDATED"
    assert out["updated"] == {"b1": "claude-haiku-5-5", "b2": "claude-sonnet-5-5"}
    assert _read_config()["buckets"] == {
        **STALE,
        "b1": "claude-haiku-5-5",
        "b2": "claude-sonnet-5-5",
    }


def test_apply_updates_says_so_when_there_is_nothing_to_do(cached_catalog):
    router_config.save_router_config(
        {"buckets": {"b1": "claude-haiku-5-5"}, "rules": []}
    )

    out = json.loads(server.jev_manage_router_config(action="apply_updates"))

    assert out["status"] == "UP_TO_DATE"


def test_turning_automatic_updates_on_saves_the_choice_and_updates_now(cached_catalog):
    _write_config()

    out = json.loads(
        server.jev_manage_router_config(action="set_auto_update", enabled=True)
    )

    saved = _read_config()
    assert out["auto_update"] is True
    assert saved["auto_update"] is True
    assert saved["buckets"]["b1"] == "claude-haiku-5-5"


def test_turning_automatic_updates_off_saves_the_choice_and_keeps_the_models(
    cached_catalog,
):
    _write_config()

    server.jev_manage_router_config(action="set_auto_update", enabled=False)

    saved = _read_config()
    assert saved["auto_update"] is False
    assert saved["buckets"] == STALE


def test_set_auto_update_needs_a_yes_or_no():
    _write_config()

    out = server.jev_manage_router_config(action="set_auto_update")

    assert "enabled" in out
    assert "auto_update" not in _read_config()


# --- start-up sync ---


def test_startup_sync_updates_buckets_when_the_user_opted_in():
    _write_config(auto_update=True)

    with patch.object(
        model_catalog, "refresh", side_effect=lambda: model_catalog.save_cache(CATALOG)
    ):
        server._sync_models()

    assert _read_config()["buckets"]["b1"] == "claude-haiku-5-5"


@pytest.mark.parametrize("extra", [{}, {"auto_update": False}])
def test_startup_sync_never_changes_models_without_consent(extra):
    _write_config(**extra)

    with patch.object(
        model_catalog, "refresh", side_effect=lambda: model_catalog.save_cache(CATALOG)
    ):
        server._sync_models()

    assert _read_config()["buckets"] == STALE


def test_startup_sync_does_not_create_a_config_file():
    with patch.object(
        model_catalog, "refresh", side_effect=lambda: model_catalog.save_cache(CATALOG)
    ):
        server._sync_models()

    assert not os.path.exists(router_config.ROUTER_CONFIG_PATH)


def test_startup_sync_survives_a_failure():
    with patch.object(model_catalog, "refresh", side_effect=RuntimeError("boom")):
        server._sync_models()  # must not raise


def test_model_sync_runs_in_the_background_at_startup(monkeypatch):
    monkeypatch.delenv("JEV_SKIP_MODEL_SYNC", raising=False)

    with patch("threading.Thread") as thread:
        server.start_model_sync()

    assert thread.call_args.kwargs["target"] is server._sync_models
    assert thread.call_args.kwargs["daemon"] is True
    thread.return_value.start.assert_called_once()


def test_model_sync_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("JEV_SKIP_MODEL_SYNC", "1")

    with patch("threading.Thread") as thread:
        server.start_model_sync()

    thread.assert_not_called()


def test_main_starts_the_model_sync_before_serving(monkeypatch):
    monkeypatch.setattr("sys.argv", ["jev-mcp"])

    with (
        patch.object(server, "start_model_sync") as sync,
        patch.object(server.mcp, "run", MagicMock()),
    ):
        server.main()

    sync.assert_called_once()


def test_get_model_notice_returns_none_when_the_config_cannot_be_read(monkeypatch):
    def unreadable():
        raise OSError("disk error")

    monkeypatch.setattr(app, "load_router_config", unreadable)

    assert app.get_model_notice() is None
