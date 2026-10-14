"""Error-branch coverage for jev_mcp.tools.router."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import app, router_config, server
from jev_mcp.router_config import save_router_config


def test_manage_router_rejects_incomplete_actions():
    assert "required" in server.jev_manage_router_config(
        action="set_bucket", bucket_id=None, model_name=None
    )
    assert "required" in server.jev_manage_router_config(
        action="add_rule", condition=None, target=None
    )
    assert "required" in server.jev_manage_router_config(
        action="remove_rule", rule_id=0
    )


@patch("jev_mcp.app.RoutingProvider")
def test_model_router_reports_engine_failure(mock_provider_class):
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.side_effect = RuntimeError("engine down")
    res = json.loads(server.jev_determine_best_model("any task", estimated_tokens=0))
    assert res["status"] == "ERROR"
    assert "engine down" in res["message"]


@patch("jev_mcp.app.RoutingProvider")
def test_handoff_injects_temperature_notice(mock_provider_class, monkeypatch):
    monkeypatch.setattr(app, "_has_notified_temp", False)
    save_router_config(
        {
            "buckets": {"b1": "m"},
            "rules": [],
            "fitted_temperature": 1.4,
            "temperature_updated_at": "Just now",
        }
    )
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.smart_provider.current_model_id = "/models/Qwen2.5-7B-Instruct.gguf"
    mock_provider.evaluate_batch.return_value = {
        "best_agent": {"choice": "A", "probabilities": {"A": 0.9}}
    }

    res = json.loads(server.jev_agent_handoff("task", {"A": "does things"}))
    assert res["status"] == "HANDOFF_RECOMMENDATION"
    assert "1.4" in res["_system_notice"]


def _handoff_notice(monkeypatch, config, model_id):
    monkeypatch.setattr(app, "_has_notified_temp", False)
    save_router_config(config)
    smart = MagicMock(current_model_id=model_id, model_size=None)
    provider = MagicMock(smart_provider=smart)
    provider.evaluate_batch.return_value = {
        "best_agent": {"choice": "A", "probabilities": {"A": 0.9}}
    }
    with patch("jev_mcp.app.RoutingProvider", return_value=provider):
        return json.loads(server.jev_agent_handoff("task", {"A": "does things"}))


def test_notice_names_the_model_and_uses_its_own_temperature(monkeypatch):
    config = router_config.with_temperature(
        {"buckets": {}, "rules": []}, "/m/Llama-3.1-8B.gguf", 0.6
    )

    res = _handoff_notice(monkeypatch, config, "/m/Llama-3.1-8B.gguf")

    assert "llama-3.1-8b.gguf" in res["_system_notice"]
    assert "0.6" in res["_system_notice"]


def test_notice_is_silent_for_a_model_with_no_temperature_of_its_own(monkeypatch):
    # The legacy global was fitted on qwen2.5-7b; another model runs at the default.
    config = {"buckets": {}, "rules": [], "fitted_temperature": 1.4}

    res = _handoff_notice(monkeypatch, config, "/m/Llama-3.1-8B.gguf")

    assert "_system_notice" not in res


def test_notice_is_silent_when_the_smart_engine_has_not_reported_a_model(monkeypatch):
    config = {"buckets": {}, "rules": [], "fitted_temperature": 1.4}

    res = _handoff_notice(monkeypatch, config, "unknown_model")

    assert "_system_notice" not in res


def test_notice_reports_the_stored_time_for_the_running_model(monkeypatch):
    config = router_config.with_temperature(
        {"buckets": {}, "rules": []},
        "/m/Llama-3.1-8B.gguf",
        0.6,
        now=datetime(2026, 3, 4, 5, 6, 7),
    )

    res = _handoff_notice(monkeypatch, config, "/m/Llama-3.1-8B.gguf")

    assert "2026-03-04 05:06:07" in res["_system_notice"]


def test_notice_reads_the_default_smart_provider_when_none_is_given(monkeypatch):
    monkeypatch.setattr(app, "_has_notified_temp", False)
    save_router_config(
        router_config.with_temperature(
            {"buckets": {}, "rules": []}, "/m/Gemma.gguf", 2.2
        )
    )
    monkeypatch.setattr(
        app.provider.smart_provider, "current_model_id", "/m/Gemma.gguf"
    )

    assert "2.2" in app.get_temp_notice()


@patch("jev_mcp.app.RoutingProvider")
def test_handoff_reports_an_unavailable_engine_instead_of_recommending_an_agent(
    mock_provider_class,
):
    from jev_mcp.provider import EngineUnavailableError

    mock_provider_class.return_value.evaluate_batch.side_effect = (
        EngineUnavailableError("No engine answered: best_agent")
    )

    res = json.loads(
        server.jev_agent_handoff("task", {"A": "does things", "B": "does other things"})
    )

    assert res["status"] == "ERROR"
    assert "recommended_agent" not in res


def test_manage_router_answers_every_action_with_json():
    save_router_config({"buckets": {"b1": "m"}, "rules": []})

    missing = json.loads(
        server.jev_manage_router_config(
            action="set_bucket", bucket_id=None, model_name=None
        )
    )
    bucket = json.loads(
        server.jev_manage_router_config(
            action="set_bucket", bucket_id="b1", model_name="new-model"
        )
    )
    added = json.loads(
        server.jev_manage_router_config(
            action="add_rule", condition="touches auth", target="b3"
        )
    )
    removed = json.loads(
        server.jev_manage_router_config(action="remove_rule", rule_id=0)
    )
    bad_rule = json.loads(
        server.jev_manage_router_config(action="remove_rule", rule_id=9)
    )
    no_flag = json.loads(
        server.jev_manage_router_config(action="set_auto_update", enabled=None)
    )
    unknown = json.loads(server.jev_manage_router_config(action="nonsense"))

    assert [
        missing["status"],
        bad_rule["status"],
        no_flag["status"],
        unknown["status"],
    ] == ["ERROR"] * 4
    assert [bucket["status"], added["status"], removed["status"]] == ["SAVED"] * 3


def test_notice_is_skipped_when_the_config_cannot_be_read(monkeypatch):
    monkeypatch.setattr(app, "_has_notified_temp", False)
    smart = MagicMock(current_model_id="/m/Gemma.gguf", model_size=None)

    with patch("jev_mcp.app.load_router_config", side_effect=OSError("unreadable")):
        assert app.get_temp_notice(smart) is None
