"""Error-branch coverage for jev_mcp.tools.router."""
import json
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import app, server
from jev_mcp.router_config import save_router_config


def test_manage_router_rejects_incomplete_actions():
    assert "required" in server.jev_manage_router_config(action="set_bucket", bucket_id=None, model_name=None)
    assert "required" in server.jev_manage_router_config(action="add_rule", condition=None, target=None)
    assert "required" in server.jev_manage_router_config(action="remove_rule", rule_id=0)


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
    save_router_config({
        "buckets": {"b1": "m"},
        "rules": [],
        "fitted_temperature": 1.4,
        "temperature_updated_at": "Just now",
    })
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {
        "best_agent": {"choice": "A", "probabilities": {"A": 0.9}}
    }

    res = json.loads(server.jev_agent_handoff("task", {"A": "does things"}))
    assert res["status"] == "HANDOFF_RECOMMENDATION"
    assert "1.4" in res["_system_notice"]


@patch("jev_mcp.app.RoutingProvider")
def test_handoff_reports_an_unavailable_engine_instead_of_recommending_an_agent(mock_provider_class):
    from jev_mcp.provider import EngineUnavailableError

    mock_provider_class.return_value.evaluate_batch.side_effect = EngineUnavailableError("No engine answered: best_agent")

    res = json.loads(server.jev_agent_handoff("task", {"A": "does things", "B": "does other things"}))

    assert res["status"] == "ERROR"
    assert "recommended_agent" not in res


def test_manage_router_answers_every_action_with_json():
    save_router_config({"buckets": {"b1": "m"}, "rules": []})

    missing = json.loads(server.jev_manage_router_config(action="set_bucket", bucket_id=None, model_name=None))
    bucket = json.loads(server.jev_manage_router_config(action="set_bucket", bucket_id="b1", model_name="new-model"))
    added = json.loads(server.jev_manage_router_config(action="add_rule", condition="touches auth", target="b3"))
    removed = json.loads(server.jev_manage_router_config(action="remove_rule", rule_id=0))
    bad_rule = json.loads(server.jev_manage_router_config(action="remove_rule", rule_id=9))
    no_flag = json.loads(server.jev_manage_router_config(action="set_auto_update", enabled=None))
    unknown = json.loads(server.jev_manage_router_config(action="nonsense"))

    assert [missing["status"], bad_rule["status"], no_flag["status"], unknown["status"]] == ["ERROR"] * 4
    assert [bucket["status"], added["status"], removed["status"]] == ["SAVED"] * 3
