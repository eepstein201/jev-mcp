import json

from jev_mcp import server
from jev_mcp.server import (
    jev_determine_best_model,
    jev_manage_router_config,
    save_router_config,
)


def test_manage_config(tmp_path, monkeypatch):
    # Route all router-config I/O to a temp file — never touch ~/.jev.
    config_path = tmp_path / "router_config.json"
    monkeypatch.setattr(server, "ROUTER_CONFIG_PATH", str(config_path))
    config_path.write_text(json.dumps({
        "buckets": {"b1": "claude-3-5-haiku"},
        "rules": [],
    }))

    # Test viewing
    res = json.loads(jev_manage_router_config(action="view_all"))
    assert "buckets" in res
    assert res["buckets"]["b1"] == "claude-3-5-haiku"

    # Test setting
    jev_manage_router_config(action="set_bucket", bucket_id="b1", model_name="test-model")
    res2 = json.loads(jev_manage_router_config(action="view_all"))
    assert res2["buckets"]["b1"] == "test-model"

    # Test adding rule
    jev_manage_router_config(action="add_rule", condition="test rule", target="b3")
    res3 = json.loads(jev_manage_router_config(action="view_all"))
    assert len(res3["rules"]) > 0
    assert res3["rules"][-1]["condition"] == "test rule"

    # Restore defaults
    res3["buckets"]["b1"] = "claude-3-5-haiku"
    res3["rules"].pop()
    save_router_config(res3)


def test_determine_model_massive_context():
    res = json.loads(jev_determine_best_model(task_description="test", estimated_tokens=150000))
    assert res["status"] == "MASSIVE_CONTEXT_DETECTED"
    assert "warning" in res
