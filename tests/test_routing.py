import pytest
import json
from jev_mcp.server import jev_determine_best_model, jev_manage_router_config, load_router_config, save_router_config

def test_manage_config():
    # Test viewing
    # Reset it
    import json
    import os
    with open(os.path.expanduser("~/.jev/router_config.json"), "r") as f:
        config = json.load(f)
    config["buckets"]["b1"] = "claude-3-5-haiku"
    with open(os.path.expanduser("~/.jev/router_config.json"), "w") as f:
        json.dump(config, f)
        
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

