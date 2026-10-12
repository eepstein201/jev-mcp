import json

from jev_mcp import router_config
from jev_mcp.server import (
    jev_determine_best_model,
    jev_manage_router_config,
    save_router_config,
)


def test_manage_config():
    # conftest points router_config.ROUTER_CONFIG_PATH at a temp file, so ~/.jev is never touched.
    save_router_config({"buckets": {"b1": "claude-3-5-haiku"}, "rules": []})
    assert "jev_home" in router_config.ROUTER_CONFIG_PATH

    # Test viewing
    res = json.loads(jev_manage_router_config(action="view_all"))
    assert "buckets" in res
    assert res["buckets"]["b1"] == "claude-3-5-haiku"

    # Test setting
    jev_manage_router_config(
        action="set_bucket", bucket_id="b1", model_name="test-model"
    )
    res2 = json.loads(jev_manage_router_config(action="view_all"))
    assert res2["buckets"]["b1"] == "test-model"

    # Test adding rule
    jev_manage_router_config(action="add_rule", condition="test rule", target="b3")
    res3 = json.loads(jev_manage_router_config(action="view_all"))
    assert len(res3["rules"]) > 0
    assert res3["rules"][-1]["condition"] == "test rule"


def test_determine_model_massive_context():
    res = json.loads(
        jev_determine_best_model(task_description="test", estimated_tokens=150000)
    )
    assert res["status"] == "MASSIVE_CONTEXT_DETECTED"
    assert "warning" in res


def _routing_provider():
    from jev_mcp.routing_provider import RoutingProvider

    return RoutingProvider()


def test_routing_raises_when_no_engine_answers():
    import pytest
    from unittest.mock import patch
    from jev_mcp.provider import EngineUnavailableError, NoulQuestion

    provider = _routing_provider()
    with (
        patch.object(provider.fast_provider, "evaluate_batch", return_value={}),
        patch.object(provider.smart_provider, "evaluate_batch", return_value={}),
    ):
        with pytest.raises(EngineUnavailableError, match="q1"):
            provider.evaluate_batch("state", [NoulQuestion(key="q1", prompt="p")])


def test_routing_raises_when_a_large_state_goes_unanswered():
    import pytest
    from unittest.mock import patch
    from jev_mcp.provider import EngineUnavailableError, NoulQuestion

    provider = _routing_provider()
    with patch.object(provider.smart_provider, "evaluate_batch", return_value={}):
        with pytest.raises(EngineUnavailableError):
            provider.evaluate_batch("x" * 20000, [NoulQuestion(key="q1", prompt="p")])


def test_routing_accepts_a_zero_confidence_answer_from_the_smart_tier():
    from unittest.mock import patch
    from jev_mcp.provider import NoulQuestion

    # The confidence gate answering "no signal" is an answer; only silence is an outage.
    gated = {"q1": {"probabilities": {"true": 0.0, "false": 0.0}, "noul": 0.0}}
    provider = _routing_provider()
    with (
        patch.object(provider.fast_provider, "evaluate_batch", return_value={}),
        patch.object(provider.smart_provider, "evaluate_batch", return_value=gated),
    ):
        results = provider.evaluate_batch("state", [NoulQuestion(key="q1", prompt="p")])

    assert results["q1"]["noul"] == 0.0


def test_routing_marks_escalation_without_changing_the_engines_result():
    from unittest.mock import patch
    from jev_mcp.provider import NoulQuestion

    smart_answer = {"q1": {"probabilities": {"true": 0.9, "false": 0.1}, "noul": 0.9}}
    provider = _routing_provider()
    with (
        patch.object(provider.fast_provider, "evaluate_batch", return_value={}),
        patch.object(
            provider.smart_provider, "evaluate_batch", return_value=smart_answer
        ),
    ):
        results = provider.evaluate_batch("state", [NoulQuestion(key="q1", prompt="p")])

    assert results["q1"]["escalated"] is True
    assert "escalated" not in smart_answer["q1"]
