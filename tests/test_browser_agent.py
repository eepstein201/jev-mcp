import pytest
from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import ChoiceQuestion

def test_daemon_choice_mapping():
    dp = DaemonProvider()
    
    # Mock get_logprobs to return logprobs for 'a' and 'b'
    def mock_get_logprobs(prompt, keys):
        return {k: -1.0 for k in keys}, "mock_model"
    
    dp.get_logprobs = mock_get_logprobs
    q = ChoiceQuestion(key="test_q", prompt="Click or Type?", options=["CLICK", "TYPE_TEXT"])
    
    res = dp.evaluate_batch({"dummy": "state"}, [q])
    
    assert "test_q" in res
    ans = res["test_q"]
    assert "confidence" in ans, "DaemonProvider must return confidence"
    assert "CLICK" in ans.get("probabilities", {}), "DaemonProvider must map probabilities back to original options"
    assert "a" not in ans.get("probabilities", {}), "DaemonProvider must not leak letter keys"

def test_kev_choice_confidence():
    kp = KevProvider()
    q = ChoiceQuestion(key="test_q", prompt="Click or Type?", options=["CLICK", "TYPE_TEXT"])
    
    raw_api_response = {
        "choice": "CLICK",
        "confidence": 0.99,
        "probabilities": {"CLICK": 0.99, "TYPE_TEXT": 0.01}
    }
    
    res = kp._parse_answer(q, raw_api_response)
    assert "confidence" in res, "KevProvider must not drop the confidence key"
    assert res["confidence"] == 0.99
