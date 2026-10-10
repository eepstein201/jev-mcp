import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import ChoiceQuestion

def test_daemon_mapping():
    print("--- Testing DaemonProvider Mapping ---")
    dp = DaemonProvider()
    
    # Mock get_logprobs to return logprobs for 'a' and 'b'
    def mock_get_logprobs(prompt, keys):
        return {k: -1.0 for k in keys}, "mock_model"
    
    dp.get_logprobs = mock_get_logprobs
    
    q = ChoiceQuestion(key="test_q", prompt="Click or Type?", options=["CLICK", "TYPE_TEXT"])
    
    res = dp.evaluate_batch({"dummy": "state"}, [q])
    print(f"DaemonProvider output for ChoiceQuestion: {res}")
    
    # Check if 'CLICK' is in probabilities
    probs = res.get("test_q", {}).get("probabilities", {})
    if "CLICK" not in probs:
        print("HOLE CONFIRMED: DaemonProvider returns letters (a, b) instead of original options (CLICK, TYPE_TEXT)!")
    if "confidence" not in res.get("test_q", {}):
        print("HOLE CONFIRMED: DaemonProvider is missing 'confidence' key!")

def test_kev_confidence():
    print("\n--- Testing KevProvider Confidence ---")
    kp = KevProvider()
    
    q = ChoiceQuestion(key="test_q", prompt="Click or Type?", options=["CLICK", "TYPE_TEXT"])
    
    # KevProvider uses singledispatchmethod _parse_answer
    # We can just call _parse_answer directly to see how it formats the response
    raw_api_response = {
        "choice": "CLICK",
        "confidence": 0.99,
        "probabilities": {"CLICK": 0.99, "TYPE_TEXT": 0.01}
    }
    
    res = kp._parse_answer(q, raw_api_response)
    print(f"KevProvider parsed output: {res}")
    
    if "confidence" not in res:
        print("HOLE CONFIRMED: KevProvider strips the 'confidence' key!")

if __name__ == "__main__":
    test_daemon_mapping()
    test_kev_confidence()
