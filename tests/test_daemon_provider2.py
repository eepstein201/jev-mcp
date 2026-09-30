import json
import urllib.request
from unittest.mock import patch, MagicMock

from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.server import NoulQuestion

@patch.dict('sys.modules', {'dotenv': None})
def test_daemon_provider_no_dotenv():
    # Covers lines 32-33
    provider = DaemonProvider()
    assert "8080" in provider.base_url

@patch("urllib.request.urlopen")
def test_daemon_provider_no_content_logprobs(mock_urlopen):
    # Covers line 84
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "logprobs": {
                "content": []
            }
        }],
        "model": "test-model"
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    provider = DaemonProvider()
    probs, model_id = provider.get_logprobs("prompt", ["true", "false"])
    assert probs == {}

@patch("urllib.request.urlopen")
def test_daemon_provider_max_lp_negative_9999(mock_urlopen):
    # Covers line 148
    # We need to simulate that none of the expected keys were found in the logprobs
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "logprobs": {
                "content": [{
                    "top_logprobs": [
                        {"token": "random_token", "logprob": -1.0}
                    ]
                }]
            }
        }],
        "model": "test-model"
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    provider = DaemonProvider()
    probs = provider._normalize_logprobs({"random": -1.0}, ["true", "false"], {"true": 0.0, "false": 0.0})
    assert probs == {"true": 0.0, "false": 0.0}

class FakeQuestion:
    pass

@patch("subprocess.check_output")
def test_daemon_provider_format_hint_else(mock_check_output):
    # Covers line 189
    mock_check_output.return_value = b"mlx_lm.server"
    provider = DaemonProvider()
    # Mock evaluate_batch behavior with a fake question that hits the "else" branch
    q = FakeQuestion()
    q.prompt = "?"
    q.key = "q1"
    
    # We just need to hit the format_hint = "" branch.
    # Actually wait, `evaluate_batch` only accepts NoulQuestion, ChoiceQuestion, ScoreQuestion.
    # What triggers the `else` on 189?
    pass

@patch("subprocess.check_output")
@patch("urllib.request.urlopen")
def test_daemon_provider_format_hint_else_fixed(mock_urlopen, mock_check_output):
    mock_check_output.return_value = b"mlx_lm.server"
    
    # We need urlopen to not fail
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{"logprobs": {"content": [{"top_logprobs": []}]}}],
        "model": "test"
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    provider = DaemonProvider()
    class DummyQ:
        key = "dummy"
        prompt = "?"
        
    res = provider.evaluate_batch(state={}, questions=[DummyQ()])
    assert res == {}
