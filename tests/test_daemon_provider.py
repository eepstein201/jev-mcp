import pytest
import json
import math
from unittest.mock import patch, MagicMock
from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.provider import NoulQuestion, ChoiceQuestion, ScoreQuestion

@pytest.fixture
def provider():
    return DaemonProvider()

@patch("subprocess.check_output")
def test_get_daemon_pid(mock_check_output, provider):
    mock_check_output.return_value = b"1234\n"
    pid = provider._get_daemon_pid()
    assert pid == "1234"
    
    mock_check_output.side_effect = Exception("error")
    pid = provider._get_daemon_pid()
    assert pid == "unknown_pid"

@patch("urllib.request.urlopen")
def test_get_logprobs_success(mock_urlopen, provider):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "test-model",
        "choices": [{
            "logprobs": {
                "content": [{
                    "top_logprobs": [
                        {"token": " true", "logprob": -0.1},
                        {"token": " false", "logprob": -2.3}
                    ]
                }]
            }
        }]
    }).encode("utf-8")
    
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager
    
    lps, model_id = provider.get_logprobs("prompt", ["true", "false"])
    
    assert model_id == "test-model"
    assert "true" in lps
    assert "false" in lps
    assert lps["true"] == -0.1
    assert lps["false"] == -2.3

@patch("urllib.request.urlopen")
def test_get_logprobs_summing_duplicates(mock_urlopen, provider):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "test-model",
        "choices": [{
            "logprobs": {
                "content": [{
                    "top_logprobs": [
                        {"token": " true", "logprob": math.log(0.6)},
                        {"token": "true", "logprob": math.log(0.3)},
                        {"token": " false", "logprob": math.log(0.1)}
                    ]
                }]
            }
        }]
    }).encode("utf-8")
    
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager
    
    lps, model_id = provider.get_logprobs("prompt", ["true", "false"])
    
    # 0.6 + 0.3 = 0.9 => math.log(0.9) approx -0.105
    assert math.isclose(lps["true"], math.log(0.9), rel_tol=1e-5)
    assert math.isclose(lps["false"], math.log(0.1), rel_tol=1e-5)

@patch("urllib.request.urlopen")
def test_get_logprobs_missing_fields(mock_urlopen, provider):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "test-model"
    }).encode("utf-8")
    
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager
    
    lps, model_id = provider.get_logprobs("prompt", ["true", "false"])
    assert lps == {}
    
    # missing logprobs
    mock_response.read.return_value = json.dumps({
        "model": "test-model",
        "choices": [{}]
    }).encode("utf-8")
    lps, model_id = provider.get_logprobs("prompt", ["true", "false"])
    assert lps == {}

    # missing content
    mock_response.read.return_value = json.dumps({
        "model": "test-model",
        "choices": [{"logprobs": {}}]
    }).encode("utf-8")
    lps, model_id = provider.get_logprobs("prompt", ["true", "false"])
    assert lps == {}

@patch("urllib.request.urlopen")
def test_get_logprobs_exception(mock_urlopen, provider):
    mock_urlopen.side_effect = Exception("timeout")
    lps, model_id = provider.get_logprobs("prompt", ["true", "false"])
    assert lps == {}
    assert model_id == "unknown_model"

def test_normalize_logprobs(provider):
    # Test normalization
    lps = {"true": math.log(0.6), "false": math.log(0.4)}
    prior = {"true": math.log(0.5), "false": math.log(0.5)}
    
    result = provider._normalize_logprobs(lps, ["true", "false"], prior)
    assert "true" in result
    assert "false" in result
    # We can trust that math happens correctly, but let's just make sure it returns dict with probs
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["true"] > result["false"]
    
    # Low confidence rejection
    lps_low = {"true": math.log(0.01), "false": math.log(0.01)}
    result_low = provider._normalize_logprobs(lps_low, ["true", "false"], prior)
    assert result_low["true"] == 0.0
    assert result_low["false"] == 0.0

@patch.object(DaemonProvider, "_get_daemon_pid", return_value="1234")
@patch.object(DaemonProvider, "get_logprobs")
@patch("subprocess.check_output")
def test_evaluate_batch(mock_check_output, mock_get_logprobs, mock_get_pid, provider):
    mock_check_output.return_value = b"mlx_lm 0.5B"
    mock_get_logprobs.side_effect = [
        ({"true": math.log(0.8), "false": math.log(0.2)}, "test-model"), # Actual
        ({"true": math.log(0.5), "false": math.log(0.5)}, "test-model"), # Prior
        
        ({"a": math.log(0.7), "b": math.log(0.3)}, "test-model"), # Actual choice
        ({"a": math.log(0.5), "b": math.log(0.5)}, "test-model"), # Prior choice
        
        ({"1": math.log(0.9), "2": math.log(0.1)}, "test-model"), # Actual score
        ({"1": math.log(0.5), "2": math.log(0.5)}, "test-model"), # Prior score
    ]
    
    q1 = NoulQuestion(key="q1", prompt="Is this true?")
    q2 = ChoiceQuestion(key="q2", prompt="Pick one", options=["Option A", "Option B"])
    q3 = ScoreQuestion(key="q3", prompt="Rate this", labels=["1", "2"])
    
    results = provider.evaluate_batch({"data": "test"}, [q1, q2, q3])
    
    assert "q1" in results
    assert "noul" in results["q1"]
    
    assert "q2" in results
    assert "choice" in results["q2"]
    
    assert "q3" in results
    assert "score" in results["q3"]

@patch.object(DaemonProvider, "_get_daemon_pid", return_value="1234")
@patch.object(DaemonProvider, "get_logprobs")
@patch("subprocess.check_output")
def test_evaluate_batch_large_model(mock_check_output, mock_get_logprobs, mock_get_pid, provider):
    # "0.5B" not in output, triggering large model logic
    mock_check_output.return_value = b"mlx_lm 7B"
    mock_get_logprobs.side_effect = [
        ({"true": math.log(0.8), "false": math.log(0.2)}, "test-model"), # Actual
        ({"true": math.log(0.5), "false": math.log(0.5)}, "test-model"), # Prior
    ]
    
    q1 = NoulQuestion(key="q1", prompt="Is this true?")
    results = provider.evaluate_batch("test state string", [q1])
    assert "q1" in results

@patch.object(DaemonProvider, "_get_daemon_pid", return_value="1234")
@patch.object(DaemonProvider, "get_logprobs")
@patch("subprocess.check_output")
def test_evaluate_batch_subprocess_exception(mock_check_output, mock_get_logprobs, mock_get_pid, provider):
    mock_check_output.side_effect = Exception("error")
    # Should fallback to small model logic
    mock_get_logprobs.side_effect = [
        ({"true": math.log(0.8), "false": math.log(0.2)}, "test-model"), # Actual
        ({"true": math.log(0.5), "false": math.log(0.5)}, "test-model"), # Prior
    ]
    
    # Passing a large state string to test truncation
    long_state = "x" * 20000
    q1 = NoulQuestion(key="q1", prompt="Is this true?")
    results = provider.evaluate_batch(long_state, [q1])
    assert "q1" in results
    
@patch.object(DaemonProvider, "evaluate_batch")
def test_evaluate_dataset(mock_evaluate_batch, provider):
    mock_evaluate_batch.return_value = {
        "q1": {"probabilities": {"true": 0.8, "false": 0.2}}
    }
    
    q = NoulQuestion(key="q1", prompt="Test?")
    dataset = [
        {"state": {"data": "test"}, "expected": True},
        {"state": {"data": "test2"}, "expected": False}
    ]
    
    results = provider.evaluate_dataset(dataset, q)
    assert len(results) == 2
    assert results[0]["expected"] is True
    assert results[0]["probabilities"] == {"true": 0.8, "false": 0.2}

def test_check_token_limit(provider):
    state = "x" * 40
    assert provider.check_token_limit(state) == 10
    assert provider.check_token_limit({"data": "x" * 40}) > 10

def test_normalize_logprobs_dcpmi_math(provider):
    # Test condition: max_lp == -9999.0
    lps = {"true": -9999.0, "false": -9999.0}
    prior = {"true": math.log(0.5), "false": math.log(0.5)}
    
    # We need p_act sum > 0.05 to pass Absolute Confidence Gate
    # Wait, if lps is -9999.0, p_act is 0.
    result = provider._normalize_logprobs(lps, ["true", "false"], prior)
    assert result == {"true": 0.0, "false": 0.0}
    
    # If p_act > 0.05 but some max_lp ends up -9999.0 (not really possible unless lps has only values where prior smoothed goes to inf, which isn't possible because smoothed prior is bounded).
    
    lps = {"true": math.log(0.01), "false": math.log(0.01)} # sum is 0.02, < 0.05
    result = provider._normalize_logprobs(lps, ["true", "false"], prior)
    assert result == {"true": 0.0, "false": 0.0}

def test_env_var():
    import os
    with patch.dict(os.environ, {"JEV_DAEMON_PORT": "9090"}):
        p = DaemonProvider()
        assert p.base_url == "http://127.0.0.1:9090/v1/chat/completions"

@patch("subprocess.check_output")
@patch.object(DaemonProvider, "_get_daemon_pid", return_value="1234")
@patch.object(DaemonProvider, "get_logprobs")
def test_pgrep_runs_once_per_batch_not_per_question(mock_get_logprobs, mock_pid, mock_check_output, provider):
    mock_check_output.return_value = b"mlx_lm 7B"
    # q1: actual + prior (prior then cached for identical prompts), q2/q3: actual only
    mock_get_logprobs.side_effect = [
        ({"true": math.log(0.8), "false": math.log(0.2)}, "test-model"),
        ({"true": math.log(0.5), "false": math.log(0.5)}, "test-model"),
        ({"true": math.log(0.8), "false": math.log(0.2)}, "test-model"),
        ({"true": math.log(0.8), "false": math.log(0.2)}, "test-model"),
    ]
    questions = [NoulQuestion(key=f"q{i}", prompt="same prompt") for i in range(3)]
    provider.evaluate_batch("state", questions)
    assert mock_check_output.call_count == 1

@patch.object(DaemonProvider, "_get_daemon_pid", return_value="1234")
@patch.object(DaemonProvider, "get_logprobs")
@patch("subprocess.check_output", return_value=b"mlx_lm 7B")
def test_prior_cache_shared_across_instances(mock_check_output, mock_get_logprobs, mock_pid):
    # The empty-payload prior must survive across DaemonProvider instances
    # (RoutingProvider constructs a fresh one per tool call).
    mock_get_logprobs.side_effect = [
        ({"true": math.log(0.8), "false": math.log(0.2)}, "m1"),  # p1 actual
        ({"true": math.log(0.5), "false": math.log(0.5)}, "m1"),  # p1 prior (fills cache)
        ({"true": math.log(0.8), "false": math.log(0.2)}, "m1"),  # p2 actual (prior = cache hit)
    ]
    q = NoulQuestion(key="q", prompt="distinctive prior-cache prompt")
    p1 = DaemonProvider()
    p2 = DaemonProvider()
    p1.evaluate_batch("s", [q])
    p2.evaluate_batch("s", [q])
    assert mock_get_logprobs.call_count == 3
