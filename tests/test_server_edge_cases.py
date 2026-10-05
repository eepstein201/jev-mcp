import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from jev_mcp import server
from jev_mcp.provider import NoulQuestion, ChoiceQuestion, ScoreQuestion
from jev_mcp.linter import LintFinding, PreflightReport

def test_dotenv_import_error():
    with patch.dict(sys.modules, {"dotenv": None}):
        import importlib
        importlib.reload(server)

@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
def test_model_based_linting_exception(mock_ctl, mock_eval):
    mock_ctl.return_value = 0
    server.provider.max_tokens = 1000
    mock_eval.side_effect = [Exception("Model failed"), {"q1": {"noul": 0.9}}]
    q = NoulQuestion(key="q1", prompt="Are you there?")
    res_str = server.jev_evaluate_batch(state={"state": 1}, questions=[q])
    # The result should be a successful string without MagicMock
    import traceback
    assert "SUCCESS" in res_str
    
@patch("jev_mcp.server.linter")
@patch("jev_mcp.server.provider")
def test_unknown_question_type_and_semantic_dedupe(mock_provider, mock_linter):
    class UnknownQuestion:
        def __init__(self):
            self.key = "u1"
            self.prompt = "unknown"
        def dict(self):
            return {"key": "u1", "prompt": "unknown"}
    q = UnknownQuestion()
    
    mock_linter.lint.return_value = PreflightReport(
        is_valid=False,
        findings=[
            LintFinding(
                severity="ERROR",
                code="GENERATIVE_INTENT_DETECTED",
                message="test",
                suggestion="test"
            )
        ]
    )
    
    mock_provider.evaluate_batch.side_effect = [
        {"u1": {"noul": 0.9}},
        {"u1": {"noul": 0.9}}
    ]
    mock_provider.check_token_limit.return_value = 0
    server.provider.max_tokens = 1000
    mock_provider.name = "mock"
    mock_provider.model_id = "mock"
    
    res_str = server.jev_evaluate_batch(state={"state": 1}, questions=[q])
    res = json.loads(res_str)
    
    assert res["status"] == "REJECTED_BY_LINTER"
    assert "original_findings" in res

@patch("jev_mcp.server.provider")
@patch("urllib.request.urlopen")
def test_jev_optimize_prompt_unknown_type_and_json_cleaning(mock_urlopen, mock_provider):
    mock_provider.evaluate_batch.return_value = {} # hit 523
    class UnknownQuestion:
        def __init__(self):
            self.key = "u1"
            self.prompt = "unknown"
        def dict(self):
            return {"key": "u1", "prompt": "unknown", "type": "noul"}
    q = UnknownQuestion()
    
    mock_res = MagicMock()
    response_data = {
        "choices": [
            {"message": {"content": "```json\n{\"variations\": \"not_a_list\"}\n```"}}
        ]
    }
    mock_res.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_res.__enter__.return_value = mock_res
    mock_urlopen.return_value = mock_res

    res_str = server.jev_optimize_prompt(state={"state": 1}, question=q)
    assert json.loads(res_str).get("error") == "No variations evaluated."
    
    response_data = {
        "choices": [
            {"message": {"content": "```\n{\"variations\": \"not_a_list\"}\n```"}}
        ]
    }
    mock_res.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_urlopen.return_value = mock_res
    res_str = server.jev_optimize_prompt(state={"state": 1}, question=q)
    assert json.loads(res_str).get("error") == "No variations evaluated."

def test_calibrate_threshold_score_question_and_else_branch():
    q = ScoreQuestion(key="sq", prompt="score?", labels=["1", "2"])
    states = [{"state": i, "expected": "1"} for i in range(11)]
    
    with patch("jev_mcp.server.provider") as mock_provider:
        results = [{"sq": {"probabilities": {"2": 1.0}}}] + [{"sq": {}}] * 10
        mock_provider.evaluate_dataset.return_value = results
        
        res_str = server.jev_calibrate_threshold(
            dataset=states,
            question=q
        )
    assert "Safe" in res_str or "Unusable" in res_str

@patch("urllib.request.urlopen")
def test_explain_decision_truncation_and_json(mock_urlopen):
    q = NoulQuestion(key="nq", prompt="a" * 1500)
    state = "s" * 20000
    decision = "d" * 200
    
    mock_res = MagicMock()
    response_data = {
        "choices": [
            {"message": {"content": "```json\n{\"evidence_sentence\": \"done\", \"reasoning\": \"done\"}\n```"}}
        ]
    }
    mock_res.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_res.__enter__.return_value = mock_res
    mock_urlopen.return_value = mock_res
    
    res = server.jev_explain_decision(state=state, question=q, decision=decision)
    assert "done" in res
    
    response_data = {
        "choices": [
            {"message": {"content": "```\n{\"evidence_sentence\": \"done\", \"reasoning\": \"done\"}\n```"}}
        ]
    }
    mock_res.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_urlopen.return_value = mock_res
    res = server.jev_explain_decision(state=state, question=q, decision=decision)
    assert "done" in res

@patch("urllib.request.urlopen")
def test_generate_synthetic_dataset_score_truncation_and_json(mock_urlopen):
    q = ScoreQuestion(key="sq", prompt="q" * 1500, labels=["a"])
    
    mock_res = MagicMock()
    response_data = {
        "choices": [
            {"message": {"content": "```json\n{\"not\": \"list\"}\n```"}}
        ]
    }
    mock_res.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_res.__enter__.return_value = mock_res
    mock_urlopen.return_value = mock_res
    
    with patch("subprocess.check_output") as mock_sp:
        mock_sp.return_value = b"mlx_lm"
        res = server.jev_generate_synthetic_dataset(question=q, num_cases=1)
        assert "INSTRUCTION TO PRIMARY LLM" in res

    response_data = {
        "choices": [
            {"message": {"content": "```\n[{\"data\": 1}]\n```"}}
        ]
    }
    mock_res.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_urlopen.return_value = mock_res
    
    with patch("subprocess.check_output") as mock_sp:
        mock_sp.return_value = b"mlx_lm"
        res = server.jev_generate_synthetic_dataset(question=q, num_cases=1)
        assert "data" in res

def test_main():
    with patch("jev_mcp.server.mcp.run") as mock_run:
        server.main()
        mock_run.assert_called_once_with(transport="stdio")

    with patch("jev_mcp.server.main") as mock_main:
        with patch.dict("sys.modules", {"jev_mcp.server": MagicMock(__name__="__main__")}):
            pass


@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
def test_coverage_231(mock_ctl, mock_eval):
    # Model-based linting raises (warning path); static linter passes; eval succeeds.
    mock_ctl.return_value = 0
    server.provider.max_tokens = 1000
    mock_eval.side_effect = [Exception("test231"), {"q1": {"noul": 0.9}}]
    q = NoulQuestion(key="q1", prompt="prompt")
    res_str = server.jev_evaluate_batch(state={"text": "t"}, questions=[q])
    assert "SUCCESS" in res_str

@patch("jev_mcp.server.call_fast_autofixer")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_coverage_267(mock_lint, mock_ctl, mock_eval, mock_autofix):
    # Both linters flag generative intent (dedupe path); autofixer fails -> rejected.
    mock_lint.return_value = PreflightReport(
        is_valid=False,
        findings=[
            LintFinding(severity="ERROR", code="GENERATIVE_INTENT_DETECTED", message="m", suggestion="s")
        ]
    )
    mock_ctl.return_value = 0
    server.provider.max_tokens = 1000
    mock_eval.return_value = {"q_0": {"noul": 0.99}}
    mock_autofix.side_effect = Exception("autofixer down")

    q = NoulQuestion(key="q1", prompt="Summarize this document.")
    res_str = server.jev_evaluate_batch(state={}, questions=[q])
    assert "REJECTED_BY_LINTER" in res_str

@patch("urllib.request.urlopen")
def test_coverage_472(mock_urlopen):
    # Non-dict JSON from the optimizer falls through to the error return.
    mock_res = MagicMock()
    mock_res.read.return_value = json.dumps({
        "choices": [{"message": {"content": "```json\n123\n```"}}]
    }).encode("utf-8")
    mock_res.__enter__.return_value = mock_res
    mock_urlopen.return_value = mock_res

    q = NoulQuestion(key="q1", prompt="prompt")
    res = server.jev_optimize_prompt(state={}, question=q)
    assert "Optimizer Generator Failed" in res

def test_coverage_633():
    # Threshold table across a mixed automation distribution (FP + TPs + abstains).
    q = ScoreQuestion(key="sq", prompt="score?", labels=["1", "2"])
    states = [{"state": i, "expected": "1"} for i in range(100)]
    states[0]["expected"] = "2"
    with patch.object(server.provider, "evaluate_dataset") as mock_eval:
        results = [{"probabilities": {"1": 0.9}}]  # the false positive
        results += [{"probabilities": {"1": 0.9}} for _ in range(4)]  # true positives
        results += [{"probabilities": {"1": 0.1}} for _ in range(95)]  # abstains
        mock_eval.return_value = results
        res_str = server.jev_calibrate_threshold(dataset=states, question=q)

    res = json.loads(res_str)
    assert res["status"] == "CALIBRATION_COMPLETE"
    assert "| > 0.50 |" in res["markdown_report"]

def test_coverage_983():
    # runpy re-executes the module into a fresh namespace with a NEW MCPServer
    # instance, so patch the stubbed class itself (importing the real "mcp"
    # package under conftest's stubs would fail).
    import runpy
    import sys
    dummy_server_cls = sys.modules["mcp.server.mcpserver"].MCPServer
    with patch.object(dummy_server_cls, "run") as mock_run:
        runpy.run_module("jev_mcp.server", run_name="__main__")
    mock_run.assert_called_once_with(transport="stdio")

def test_new_prompts():
    from jev_mcp.server import scan_repo_prompt, read_file_prompt
    assert "scan" in scan_repo_prompt()
    assert "read" in read_file_prompt()
