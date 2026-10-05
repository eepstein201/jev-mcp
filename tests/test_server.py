import pytest
import json
import os
import urllib.request
import subprocess
from unittest.mock import patch, MagicMock

from jev_mcp import server
from jev_mcp.provider import NoulQuestion, ChoiceQuestion, ScoreQuestion
from jev_mcp.linter import LintFinding, PreflightReport


def test_clean_json():
    # Test valid JSON extraction
    text = "```json\n{\"test\": 1}\n```"
    assert server._clean_json(text) == '{"test": 1}'

    text2 = "```\n[1, 2]\n```"
    assert server._clean_json(text2) == "[1, 2]"
    
    text3 = "Some conversational text... {\"key\": \"value\"} ...more chat"
    assert server._clean_json(text3) == '{"key": "value"}'

    text4 = "No json here"
    assert server._clean_json(text4) == "No json here"

    # Test arrays and objects overlapping correctly
    text5 = "Here is an array: [{\"a\": 1}]"
    assert server._clean_json(text5) == '[{"a": 1}]'


@patch("urllib.request.urlopen")
def test_call_fast_autofixer(mock_urlopen):
    state = {"test": 1}
    questions = [{"key": "q1", "prompt": "bad prompt", "type": "noul"}]
    errors = [{"code": "ERR"}]

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"fixed_questions": [{"key": "q1", "prompt": "good prompt", "type": "noul"}]}'
            }
        }]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context

    res = server.call_fast_autofixer(state, questions, errors)
    assert len(res) == 1
    assert res[0]["prompt"] == "good prompt"


@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_success(mock_lint, mock_check_limit, mock_evaluate_batch, mock_urlopen):
    # Setup linting to return no errors
    mock_report = MagicMock()
    mock_report.findings = []
    mock_lint.return_value = mock_report
    mock_check_limit.return_value = 100
    
    # Model based linting returns small noul probabilities (no generative intent)
    # The actual evaluation returns actual evaluation results
    mock_evaluate_batch.side_effect = [
        {"q1": {"noul": 0.1}}, # for linter
        {"q1": {"noul": 0.9}}  # actual eval
    ]

    q1 = NoulQuestion(key="q1", prompt="Test?")
    result_str = server.jev_evaluate_batch(state={"test": 1}, questions=[q1])
    
    result = json.loads(result_str)
    assert result["status"] == "SUCCESS"
    assert result["data"]["q1"]["noul"] == 0.9

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_qfe_compression(mock_lint, mock_check_limit, mock_evaluate_batch, mock_urlopen):
    # Setup linting to return no errors
    mock_report = MagicMock()
    mock_report.findings = []
    mock_lint.return_value = mock_report
    
    # Initial state too big, then after compression it's small enough
    mock_check_limit.side_effect = [10000, 100]

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "Compressed state text"
            }
        }]
    }).encode("utf-8")
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager

    # Linter evaluates, then actual evaluates
    mock_evaluate_batch.side_effect = [
        {"q1": {"noul": 0.1}}, # linter
        {"q1": {"noul": 0.9}}  # actual
    ]

    q1 = NoulQuestion(key="q1", prompt="Test?")
    result_str = server.jev_evaluate_batch(state={"test": "a"*40000}, questions=[q1])

    result = json.loads(result_str)
    assert result["status"] == "SUCCESS"
    assert result.get("was_compressed") is True
    assert result["qfe_state"]["qfe_extracted_context"] == "Compressed state text"

@patch("urllib.request.urlopen")
@patch.object(server.provider, "check_token_limit")
def test_jev_evaluate_batch_qfe_compression_fails(mock_check_limit, mock_urlopen):
    mock_check_limit.side_effect = [10000, 10000] # Still too big after compression
    
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "Compressed state text"
            }
        }]
    }).encode("utf-8")
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager

    q1 = NoulQuestion(key="q1", prompt="Test?")
    result_str = server.jev_evaluate_batch(state={"test": "a"*40000}, questions=[q1])

    assert "CRITICAL SYSTEM ERROR" in result_str
    assert "QFE output still exceeded max tokens." in result_str

@patch("urllib.request.urlopen")
@patch.object(server.provider, "check_token_limit")
def test_jev_evaluate_batch_qfe_compression_exception(mock_check_limit, mock_urlopen):
    mock_check_limit.return_value = 10000
    mock_urlopen.side_effect = Exception("Network Error")

    q1 = NoulQuestion(key="q1", prompt="Test?")
    result_str = server.jev_evaluate_batch(state={"test": "a"*40000}, questions=[q1])

    assert "CRITICAL SYSTEM ERROR" in result_str
    assert "Network Error" in result_str

@patch("jev_mcp.server.call_fast_autofixer")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_model_linter_generative(mock_lint, mock_check_limit, mock_evaluate_batch, mock_autofix):
    mock_report = MagicMock()
    mock_report.findings = []
    mock_lint.return_value = mock_report
    mock_check_limit.return_value = 100
    
    # Model based linter thinks it's generative (>0.85)
    # The first call to evaluate_batch is for lint_qs (key q_0)
    mock_evaluate_batch.return_value = {"q_0": {"noul": 0.99}}
    
    mock_autofix.side_effect = Exception("Autofixer failed")

    q1 = NoulQuestion(key="q1", prompt="Summarize this text?")
    result_str = server.jev_evaluate_batch(state={"test": 1}, questions=[q1], auto_apply_fixes=False)
    
    result = json.loads(result_str)
    assert result["status"] == "REJECTED_BY_LINTER"
    assert any(f["code"] == "SEMANTIC_GENERATIVE_INTENT" for f in result["original_findings"])

@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_static_linter_error(mock_lint, mock_check_limit, mock_evaluate_batch):
    mock_report = MagicMock()
    mock_report.findings = [
        LintFinding(
            severity="ERROR",
            code="SOME_ERROR",
            message="Error msg",
            suggestion="Fix it"
        )
    ]
    mock_lint.return_value = mock_report
    mock_check_limit.return_value = 100
    
    mock_evaluate_batch.return_value = {"q1": {"noul": 0.1}}

    q1 = ChoiceQuestion(key="q1", prompt="Pick one?", options=["a"])
    q2 = ScoreQuestion(key="q2", prompt="Score?", labels=["1", "2"])

    result_str = server.jev_evaluate_batch(state={"test": 1}, questions=[q1, q2])
    
    result = json.loads(result_str)
    assert result["status"] == "REJECTED_BY_LINTER"
    assert any(f["code"] == "SOME_ERROR" for f in result["original_findings"])

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_autofix_requires_approval(mock_lint, mock_check_limit, mock_evaluate_batch, mock_urlopen):
    # Static linter fails initially, then passes
    mock_report_fail = MagicMock()
    mock_report_fail.findings = [
        LintFinding(severity="ERROR", code="SOME_ERROR", message="msg", suggestion="sug")
    ]
    mock_report_pass = MagicMock()
    mock_report_pass.findings = []
    
    mock_lint.side_effect = [mock_report_fail, mock_report_pass]
    mock_check_limit.return_value = 100
    
    # Eval batch for linter
    mock_evaluate_batch.return_value = {"q1": {"noul": 0.1}}

    # Autofixer response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"fixed_questions": [{"key": "q1", "prompt": "fixed prompt", "type": "noul"}]}'
            }
        }]
    }).encode("utf-8")
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager

    q1 = NoulQuestion(key="q1", prompt="bad prompt?")
    result_str = server.jev_evaluate_batch(state={"test": 1}, questions=[q1], auto_apply_fixes=False)

    result = json.loads(result_str)
    assert result["status"] == "REQUIRES_APPROVAL"
    assert len(result["suggested_fixes"]) == 1
    assert result["suggested_fixes"][0]["prompt"] == "fixed prompt"

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_autofix_auto_apply(mock_lint, mock_check_limit, mock_evaluate_batch, mock_urlopen):
    # Static linter fails initially, then passes
    mock_report_fail = MagicMock()
    mock_report_fail.findings = [
        LintFinding(severity="ERROR", code="SOME_ERROR", message="msg", suggestion="sug")
    ]
    mock_report_pass = MagicMock()
    mock_report_pass.findings = []
    
    mock_lint.side_effect = [mock_report_fail, mock_report_fail, mock_report_pass, mock_report_pass]
    mock_check_limit.return_value = 100
    
    # Eval batch for linter, then eval batch for autofixed
    mock_evaluate_batch.side_effect = [
        {"q1": {"noul": 0.1}}, # linter fail loop
        {"q1": {"noul": 0.1}}, # linter pass loop
        {"q1": {"noul": 0.8}}  # actual eval
    ]

    # Autofixer response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"fixed_questions": [{"key": "c1", "prompt": "fixed choice", "type": "choice", "options": ["X"]}, {"key": "s1", "prompt": "fixed score", "type": "score", "labels": ["Y"]}]}'
            }
        }]
    }).encode("utf-8")
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context_manager

    q_c = ChoiceQuestion(key="c1", prompt="bad choice?", options=["A"])
    q_s = ScoreQuestion(key="s1", prompt="bad score?", labels=["1"])

    result_str = server.jev_evaluate_batch(state={"test": 1}, questions=[q_c, q_s], auto_apply_fixes=True)

    result = json.loads(result_str)
    assert result["status"] == "SUCCESS"
    assert result["was_autofixed"] is True
    assert len(result["autofix_transformations"]) == 2

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
@patch.object(server.provider, "check_token_limit")
@patch.object(server.linter, "lint")
def test_jev_evaluate_batch_engine_failure(mock_lint, mock_check_limit, mock_evaluate_batch, mock_urlopen):
    mock_report = MagicMock()
    mock_report.findings = []
    mock_lint.return_value = mock_report
    mock_check_limit.return_value = 100
    
    mock_evaluate_batch.side_effect = [
        {"q1": {"noul": 0.1}}, # linter loop
        Exception("Eval exploded")
    ]

    q1 = NoulQuestion(key="q1", prompt="Test?")
    result_str = server.jev_evaluate_batch(state={"test": 1}, questions=[q1])

    assert "INTERNAL ENGINE ERROR: Eval exploded" in result_str

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
def test_jev_optimize_prompt_success(mock_evaluate_batch, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"variations": ["var1", "var2"]}'
            }
        }]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    # Original (idx 0), var1, var2
    mock_evaluate_batch.return_value = {
        "var_0": {"noul": 0.5},
        "var_1": {"noul": 0.95}, # Best
        "var_2": {"noul": 0.3}
    }

    q1 = NoulQuestion(key="q1", prompt="Original prompt?")
    res = server.jev_optimize_prompt(state={"a": 1}, question=q1)
    
    res_data = json.loads(res)
    assert res_data["optimized_prompt"] == "var1"
    assert "95.00%" in res_data["optimized_confidence"]
    assert "50.00%" in res_data["original_confidence"]

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
def test_jev_optimize_prompt_choice(mock_evaluate_batch, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"variations": ["var1"]}'
            }
        }]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    mock_evaluate_batch.return_value = {
        "var_0": {"probabilities": {"A": 0.6, "B": 0.4}},
        "var_1": {"probabilities": {"A": 0.9, "B": 0.1}}, # Best
    }

    q1 = ChoiceQuestion(key="c1", prompt="Original prompt?", options=["A", "B"])
    res = server.jev_optimize_prompt(state={"a": 1}, question=q1)
    
    res_data = json.loads(res)
    assert res_data["optimized_prompt"] == "var1"

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
def test_jev_optimize_prompt_score(mock_evaluate_batch, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"variations": "var1"}' # tests string fallback
            }
        }]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    mock_evaluate_batch.return_value = {
        "var_0": {"confidence": 0.2},
        "var_1": {"confidence": 0.8}, # Best
    }

    q1 = ScoreQuestion(key="s1", prompt="Original prompt?", labels=["1", "2"])
    res = server.jev_optimize_prompt(state={"a": 1}, question=q1)
    
    res_data = json.loads(res)
    assert res_data["optimized_prompt"] == "var1"

@patch("urllib.request.urlopen")
def test_jev_optimize_prompt_gen_fail(mock_urlopen):
    mock_urlopen.side_effect = Exception("HTTP 500")
    q1 = NoulQuestion(key="q1", prompt="Original prompt?")
    res = server.jev_optimize_prompt(state={"a": 1}, question=q1)
    assert "Optimizer Generator Failed" in res

@patch("urllib.request.urlopen")
@patch.object(server.provider, "evaluate_batch")
def test_jev_optimize_prompt_eval_fail(mock_evaluate_batch, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"variations": []}'}}]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    mock_evaluate_batch.side_effect = Exception("Eval Error")
    
    q1 = NoulQuestion(key="q1", prompt="Original prompt?")
    res = server.jev_optimize_prompt(state={"a": 1}, question=q1)
    assert "Optimizer Evaluation Failed" in res

@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_noul(mock_eval_dataset):
    mock_eval_dataset.return_value = [
        {"noul": 0.9}, # tp at >0.5
        {"noul": 0.8}, # fp
        {"noul": 0.2}, # tn
        {"noul": 0.1}  # fn
    ]
    dataset = [
        {"expected": True},
        {"expected": False},
        {"expected": False},
        {"expected": True}
    ]
    q = NoulQuestion(key="q1", prompt="?")
    res_str = server.jev_calibrate_threshold(dataset, q)
    res = json.loads(res_str)
    assert res["status"] == "CALIBRATION_COMPLETE"
    assert "Markdown" or "markdown" in res["markdown_report"].lower()
    
@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_choice(mock_eval_dataset):
    mock_eval_dataset.return_value = [
        {"probabilities": {"a": 0.9, "b": 0.1}}, # tp
        {"probabilities": {"a": 0.8, "b": 0.2}}, # fp
        {"probabilities": {"b": 0.9, "a": 0.1}}, # tn
        {"probabilities": {"b": 0.8, "a": 0.2}}  # fn
    ]
    dataset = [
        {"expected": "A"},
        {"expected": "B"},
        {"expected": "B"},
        {"expected": "A"}
    ]
    q = ChoiceQuestion(key="c1", prompt="?", options=["A", "B"])
    res_str = server.jev_calibrate_threshold(dataset, q)
    res = json.loads(res_str)
    assert res["status"] == "CALIBRATION_COMPLETE"

@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_fail(mock_eval_dataset):
    mock_eval_dataset.side_effect = Exception("Calibration error")
    dataset = [{"expected": True}]
    q = NoulQuestion(key="q1", prompt="?")
    res = server.jev_calibrate_threshold(dataset, q)
    assert "Calibration Failed" in res

@patch("urllib.request.urlopen")
def test_jev_explain_decision_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"evidence_sentence": "The user clicked yes.", "reasoning": "Clear evidence"}'
            }
        }]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    q = NoulQuestion(key="q1", prompt="Did they click?")
    res_str = server.jev_explain_decision(state={"text": "The user clicked yes."}, question=q, decision="True")
    res = json.loads(res_str)
    assert res["evidence"] == "The user clicked yes."

@patch("urllib.request.urlopen")
def test_jev_explain_decision_fail(mock_urlopen):
    mock_urlopen.side_effect = Exception("HTTP 500")
    q = NoulQuestion(key="q1", prompt="Did they click?")
    res = server.jev_explain_decision(state={"text": "xyz"}, question=q, decision="True")
    assert "Evidence Extraction Failed" in res

@patch("subprocess.check_output")
@patch("urllib.request.urlopen")
def test_jev_generate_synthetic_dataset_local_success(mock_urlopen, mock_check_output):
    mock_check_output.return_value = b"1234 mlx_lm.server\n"
    
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": '{"dataset": [{"state": {"text": "A"}, "expected": true}]}'
            }
        }]
    }).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_context
    
    q = NoulQuestion(key="q1", prompt="?")
    res_str = server.jev_generate_synthetic_dataset(question=q, num_cases=1)
    res = json.loads(res_str)
    assert res["status"] == "DATASET_GENERATED"
    assert len(res["dataset"]) == 1

@patch("subprocess.check_output")
@patch("urllib.request.urlopen")
def test_jev_generate_synthetic_dataset_local_fail_fallback(mock_urlopen, mock_check_output):
    mock_check_output.return_value = b"1234 mlx_lm.server\n"
    mock_urlopen.side_effect = Exception("Server error")
    
    q = NoulQuestion(key="q1", prompt="Some prompt")
    res = server.jev_generate_synthetic_dataset(question=q, num_cases=1)
    
    # Falls back to prompt
    assert "INSTRUCTION TO PRIMARY LLM:" in res

@patch("subprocess.check_output")
def test_jev_generate_synthetic_dataset_no_local(mock_check_output):
    mock_check_output.side_effect = Exception("pgrep failed")
    
    q = ChoiceQuestion(key="q1", prompt="Some prompt", options=["A", "B"])
    res = server.jev_generate_synthetic_dataset(question=q, num_cases=2)
    
    assert "INSTRUCTION TO PRIMARY LLM:" in res
    assert "one of the choice options" in res

@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_platt_auto_triggered(mock_eval_dataset):
    # Simulate overconfident model: 10 items, all >0.99 logprobs
    # 7 heads, 3 tails expected
    mock_eval_dataset.return_value = [
        {"probabilities": {"a": 0.999, "b": 0.001}} for _ in range(10)
    ]
    dataset = [{"expected": "A"} for _ in range(7)] + [{"expected": "B"} for _ in range(3)]
    
    q = ChoiceQuestion(key="coin", prompt="?", options=["A", "B"])
    res_str = server.jev_calibrate_threshold(dataset, q, apply_platt_scaling="auto")
    res = json.loads(res_str)
    
    assert res["status"] == "CALIBRATION_COMPLETE"
    # Should say Platt Scaling Applied
    assert "Platt Scaling Applied" in res["markdown_report"]
    # The >0.99 threshold should now have 0 automation rate because probabilities were squished
    assert "| > 0.99 | 0.0% |" in res["markdown_report"]

@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_platt_always(mock_eval_dataset):
    # Not inherently overconfident (only 2 items)
    mock_eval_dataset.return_value = [
        {"probabilities": {"a": 0.6, "b": 0.4}},
        {"probabilities": {"a": 0.4, "b": 0.6}},
    ]
    dataset = [{"expected": "A"}, {"expected": "B"}]
    
    q = ChoiceQuestion(key="coin", prompt="?", options=["A", "B"])
    # Force apply_platt_scaling
    res_str = server.jev_calibrate_threshold(dataset, q, apply_platt_scaling="always")
    res = json.loads(res_str)
    
    assert "Platt Scaling Applied" in res["markdown_report"]

@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_platt_never(mock_eval_dataset):
    # Overconfident
    mock_eval_dataset.return_value = [
        {"probabilities": {"a": 0.999, "b": 0.001}} for _ in range(10)
    ]
    dataset = [{"expected": "A"} for _ in range(7)] + [{"expected": "B"} for _ in range(3)]
    
    q = ChoiceQuestion(key="coin", prompt="?", options=["A", "B"])
    res_str = server.jev_calibrate_threshold(dataset, q, apply_platt_scaling="never")
    res = json.loads(res_str)
    
    # Should detect it but suggest the user turn it on
    assert "To fix this automatically, set `apply_platt_scaling='auto'`" in res["markdown_report"]
    assert "Platt Scaling Applied" not in res["markdown_report"]

@patch.object(server.provider, "evaluate_dataset")
def test_jev_calibrate_threshold_platt_missing_sklearn(mock_eval_dataset):
    # Overconfident
    mock_eval_dataset.return_value = [
        {"probabilities": {"a": 0.999, "b": 0.001}} for _ in range(10)
    ]
    dataset = [{"expected": "A"} for _ in range(7)] + [{"expected": "B"} for _ in range(3)]
    
    q = ChoiceQuestion(key="coin", prompt="?", options=["A", "B"])
    
    # Mock ImportError
    with patch.dict('sys.modules', {'sklearn.linear_model': None}):
        res_str = server.jev_calibrate_threshold(dataset, q, apply_platt_scaling="auto")
        res = json.loads(res_str)
        
        assert "scikit-learn" in res["markdown_report"]

@patch("jev_mcp.server.RoutingProvider")
def test_jev_determine_best_model(mock_provider_class):
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {
        "complexity": {
            "score": "3",
            "probabilities": {"1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1}
        }
    }
    
    res_str = server.jev_determine_best_model("Implement a distributed queue.", estimated_tokens=0)
    res = json.loads(res_str)
    assert res["status"] in ["EVALUATED", "COMPLEXITY_EVALUATED"]
    assert res["complexity_score"] == 3
    assert res["recommended_model"] == "claude-3-opus"

@patch("jev_mcp.server.RoutingProvider")
def test_jev_agent_handoff(mock_provider_class):
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {
        "best_agent": {
            "choice": "Python_Agent",
            "probabilities": {"Python_Agent": 0.8, "DB_Agent": 0.2}
        }
    }
    
    agents = {"Python_Agent": "Writes code", "DB_Agent": "Writes SQL"}
    res_str = server.jev_agent_handoff("Fix python bug", agents)
    res = json.loads(res_str)
    assert res["status"] == "HANDOFF_RECOMMENDATION"
    assert res["recommended_agent"] == "Python_Agent"
    
def test_jev_train_lora_rejects_missing_dataset():
    res = json.loads(server.jev_train_lora("/nonexistent/path/data.jsonl"))
    assert res["status"] == "ERROR"
    assert ".jsonl" in res["message"] or "not" in res["message"].lower()

@patch("shutil.which", return_value="/opt/mlx/bin/mlx_lm.lora")
@patch("subprocess.Popen")
def test_jev_train_lora_starts_training(mock_popen, mock_which, tmp_path):
    mock_proc = MagicMock()
    mock_proc.pid = 4242
    mock_popen.return_value = mock_proc

    ds = tmp_path / "mydata.jsonl"
    ds.write_text('{"messages": []}\n')

    res = json.loads(server.jev_train_lora(str(ds)))

    assert res["status"] == "TRAINING_STARTED"
    assert res["pid"] == 4242
    cmd = mock_popen.call_args[0][0]
    assert "mlx_lm.lora" in cmd
    assert "mlx-community/Qwen2.5-7B-Instruct-4bit" in cmd
    # A train.jsonl link must exist in the run data dir passed via --data
    data_dir = res["data_dir"]
    assert os.path.exists(os.path.join(data_dir, "train.jsonl"))

@patch("jev_mcp.server.post_json")
def test_chat_completion_sends_system_and_json_format(mock_post):
    mock_post.return_value = {"choices": [{"message": {"content": "hello"}}]}
    out = server._chat_completion(
        "user-msg",
        "system-msg",
        temperature=0.5,
        max_tokens=10,
        response_format={"type": "json_object"},
    )
    assert out == "hello"
    url, payload = mock_post.call_args[0][0], mock_post.call_args[0][1]
    assert url.endswith(":8081/v1/chat/completions")
    assert payload["messages"] == [
        {"role": "system", "content": "system-msg"},
        {"role": "user", "content": "user-msg"},
    ]
    assert payload["response_format"] == {"type": "json_object"}

@patch("jev_mcp.server.post_json")
def test_chat_completion_without_system_or_format(mock_post, monkeypatch):
    monkeypatch.setenv("JEV_FAST_PORT", "9999")
    mock_post.return_value = {"choices": [{"message": {"content": "raw text"}}]}
    out = server._chat_completion(
        "user-msg",
        port_env="JEV_FAST_PORT",
        default_port="8080",
        temperature=0.0,
        max_tokens=200,
        timeout=120,
    )
    assert out == "raw text"
    url, payload = mock_post.call_args[0][0], mock_post.call_args[0][1]
    assert url.endswith(":9999/v1/chat/completions")
    assert payload["messages"] == [{"role": "user", "content": "user-msg"}]
    assert "response_format" not in payload

@patch("jev_mcp.server.load_router_config")
@patch("jev_mcp.server.save_router_config")
def test_jev_manage_router_config(mock_save, mock_load):
    mock_load.return_value = {"custom_rules": [], "buckets": {}}
    
    # View
    res_str = server.jev_manage_router_config("view_all")
    res = json.loads(res_str)
    assert "custom_rules" in res
    
    # Update
    server.jev_manage_router_config("set_bucket", bucket_id="b1", model_name="gpt-4")
    mock_save.assert_called_once()

@patch("jev_mcp.server.load_router_config")
@patch("jev_mcp.server.save_router_config")
def test_jev_manage_temperature(mock_save, mock_load):
    mock_load.return_value = {"fitted_temperature": 1.5, "temperature_updated_at": "Today"}
    
    # View
    res_str = server.jev_manage_temperature("view")
    res = json.loads(res_str)
    assert res["current_temperature"] == 1.5
    
    # Set
    server.jev_manage_temperature("set", 1.8)
    saved_cfg = mock_save.call_args[0][0]
    assert saved_cfg["fitted_temperature"] == 1.8
    assert "temperature_updated_at" in saved_cfg
    
    # Reset
    server.jev_manage_temperature("reset")
    saved_cfg_reset = mock_save.call_args[0][0]
    assert saved_cfg_reset["fitted_temperature"] == 1.0

@patch("jev_mcp.server.load_router_config")
@patch("jev_mcp.server.RoutingProvider")
def test_system_notice_injection(mock_provider_class, mock_load):
    # Reset global state for test
    server._has_notified_temp = False
    
    # Mock config to have custom temp
    mock_load.return_value = {"fitted_temperature": 1.34, "temperature_updated_at": "Just now", "rules": [], "buckets": {}}
    
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {
        "complexity": {"score": "1", "probabilities": {}}
    }
    
    # First call should inject notice
    res_str = server.jev_determine_best_model("task", estimated_tokens=0)
    res = json.loads(res_str)
    assert "_system_notice" in res
    assert "1.34" in res["_system_notice"]
    
    # Second call should NOT inject notice (flag is True)
    res_str2 = server.jev_determine_best_model("task2", estimated_tokens=0)
    res2 = json.loads(res_str2)
    assert "_system_notice" not in res2

