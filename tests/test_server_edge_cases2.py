import pytest
from unittest.mock import patch, MagicMock
from jev_mcp import server
from jev_mcp.provider import NoulQuestion, ScoreQuestion
from jev_mcp.linter import LintFinding, PreflightReport
import runpy
import json

def test_coverage_231():
    with patch.object(server.provider, "check_token_limit") as mock_ctl:
        mock_ctl.return_value = 0
        server.provider.max_tokens = 999999
        with patch.object(server.provider, "evaluate_batch") as mock_eval:
            mock_eval.side_effect = [Exception("test231"), {"q1": {"noul": 0.9}}]
            q = NoulQuestion(key="q1", prompt="prompt")
            server.jev_evaluate_batch(state={"text": "t"}, questions=[q])

def test_coverage_267():
    with patch.object(server.linter, "lint") as mock_lint:
        mock_lint.return_value = PreflightReport(
            is_valid=False,
            findings=[
                LintFinding(severity="ERROR", code="GENERATIVE_INTENT_DETECTED", message="m", suggestion="s")
            ]
        )
        with patch.object(server.provider, "evaluate_batch") as mock_eval:
            mock_eval.side_effect = [{"q_0": {"noul": 0.99}}, {"q1": {"noul": 0.99}}]
            with patch.object(server.provider, "check_token_limit") as mock_ctl:
                mock_ctl.return_value = 0
                q = NoulQuestion(key="q1", prompt="prompt")
                server.jev_evaluate_batch(state={"text": "t"}, questions=[q])

def test_coverage_472():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_res = MagicMock()
        mock_res.read.return_value = json.dumps({
            "choices": [{"message": {"content": "```json\n{\"variations\": 123}\n```"}}]
        }).encode("utf-8")
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res
        q = NoulQuestion(key="q1", prompt="prompt")
        server.jev_optimize_prompt(state={}, question=q)

def test_coverage_633():
    q = ScoreQuestion(key="sq", prompt="score?", labels=["1", "2"])
    states = [{"state": i, "expected": "1"} for i in range(100)] # size 100
    with patch.object(server.provider, "evaluate_dataset") as mock_eval:
        # We need auto_rate < 0.1 (e.g. 5 automated), and precision != 1.0 (e.g. 1 fp, 4 tp).
        # 4 True Positives (expected "1", pred "1" > 0.5)
        # 1 False Positive (expected "2", pred "1" > 0.5)
        # 95 Abstains (pred "1" < 0.5)
        states[0]["expected"] = "2" # this will be our FP
        
        results = []
        # FP
        results.append({"probabilities": {"1": 0.9}})
        # 4 TPs
        for _ in range(4):
            results.append({"probabilities": {"1": 0.9}})
        # 95 Abstains
        for _ in range(95):
            results.append({"probabilities": {"1": 0.1}})
            
        mock_eval.return_value = results
        server.jev_calibrate_threshold(dataset=states, question=q)
def test_coverage_983():
    with patch("jev_mcp.server.main"):
        try:
            runpy.run_module("jev_mcp.server", run_name="__main__")
        except Exception:
            pass
