"""calibrate: scoring of choice questions and JSON error results."""

import json
from unittest.mock import MagicMock, patch

from jev_mcp import server
from jev_mcp.provider import ChoiceQuestion


def _row(report: str, threshold: str) -> list[str]:
    line = next(
        line for line in report.splitlines() if line.startswith(f"| > {threshold}")
    )
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_calibrate_counts_a_choice_answer_keyed_by_its_option_label_as_correct():
    # Both tiers key choice probabilities by the option label, not by a letter.
    question = ChoiceQuestion(
        key="animal", prompt="Which animal?", options=["cat", "dog"]
    )
    dataset = [
        {"state": {"t": "meow"}, "expected": "cat"},
        {"state": {"t": "woof"}, "expected": "dog"},
    ]
    results = [
        {"expected": "cat", "probabilities": {"cat": 0.95, "dog": 0.05}},
        {"expected": "dog", "probabilities": {"cat": 0.10, "dog": 0.90}},
    ]
    provider = MagicMock()
    provider.evaluate_dataset.return_value = results

    with patch("jev_mcp.app.provider", provider):
        out = json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset, question=question, apply_platt_scaling="never"
            )
        )

    _, automation, precision, recall, false_positives, _ = _row(
        out["markdown_report"], "0.50"
    )
    assert (automation, precision, recall, false_positives) == (
        "100.0%",
        "100.0%",
        "100.0%",
        "0",
    )


def test_calibrate_reports_an_engine_failure_as_json():
    question = ChoiceQuestion(
        key="animal", prompt="Which animal?", options=["cat", "dog"]
    )
    provider = MagicMock()
    provider.evaluate_dataset.side_effect = RuntimeError("engine down")

    with patch("jev_mcp.app.provider", provider):
        out = json.loads(
            server.jev_calibrate_threshold(
                dataset=[{"state": {}, "expected": "cat"}], question=question
            )
        )

    assert out["status"] == "ERROR"
    assert "engine down" in out["message"]


@patch("jev_mcp.app.load_router_config", return_value={"fitted_temperature": 1.5})
@patch("jev_mcp.app.save_router_config")
def test_temperature_set_rejects_values_that_are_not_positive_finite_numbers(
    mock_save, mock_load
):
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        res = json.loads(server.jev_manage_temperature("set", bad))
        assert res["status"] == "ERROR", bad
    mock_save.assert_not_called()
