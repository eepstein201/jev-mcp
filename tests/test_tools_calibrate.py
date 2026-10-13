"""calibrate: scoring of choice questions and JSON error results."""

import json
from unittest.mock import MagicMock, patch

from jev_mcp import server
from jev_mcp.provider import ChoiceQuestion, NoulQuestion


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


# ---- Characterization tests: the entire parsed JSON of each calibrate path, ----
# ---- captured from a real run (real sklearn fit, mocked provider) before the  ----
# ---- function split. They must not change.                                     ----


def _noul_rows(rows):
    """Dataset and mocked provider results for (expected, p_true, p_false) rows."""
    dataset = [
        {"state": {"build": i}, "expected": expected}
        for i, (expected, _, _) in enumerate(rows)
    ]
    results = [
        {
            "expected": expected,
            "noul": p_true,
            "probabilities": {"true": p_true, "false": p_false},
        }
        for expected, p_true, p_false in rows
    ]
    return dataset, results


def _calibrate_provider(results):
    provider = MagicMock()
    provider.evaluate_dataset.return_value = results
    return provider


def test_calibrate_noul_with_cross_validated_platt_scaling_entire_result():
    # 5 True and 5 False rows: enough for the 5-fold cross-validated Platt fit.
    dataset, results = _noul_rows(
        [
            (True, 0.99, 0.01),
            (True, 0.97, 0.03),
            (True, 0.98, 0.02),
            (True, 0.02, 0.98),  # confident mistake
            (True, 0.96, 0.04),
            (False, 0.01, 0.99),
            (False, 0.03, 0.97),
            (False, 0.99, 0.01),  # confident mistake
            (False, 0.02, 0.98),
            (False, 0.04, 0.96),
        ]
    )
    question = NoulQuestion(key="ship", prompt="Should this deploy ship?")

    with patch("jev_mcp.app.provider", _calibrate_provider(results)):
        out = json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset, question=question, apply_platt_scaling="always"
            )
        )

    assert out == {
        "status": "CALIBRATION_COMPLETE",
        "dataset_size": 10,
        "fitted_temperature": 3.085,
        "temperature_saved": False,
        "markdown_report": "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| > 0.50 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.60 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.70 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.80 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.85 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.90 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.95 | 100.0% | 80.0% | 80.0% | 1 | ⚠️ Risky |\n| > 0.99 | 0.0% | 100.0% | 0.0% | 0 | ✅ Safe (Low Volume) |\n\n### 🛠️ Platt Scaling Applied\nLogit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds.\nThe calibrated scores are 5-fold cross-validated: each row was scored by a model fitted without it.\n\n### 🌡️ Fitted Temperature: 3.085\nNot saved. It was fitted on this dataset only and would change the scores of every question, so apply it deliberately: re-run with `save_to_config=True`, or use the `jev_manage_temperature` tool.",
    }


def test_calibrate_dataset_too_small_for_cross_validation_entire_result():
    # 2 True and 2 False rows: the Platt fit is scored on the rows it was fitted on.
    dataset, results = _noul_rows(
        [
            (True, 0.99, 0.01),
            (True, 0.97, 0.03),
            (False, 0.02, 0.98),
            (False, 0.03, 0.97),
        ]
    )
    question = NoulQuestion(key="ship", prompt="Should this deploy ship?")

    with patch("jev_mcp.app.provider", _calibrate_provider(results)):
        out = json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset, question=question, apply_platt_scaling="always"
            )
        )

    assert out == {
        "status": "CALIBRATION_COMPLETE",
        "dataset_size": 4,
        "fitted_temperature": 0.332,
        "temperature_saved": False,
        "markdown_report": "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| > 0.50 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.60 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.70 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.80 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.85 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.90 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.95 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.99 | 0.0% | 100.0% | 0.0% | 0 | ✅ Safe (Low Volume) |\n\n### 🛠️ Platt Scaling Applied\nLogit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds.\nThe model was fitted and scored on the same rows (fewer than 5 rows in one class), so this table is optimistic.\n\n### 🌡️ Fitted Temperature: 0.332\nNot saved. It was fitted on this dataset only and would change the scores of every question, so apply it deliberately: re-run with `save_to_config=True`, or use the `jev_manage_temperature` tool.",
    }


def test_calibrate_save_to_config_entire_result():
    from jev_mcp import router_config

    dataset, results = _noul_rows(
        [
            (True, 0.99, 0.01),
            (True, 0.97, 0.03),
            (False, 0.02, 0.98),
            (False, 0.03, 0.97),
        ]
    )
    question = NoulQuestion(key="ship", prompt="Should this deploy ship?")

    with patch("jev_mcp.app.provider", _calibrate_provider(results)):
        out = json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset,
                question=question,
                apply_platt_scaling="always",
                save_to_config=True,
            )
        )

    assert out == {
        "status": "CALIBRATION_COMPLETE",
        "dataset_size": 4,
        "fitted_temperature": 0.332,
        "temperature_saved": True,
        "markdown_report": "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| > 0.50 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.60 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.70 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.80 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.85 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.90 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.95 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.99 | 0.0% | 100.0% | 0.0% | 0 | ✅ Safe (Low Volume) |\n\n### 🛠️ Platt Scaling Applied\nLogit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds.\nThe model was fitted and scored on the same rows (fewer than 5 rows in one class), so this table is optimistic.\n\n### 🌡️ Fitted Temperature: 0.332\nSaved to the router config. It now applies to every question the smart engine scores; undo it with the `jev_manage_temperature` tool (`action='reset'`).",
    }
    with open(router_config.ROUTER_CONFIG_PATH) as f:
        saved = json.load(f)
    assert saved["fitted_temperature"] == out["fitted_temperature"]
    assert "temperature_updated_at" in saved
