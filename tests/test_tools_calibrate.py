"""calibrate: scoring of choice questions and JSON error results."""

import json
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import router_config, server
from jev_mcp.provider import ChoiceQuestion, NoulQuestion

QWEN = "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QWEN_KEY = "qwen2.5-7b-instruct-q4_k_m.gguf"
OTHER = "/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OTHER_KEY = "llama-3.1-8b-instruct-q4_k_m.gguf"


def _calibrate_provider(results, model_id=QWEN, model_size=None):
    """A router whose smart engine answers with results and reports model_id."""
    provider = MagicMock()
    provider.smart_provider.evaluate_dataset.return_value = results
    provider.smart_provider.current_model_id = model_id
    provider.smart_provider.model_size = model_size
    return provider


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
    provider = _calibrate_provider(results)

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
    provider = _calibrate_provider([])
    provider.smart_provider.evaluate_dataset.side_effect = RuntimeError("engine down")

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
    mock_save, mock_load, smart_model
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
        "model": QWEN_KEY,
        "calibrated": False,
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
        "model": QWEN_KEY,
        "calibrated": False,
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
        "model": QWEN_KEY,
        "calibrated": True,
        "dataset_size": 4,
        "fitted_temperature": 0.332,
        "temperature_saved": True,
        "markdown_report": "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| > 0.50 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.60 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.70 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.80 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.85 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.90 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.95 | 100.0% | 100.0% | 100.0% | 0 | ✅ Optimal |\n| > 0.99 | 0.0% | 100.0% | 0.0% | 0 | ✅ Safe (Low Volume) |\n\n### 🛠️ Platt Scaling Applied\nLogit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds.\nThe model was fitted and scored on the same rows (fewer than 5 rows in one class), so this table is optimistic.\n\n### 🌡️ Fitted Temperature: 0.332\nSaved to the router config for `qwen2.5-7b-instruct-q4_k_m.gguf`. It now applies to every question that model scores on the smart engine; undo it with the `jev_manage_temperature` tool (`action='reset'`).",
    }
    with open(router_config.ROUTER_CONFIG_PATH) as f:
        saved = json.load(f)
    assert saved["temperatures"][QWEN_KEY]["temperature"] == out["fitted_temperature"]
    assert "fitted_temperature" not in saved  # the legacy global is not rewritten
    assert "temperature_updated_at" in saved


# ---- Per-model temperature: calibration fits on the smart engine alone. ----


def _overconfident_rows():
    return _noul_rows(
        [(True, 0.99, 0.01)] * 4
        + [(True, 0.02, 0.98)]
        + [(False, 0.01, 0.99)] * 4
        + [(False, 0.99, 0.01)]
    )


def _calibrate(provider, **kwargs):
    dataset, _ = _overconfident_rows()
    kwargs.setdefault("apply_platt_scaling", "always")
    with patch("jev_mcp.app.provider", provider):
        return json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset,
                question=NoulQuestion(key="ship", prompt="Ship it?"),
                **kwargs,
            )
        )


def _provider_for(model_id=QWEN, model_size=None):
    _, results = _overconfident_rows()
    return _calibrate_provider(results, model_id, model_size)


def _stored():
    return router_config.load_router_config()


def test_calibrate_scores_through_the_smart_engine_and_never_the_router():
    # Arrange
    provider = _provider_for()

    # Act
    out = _calibrate(provider)

    # Assert
    assert out["status"] == "CALIBRATION_COMPLETE"
    provider.smart_provider.evaluate_dataset.assert_called_once()
    provider.evaluate_dataset.assert_not_called()


def test_calibrate_names_the_model_and_says_whether_it_was_already_calibrated():
    fresh = _calibrate(_provider_for(OTHER))
    router_config.save_router_config(
        router_config.with_temperature(_stored(), OTHER, 2.0)
    )
    known = _calibrate(_provider_for(OTHER))

    assert (fresh["model"], fresh["calibrated"]) == (OTHER_KEY, False)
    assert (known["model"], known["calibrated"]) == (OTHER_KEY, True)


def test_calibrate_saves_under_the_model_that_answered_with_its_size():
    router_config.save_router_config({"buckets": {}, "rules": []})

    out = _calibrate(
        _provider_for(OTHER, model_size=4_900_000_000), save_to_config=True
    )

    entry = _stored()["temperatures"][OTHER_KEY]
    assert entry["temperature"] == out["fitted_temperature"]
    assert entry["model_size"] == 4_900_000_000
    assert out["calibrated"] is True


def test_calibrate_save_leaves_the_other_models_temperatures_alone():
    router_config.save_router_config(
        router_config.with_temperature({"buckets": {}, "rules": []}, QWEN, 1.7)
    )

    _calibrate(_provider_for(OTHER), save_to_config=True)

    assert _stored()["temperatures"][QWEN_KEY]["temperature"] == 1.7


def test_calibrate_composes_with_the_temperature_of_the_model_that_answered():
    base = _calibrate(_provider_for(OTHER))["fitted_temperature"]
    router_config.save_router_config(
        router_config.with_temperature(_stored(), OTHER, 2.0)
    )

    scaled = _calibrate(_provider_for(OTHER))["fitted_temperature"]

    assert scaled == pytest.approx(2.0 * base, rel=1e-3)


def test_calibrate_ignores_another_models_temperature_when_composing():
    base = _calibrate(_provider_for(OTHER))["fitted_temperature"]
    router_config.save_router_config(
        router_config.with_temperature(_stored(), QWEN, 5.0)
    )

    other = _calibrate(_provider_for(OTHER))["fitted_temperature"]

    assert other == base


def test_calibrate_applies_the_legacy_global_only_to_qwen_when_composing():
    router_config.save_router_config(
        {"buckets": {}, "rules": [], "fitted_temperature": 2.0}
    )
    base = _calibrate(_provider_for(OTHER))["fitted_temperature"]

    qwen = _calibrate(_provider_for(QWEN))["fitted_temperature"]

    assert qwen == pytest.approx(2.0 * base, rel=1e-3)


@pytest.mark.parametrize("reported", ["unknown_model", ""])
def test_calibrate_refuses_to_write_when_the_smart_engine_reports_no_model(reported):
    router_config.save_router_config({"buckets": {}, "rules": []})
    before = _stored()

    out = _calibrate(_provider_for(reported), save_to_config=True)

    assert out["status"] == "ERROR"
    assert "did not report a model" in out["message"]
    assert _stored() == before


def test_calibrate_reports_a_save_failure_without_raising():
    with patch("jev_mcp.app.save_router_config", side_effect=OSError("read-only")):
        out = _calibrate(_provider_for(), save_to_config=True)

    assert out["temperature_saved"] is False
    assert out["calibrated"] is False


def test_calibrate_still_answers_when_the_config_cannot_be_read_back():
    real = router_config.load_router_config()
    reads = [real, OSError("unreadable")]

    def load():
        item = reads.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with patch("jev_mcp.app.load_router_config", side_effect=load):
        out = _calibrate(_provider_for())

    assert out["status"] == "CALIBRATION_COMPLETE"
    assert out["calibrated"] is False


# ---- jev_manage_temperature acts on the smart model that is running. ----


def _manage(action, value=None, model_id=QWEN, model_size=None):
    with patch("jev_mcp.app.provider", _provider_for(model_id, model_size)):
        return json.loads(server.jev_manage_temperature(action, value))


def test_view_reports_the_default_for_a_model_that_was_never_calibrated():
    out = _manage("view", model_id=OTHER)

    assert out["status"] == "SUCCESS"
    assert out["model"] == OTHER_KEY
    assert out["calibrated"] is False
    assert out["current_temperature"] == 1.0
    assert out["updated_at"] == "Never"


def test_view_reports_the_legacy_global_for_qwen_only():
    router_config.save_router_config(
        {
            "buckets": {},
            "rules": [],
            "fitted_temperature": 1.7,
            "temperature_updated_at": "2026-01-01 00:00:00",
        }
    )

    qwen = _manage("view", model_id=QWEN)
    other = _manage("view", model_id=OTHER)

    assert (qwen["current_temperature"], qwen["calibrated"]) == (1.7, True)
    assert qwen["updated_at"] == "2026-01-01 00:00:00"
    assert (other["current_temperature"], other["calibrated"]) == (1.0, False)


def test_set_stores_the_value_for_the_running_model_only():
    router_config.save_router_config(
        router_config.with_temperature({"buckets": {}, "rules": []}, QWEN, 1.7)
    )

    out = _manage("set", 2.5, model_id=OTHER, model_size=123)

    assert out["status"] == "SUCCESS"
    assert out["model"] == OTHER_KEY
    assert out["current_temperature"] == 2.5
    assert out["message"] == "Temperature successfully set to 2.5."
    stored = _stored()["temperatures"]
    assert stored[OTHER_KEY]["temperature"] == 2.5
    assert stored[OTHER_KEY]["model_size"] == 123
    assert stored[QWEN_KEY]["temperature"] == 1.7


def test_reset_puts_the_running_model_back_to_the_default_even_over_the_legacy_global():
    router_config.save_router_config(
        {"buckets": {}, "rules": [], "fitted_temperature": 1.7}
    )

    out = _manage("reset", model_id=QWEN)

    assert out["message"] == "Temperature successfully reset to default (1.0)."
    assert out["current_temperature"] == 1.0
    assert _manage("view", model_id=QWEN)["current_temperature"] == 1.0


@pytest.mark.parametrize("action,value", [("set", 2.0), ("reset", None)])
def test_set_and_reset_error_when_the_smart_engine_reports_no_model(action, value):
    before = _stored()

    out = _manage(action, value, model_id="unknown_model")

    assert out["status"] == "ERROR"
    assert "did not report a model" in out["message"]
    assert _stored() == before


def test_view_still_answers_when_the_smart_engine_reports_no_model():
    router_config.save_router_config(
        router_config.with_temperature(
            {"buckets": {}, "rules": [], "fitted_temperature": 1.7}, OTHER, 0.6
        )
    )

    out = _manage("view", model_id="unknown_model")

    assert out["status"] == "SUCCESS"
    assert out["model"] is None
    assert out["calibrated"] is False
    assert out["current_temperature"] == 1.0
    assert "did not report a model" in out["message"]
    assert out["stored_temperatures"] == {OTHER_KEY: 0.6}
    assert out["legacy_fitted_temperature"] == 1.7


def test_set_without_a_value_is_an_error():
    out = _manage("set")

    assert out["status"] == "ERROR"
    assert "value" in out["message"]


def test_an_unknown_action_is_an_error():
    out = _manage("bogus")

    assert out["status"] == "ERROR"
    assert "bogus" in out["message"]


def test_manage_temperature_reports_a_config_failure_as_json():
    with (
        patch("jev_mcp.app.provider", _provider_for()),
        patch("jev_mcp.app.load_router_config", side_effect=OSError("unreadable")),
    ):
        out = json.loads(server.jev_manage_temperature("view"))

    assert out == {"status": "ERROR", "message": "unreadable"}


# ---- Malformed hand-edited temperatures and blank model ids ----

MALFORMED = {
    "buckets": {},
    "rules": [],
    "fitted_temperature": 1.7,
    "temperatures": None,
}
BROKEN_ENTRY = {
    "buckets": {},
    "rules": [],
    "fitted_temperature": 1.7,
    "temperatures": {QWEN_KEY: {"updated_at": "x"}, OTHER_KEY: "hot"},
}


@pytest.mark.parametrize("config", [MALFORMED, BROKEN_ENTRY], ids=["null", "entries"])
def test_manage_temperature_survives_a_malformed_temperatures_section(config):
    router_config.save_router_config(config)

    qwen = _manage("view", model_id=QWEN)
    other = _manage("view", model_id=OTHER)
    unknown = _manage("view", model_id="unknown_model")
    set_out = _manage("set", 2.0, model_id=OTHER)

    assert (qwen["status"], qwen["current_temperature"]) == ("SUCCESS", 1.7)
    assert (other["status"], other["current_temperature"]) == ("SUCCESS", 1.0)
    assert unknown["status"] == "SUCCESS"
    assert unknown["stored_temperatures"] == {}
    assert set_out["status"] == "SUCCESS"
    assert _manage("view", model_id=OTHER)["current_temperature"] == 2.0


@pytest.mark.parametrize("config", [MALFORMED, BROKEN_ENTRY], ids=["null", "entries"])
def test_calibrate_survives_a_malformed_temperatures_section(config):
    router_config.save_router_config(config)

    out = _calibrate(_provider_for(OTHER), save_to_config=True)

    assert out["status"] == "CALIBRATION_COMPLETE"
    assert out["temperature_saved"] is True
    assert out["calibrated"] is True


@pytest.mark.parametrize("blank", ["   ", "\t", " \n "])
def test_a_whitespace_only_model_id_is_treated_as_no_model(blank):
    router_config.save_router_config({"buckets": {}, "rules": []})
    before = _stored()

    calibrated = _calibrate(_provider_for(blank), save_to_config=True)
    set_out = _manage("set", 2.0, model_id=blank)
    reset_out = _manage("reset", model_id=blank)

    assert calibrated["status"] == "ERROR"
    assert "did not report a model" in calibrated["message"]
    assert set_out["status"] == "ERROR"
    assert reset_out["status"] == "ERROR"
    assert _stored() == before
