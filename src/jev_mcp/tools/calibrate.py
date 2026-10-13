"""The jev_calibrate_threshold and jev_manage_temperature tools."""

import json
import math
from typing import Any, Dict, List, Literal, Optional

from mcp.types import ToolAnnotations
from pydantic import Field

from jev_mcp import app
from jev_mcp.app import PLATT_CV_FOLDS, logger, mcp
from jev_mcp.provider import ChoiceQuestion, NoulQuestion, QuestionType, ScoreQuestion


def _rebuild_question(question: QuestionType) -> QuestionType:
    """A fresh copy of the question carrying only the fields of its own type."""
    q_obj: QuestionType
    if isinstance(question, NoulQuestion):
        q_obj = NoulQuestion(key=question.key, prompt=question.prompt)
    elif isinstance(question, ChoiceQuestion):
        q_obj = ChoiceQuestion(
            key=question.key, prompt=question.prompt, options=question.options
        )
    elif isinstance(question, ScoreQuestion):
        q_obj = ScoreQuestion(
            key=question.key, prompt=question.prompt, labels=question.labels
        )
    return q_obj


def _expected_answers(q_obj: QuestionType, expected: Any) -> set[str]:
    """Keys a correct result may carry: the expected value, or its option letter for a choice."""
    text = str(expected).lower()
    answers = {text}
    if isinstance(q_obj, ChoiceQuestion):
        answers.update(
            chr(97 + idx)
            for idx, opt in enumerate(q_obj.options)
            if str(opt).lower() == text
        )
    return answers


def _confusion_row(dataset, q_obj, results, t):
    """tp, fp, fn, tn and the automated count for one threshold."""
    tp = fp = fn = tn = automated = 0

    for i, res in enumerate(results):
        expected = dataset[i].get("expected") if isinstance(dataset[i], dict) else None

        if isinstance(q_obj, NoulQuestion):
            is_positive = expected is True or str(expected).lower() == "true"
            pred_prob = res.get("noul", res.get("probabilities", {}).get("true", 0.0))
            pred_positive = pred_prob > t

            if pred_prob > t or (1.0 - pred_prob) > t:
                automated += 1

            if is_positive and pred_positive:
                tp += 1
            elif not is_positive and pred_positive:
                fp += 1
            elif is_positive and not pred_positive:
                fn += 1
            elif not is_positive and not pred_positive:
                tn += 1
        else:
            probs = res.get("probabilities", {})
            if probs:
                pred_val = max(probs.items(), key=lambda x: x[1])[0]
                conf = probs[pred_val]
            else:
                pred_val = ""
                conf = 0.0

            is_positive = str(pred_val).lower() in _expected_answers(q_obj, expected)

            if conf > t:
                automated += 1
                if is_positive:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_positive:
                    fn += 1
                else:
                    tn += 1

    return tp, fp, fn, tn, automated


def _evaluate_thresholds(dataset, q_obj, results):
    """The threshold table plus the automation rate at the 0.99 cutoff."""
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    table = "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n"
    table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"

    auto_rate_at_99 = 0.0

    for t in thresholds:
        tp, fp, fn, tn, automated = _confusion_row(dataset, q_obj, results, t)

        total = len(dataset)
        auto_rate = automated / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        rec = "⚠️ Risky"
        if tp == 0 and fn > 0 and auto_rate >= 0.1:
            # Decisions are being automated but no positive is ever caught.
            rec = "❌ Unusable"
        elif precision > 0.98 and auto_rate > 0.4:
            rec = "✅ Optimal"
        elif precision == 1.0:
            rec = "✅ Safe (Low Volume)"
        elif auto_rate < 0.1:
            rec = "❌ Unusable"

        table += f"| > {t:.2f} | {auto_rate * 100:.1f}% | {precision * 100:.1f}% | {recall * 100:.1f}% | {fp} | {rec} |\n"

        if t == 0.99:
            auto_rate_at_99 = auto_rate

    return table, auto_rate_at_99


def _platt_rows(results, dataset, q_obj):
    """The log-odds feature, label and index of every 2-choice row."""
    X_raw = []
    y_true = []
    valid_indices = []

    for i, res in enumerate(results):
        expected = dataset[i].get("expected") if isinstance(dataset[i], dict) else None
        probs = res.get("probabilities", {})

        # Currently only implemented for 2-choice questions
        if len(probs) == 2:
            keys = list(probs.keys())
            p_a = max(min(probs[keys[0]], 0.999999), 0.000001)
            p_b = max(min(probs[keys[1]], 0.999999), 0.000001)

            log_odds = math.log(p_a) - math.log(p_b)

            # 1 if expected matches keys[0], 0 otherwise
            is_positive = (
                1 if str(keys[0]).lower() in _expected_answers(q_obj, expected) else 0
            )

            X_raw.append([log_odds])
            y_true.append(is_positive)
            valid_indices.append(i)

    return X_raw, y_true, valid_indices


def _calibrated_probabilities(clf, X_arr, y_arr, y_true):
    """Cross-validated probabilities when each class has enough rows, else in-sample."""
    if min(y_true.count(0), y_true.count(1)) >= PLATT_CV_FOLDS:
        # Score each row with a model that never saw it, so the table is not optimistic.
        from sklearn.model_selection import (  # type: ignore[import-untyped]
            StratifiedKFold,
            cross_val_predict,
        )

        folds = StratifiedKFold(n_splits=PLATT_CV_FOLDS, shuffle=True, random_state=0)
        calibrated_probs = cross_val_predict(
            clf, X_arr, y_arr, cv=folds, method="predict_proba"
        )
        return calibrated_probs, [0, 1], True

    clf.fit(X_arr, y_arr)
    return clf.predict_proba(X_arr), list(clf.classes_), False


def _apply_calibrated_probabilities(results, valid_indices, calibrated_probs, classes):
    """Overwrite each qualifying row's probabilities with its calibrated pair."""
    for j, idx in enumerate(valid_indices):
        keys = list(results[idx]["probabilities"].keys())
        # Column order follows the class labels (0, 1).
        # We assigned y=1 if expected matches keys[0], so class 1 is keys[0]
        idx_1 = classes.index(1)
        idx_0 = classes.index(0)

        results[idx]["probabilities"][keys[0]] = calibrated_probs[j][idx_1]
        results[idx]["probabilities"][keys[1]] = calibrated_probs[j][idx_0]


def _fit_platt_scaling(results, dataset, q_obj):
    """Fit Platt scaling against the raw log-odds of the dataset and rescore the rows.

    Returns (applied, cross_validated, fitted_temperature, table); the table is
    None when the rows did not qualify for a fit.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    import numpy as np

    logger.info("High overconfidence detected. Applying Platt Scaling...")

    X_raw, y_true, valid_indices = _platt_rows(results, dataset, q_obj)

    platt_applied = False
    platt_cross_validated = False
    fitted_temperature: float | None = None
    final_table = None

    if len(X_raw) > 1 and len(set(y_true)) > 1:
        clf = LogisticRegression(
            C=np.inf, solver="lbfgs"
        )  # C=inf ≡ unregularized (penalty=None is removed in sklearn 1.10)
        X_arr, y_arr = np.array(X_raw), np.array(y_true)
        calibrated_probs, classes, platt_cross_validated = _calibrated_probabilities(
            clf, X_arr, y_arr, y_true
        )

        # The slope of a fit on every row is the temperature correction: scores were
        # produced under the current temperature, so dividing it by the slope composes them.
        slope = float(clf.fit(X_arr, y_arr).coef_[0][0])
        if slope > 0:
            current_temperature = float(
                app.load_router_config().get("fitted_temperature", 1.0)
            )
            fitted_temperature = round(current_temperature / slope, 3)

        # Update the results inline
        _apply_calibrated_probabilities(
            results, valid_indices, calibrated_probs, classes
        )

        platt_applied = True
        final_table, _ = _evaluate_thresholds(dataset, q_obj, results)

    return platt_applied, platt_cross_validated, fitted_temperature, final_table


def _maybe_apply_platt(
    results, dataset, q_obj, apply_platt_scaling, auto_rate_99, final_table
):
    """Run Platt scaling when asked (or, on auto, when the model is overconfident).

    Returns (platt_applied, platt_cross_validated, fitted_temperature, final_table).
    """
    platt_applied = False
    platt_cross_validated = False
    fitted_temperature: float | None = None

    if apply_platt_scaling == "always" or (
        apply_platt_scaling == "auto" and auto_rate_99 > 0.5
    ):
        try:
            (
                platt_applied,
                platt_cross_validated,
                fitted_temperature,
                platt_table,
            ) = _fit_platt_scaling(results, dataset, q_obj)
        except ImportError:
            logger.warning(
                "scikit-learn is required for Platt Scaling. Skipping calibration."
            )
            final_table += "\n### 🚨 Logit Overconfidence Detected\n"
            final_table += "The model exhibits extreme epistemic certainty (>99% confidence), rendering standard thresholding unsafe.\n"
            final_table += "**Run `pip install scikit-learn` to enable automatic Platt Scaling (Logistic Calibration) which will mathematically fix this.**"
        else:
            if platt_table is not None:
                final_table = platt_table

    return platt_applied, platt_cross_validated, fitted_temperature, final_table


def _platt_notes(platt_applied, platt_cross_validated, auto_rate_99):
    """The report section explaining whether Platt scaling ran, and why if it did not."""
    notes = ""
    if not platt_applied and auto_rate_99 > 0.5:
        notes += "\n### 🚨 Logit Overconfidence Detected\n"
        notes += "The model exhibits extreme epistemic certainty (>99% confidence), rendering standard thresholding unsafe.\n"
        notes += "To fix this automatically, set `apply_platt_scaling='auto'` to mathematically scale the probabilities down to their true fractional uncertainty."
    if platt_applied:
        notes += "\n### 🛠️ Platt Scaling Applied\n"
        notes += "Logit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds."
        if platt_cross_validated:
            notes += f"\nThe calibrated scores are {PLATT_CV_FOLDS}-fold cross-validated: each row was scored by a model fitted without it."
        else:
            notes += f"\nThe model was fitted and scored on the same rows (fewer than {PLATT_CV_FOLDS} rows in one class), so this table is optimistic."
    return notes


def _temperature_section(fitted_temperature, save_to_config):
    """The fitted-temperature report section, persisting the temperature first when asked."""
    temperature_saved = False
    section = ""
    if fitted_temperature is not None:
        if save_to_config is True:
            from datetime import datetime

            try:
                app.save_router_config(
                    {
                        **app.load_router_config(),
                        "fitted_temperature": fitted_temperature,
                        "temperature_updated_at": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )
                temperature_saved = True
            except Exception as e:
                logger.error(f"Could not save the fitted temperature: {e}")
        section += f"\n\n### 🌡️ Fitted Temperature: {fitted_temperature}\n"
        if temperature_saved:
            section += "Saved to the router config. It now applies to every question the smart engine scores; undo it with the `jev_manage_temperature` tool (`action='reset'`)."
        else:
            section += "Not saved. It was fitted on this dataset only and would change the scores of every question, so apply it deliberately: re-run with `save_to_config=True`, or use the `jev_manage_temperature` tool."
    return temperature_saved, section


@mcp.tool(
    name="jev_calibrate_threshold",
    title="Calibrate a decision threshold",
    description="Evaluation Sandbox: Pass a dataset of Golden Edge Cases to mathematically map out the optimal Logit Confidence Threshold for production automation.",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
)
def jev_calibrate_threshold(
    dataset: List[Dict[str, Any]] = Field(
        description="Array of dicts: [{'state': {...}, 'expected': True/False/String}]"
    ),
    question: QuestionType = Field(description="The question to calibrate."),
    apply_platt_scaling: Literal["auto", "always", "never"] = Field(
        default="auto",
        description="Whether to apply Platt Scaling (Logistic Calibration) to mathematically fix overconfident >99% logits.",
    ),
    save_to_config: bool = Field(
        default=False,
        description="Save the fitted temperature to ~/.jev/router_config.json. It is global: it changes the scores of every question, not only this one.",
    ),
) -> str:
    logger.info(f"Starting Threshold Calibration for {len(dataset)} rows...")

    try:
        q_obj = _rebuild_question(question)
        results = app.provider.evaluate_dataset(dataset, q_obj)
    except Exception as e:
        return json.dumps(
            {"status": "ERROR", "message": f"Calibration Failed: {str(e)}"}
        )

    # Initial evaluation
    initial_table, auto_rate_99 = _evaluate_thresholds(dataset, q_obj, results)

    platt_applied, platt_cross_validated, fitted_temperature, final_table = (
        _maybe_apply_platt(
            results, dataset, q_obj, apply_platt_scaling, auto_rate_99, initial_table
        )
    )
    final_table += _platt_notes(platt_applied, platt_cross_validated, auto_rate_99)
    temperature_saved, temperature_section = _temperature_section(
        fitted_temperature, save_to_config
    )
    final_table += temperature_section

    return json.dumps(
        {
            "status": "CALIBRATION_COMPLETE",
            "dataset_size": len(dataset),
            "fitted_temperature": fitted_temperature,
            "temperature_saved": temperature_saved,
            "markdown_report": final_table,
        },
        indent=2,
    )


@mcp.tool(
    name="jev_manage_temperature",
    title="Read or set the scoring temperature",
    description="View, set, or reset the global MLX calibration temperature for Jev. This temperature globally affects all Jev evaluations, handoffs, and prompts.",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def jev_manage_temperature(
    action: Literal["view", "set", "reset"], value: Optional[float] = None
) -> str:
    """
    Manages the global fitted_temperature in router_config.json.
    """
    try:
        from datetime import datetime

        config = app.load_router_config()

        if action == "view":
            temp = config.get("fitted_temperature", 1.0)
            updated_at = config.get("temperature_updated_at", "Never")
            return json.dumps(
                {
                    "status": "SUCCESS",
                    "current_temperature": temp,
                    "updated_at": updated_at,
                },
                indent=2,
            )

        elif action == "reset":
            config["fitted_temperature"] = 1.0
            config["temperature_updated_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            app.save_router_config(config)
            return json.dumps(
                {
                    "status": "SUCCESS",
                    "message": "Temperature successfully reset to default (1.0).",
                },
                indent=2,
            )

        elif action == "set":
            if value is None:
                return json.dumps(
                    {
                        "status": "ERROR",
                        "message": "You must provide a 'value' when using action='set'.",
                    }
                )
            # Logits are divided by this, so zero, a negative, NaN or infinity would corrupt every evaluation.
            if not math.isfinite(float(value)) or float(value) <= 0:
                return json.dumps(
                    {
                        "status": "ERROR",
                        "message": "Temperature must be a positive, finite number.",
                    }
                )
            config["fitted_temperature"] = round(float(value), 3)
            config["temperature_updated_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            app.save_router_config(config)
            return json.dumps(
                {
                    "status": "SUCCESS",
                    "message": f"Temperature successfully set to {round(float(value), 3)}.",
                },
                indent=2,
            )

    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})
