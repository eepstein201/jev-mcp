"""The jev_calibrate_threshold and jev_manage_temperature tools."""

import json
import math
from typing import Any, Dict, List, Literal, Optional

from mcp.types import ToolAnnotations
from pydantic import Field

from jev_mcp import app
from jev_mcp.app import PLATT_CV_FOLDS, logger, mcp
from jev_mcp.provider import ChoiceQuestion, NoulQuestion, QuestionType, ScoreQuestion


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

    try:
        results = app.provider.evaluate_dataset(dataset, q_obj)
    except Exception as e:
        return json.dumps(
            {"status": "ERROR", "message": f"Calibration Failed: {str(e)}"}
        )

    def expected_answers(expected: Any) -> set[str]:
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

    def evaluate_thresholds(current_results):
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
        table = "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n"
        table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"

        auto_rate_at_99 = 0.0

        for t in thresholds:
            tp = fp = fn = tn = automated = 0

            for i, res in enumerate(current_results):
                expected = (
                    dataset[i].get("expected") if isinstance(dataset[i], dict) else None
                )

                if isinstance(q_obj, NoulQuestion):
                    is_positive = expected is True or str(expected).lower() == "true"
                    pred_prob = res.get(
                        "noul", res.get("probabilities", {}).get("true", 0.0)
                    )
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

                    is_positive = str(pred_val).lower() in expected_answers(expected)

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

    # Initial evaluation
    initial_table, auto_rate_99 = evaluate_thresholds(results)

    platt_applied = False
    platt_cross_validated = False
    fitted_temperature: float | None = None
    final_table = initial_table

    if apply_platt_scaling == "always" or (
        apply_platt_scaling == "auto" and auto_rate_99 > 0.5
    ):
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
            import numpy as np
            import math

            logger.info("High overconfidence detected. Applying Platt Scaling...")

            X_raw = []
            y_true = []
            valid_indices = []

            for i, res in enumerate(results):
                expected = (
                    dataset[i].get("expected") if isinstance(dataset[i], dict) else None
                )
                probs = res.get("probabilities", {})

                # Currently only implemented for 2-choice questions
                if len(probs) == 2:
                    keys = list(probs.keys())
                    p_a = max(min(probs[keys[0]], 0.999999), 0.000001)
                    p_b = max(min(probs[keys[1]], 0.999999), 0.000001)

                    log_odds = math.log(p_a) - math.log(p_b)

                    # 1 if expected matches keys[0], 0 otherwise
                    is_positive = (
                        1 if str(keys[0]).lower() in expected_answers(expected) else 0
                    )

                    X_raw.append([log_odds])
                    y_true.append(is_positive)
                    valid_indices.append(i)

            if len(X_raw) > 1 and len(set(y_true)) > 1:
                clf = LogisticRegression(
                    C=np.inf, solver="lbfgs"
                )  # C=inf ≡ unregularized (penalty=None is removed in sklearn 1.10)
                X_arr, y_arr = np.array(X_raw), np.array(y_true)
                if min(y_true.count(0), y_true.count(1)) >= PLATT_CV_FOLDS:
                    # Score each row with a model that never saw it, so the table is not optimistic.
                    from sklearn.model_selection import (  # type: ignore[import-untyped]
                        StratifiedKFold,
                        cross_val_predict,
                    )

                    folds = StratifiedKFold(
                        n_splits=PLATT_CV_FOLDS, shuffle=True, random_state=0
                    )
                    calibrated_probs = cross_val_predict(
                        clf, X_arr, y_arr, cv=folds, method="predict_proba"
                    )
                    classes = [0, 1]
                    platt_cross_validated = True
                else:
                    clf.fit(X_arr, y_arr)
                    calibrated_probs = clf.predict_proba(X_arr)
                    classes = list(clf.classes_)

                # The slope of a fit on every row is the temperature correction: scores were
                # produced under the current temperature, so dividing it by the slope composes them.
                slope = float(clf.fit(X_arr, y_arr).coef_[0][0])
                if slope > 0:
                    current_temperature = float(
                        app.load_router_config().get("fitted_temperature", 1.0)
                    )
                    fitted_temperature = round(current_temperature / slope, 3)

                # Update the results inline
                for j, idx in enumerate(valid_indices):
                    keys = list(results[idx]["probabilities"].keys())
                    # Column order follows the class labels (0, 1).
                    # We assigned y=1 if expected matches keys[0], so class 1 is keys[0]
                    idx_1 = classes.index(1)
                    idx_0 = classes.index(0)

                    results[idx]["probabilities"][keys[0]] = calibrated_probs[j][idx_1]
                    results[idx]["probabilities"][keys[1]] = calibrated_probs[j][idx_0]

                platt_applied = True
                final_table, _ = evaluate_thresholds(results)

        except ImportError:
            logger.warning(
                "scikit-learn is required for Platt Scaling. Skipping calibration."
            )
            final_table += "\n### 🚨 Logit Overconfidence Detected\n"
            final_table += "The model exhibits extreme epistemic certainty (>99% confidence), rendering standard thresholding unsafe.\n"
            final_table += "**Run `pip install scikit-learn` to enable automatic Platt Scaling (Logistic Calibration) which will mathematically fix this.**"

    if not platt_applied and auto_rate_99 > 0.5:
        final_table += "\n### 🚨 Logit Overconfidence Detected\n"
        final_table += "The model exhibits extreme epistemic certainty (>99% confidence), rendering standard thresholding unsafe.\n"
        final_table += "To fix this automatically, set `apply_platt_scaling='auto'` to mathematically scale the probabilities down to their true fractional uncertainty."

    if platt_applied:
        final_table += "\n### 🛠️ Platt Scaling Applied\n"
        final_table += "Logit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds."
        if platt_cross_validated:
            final_table += f"\nThe calibrated scores are {PLATT_CV_FOLDS}-fold cross-validated: each row was scored by a model fitted without it."
        else:
            final_table += f"\nThe model was fitted and scored on the same rows (fewer than {PLATT_CV_FOLDS} rows in one class), so this table is optimistic."

    temperature_saved = False
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
        final_table += f"\n\n### 🌡️ Fitted Temperature: {fitted_temperature}\n"
        if temperature_saved:
            final_table += "Saved to the router config. It now applies to every question the smart engine scores; undo it with the `jev_manage_temperature` tool (`action='reset'`)."
        else:
            final_table += "Not saved. It was fitted on this dataset only and would change the scores of every question, so apply it deliberately: re-run with `save_to_config=True`, or use the `jev_manage_temperature` tool."

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
