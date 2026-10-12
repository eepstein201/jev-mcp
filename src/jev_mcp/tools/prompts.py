"""The optimize-prompt tool."""

import json
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field

from jev_mcp import app
from jev_mcp.app import logger, mcp
from jev_mcp.provider import ChoiceQuestion, NoulQuestion, QuestionType, ScoreQuestion


@mcp.tool(
    name="optimize-prompt",
    description="Automated Prompt Engineer: Generates semantic variations of a prompt and scores them. With `dataset` (rows of state + expected) it returns the phrasing with the highest measured accuracy. With only a single `state` it can only return the most confident phrasing, which is not the same as the most accurate.",
)
def jev_optimize_prompt(
    question: Annotated[
        QuestionType, Field(description="The draft question to optimize.")
    ],
    state: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description="One sample context state. Used only when no dataset is given; selection is then by confidence."
        ),
    ] = None,
    dataset: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(
            description='Labelled rows, each {"state": ..., "expected": ...}: true/false for a noul question, the option for a choice question, the label for a score question. Selection is then by accuracy.'
        ),
    ] = None,
) -> str:
    import json
    from jev_mcp import prompt_selection
    from jev_mcp.security import sanitize_payload

    if dataset is not None:
        rows_ok = (
            isinstance(dataset, list)
            and bool(dataset)
            and all(
                isinstance(row, dict) and "state" in row and "expected" in row
                for row in dataset
            )
        )
        if not rows_ok:
            return json.dumps(
                {
                    "status": "ERROR",
                    "message": 'dataset must be a non-empty list of rows, each with "state" and "expected".',
                }
            )
    elif state is None:
        return json.dumps(
            {
                "status": "ERROR",
                "message": "Provide a dataset of labelled rows (preferred) or a single state.",
            }
        )

    logger.info("Starting Prompt Optimization...")
    q_dict: Dict[str, Any] = {}
    if hasattr(question, "type"):
        if question.type == "noul":
            q_dict = {"type": "noul", "key": question.key, "prompt": question.prompt}
        elif question.type == "choice":
            q_dict = {
                "type": "choice",
                "key": question.key,
                "prompt": question.prompt,
                "options": getattr(question, "options", []),
            }
        elif question.type == "score":
            q_dict = {
                "type": "score",
                "key": question.key,
                "prompt": question.prompt,
                "labels": getattr(question, "labels", []),
            }
    else:
        q_dict = question.dict() if hasattr(question, "dict") else vars(question)

    q_dict_str = sanitize_payload(json.dumps(q_dict))

    system_prompt = (
        "You are an expert Prompt Engineer for a Logit-based System-1 decision model. "
        "The user will provide a draft question. Generate exactly 5 semantic variations of the prompt phrasing. "
        "The variations must test different semantic angles (e.g. active vs passive, formal vs informal, broad vs specific) but must NOT change the core intent. "
        "Output ONLY a JSON object.\n"
        'Format: {"variations": ["Variation 1", "Variation 2", ...]}'
    )
    try:
        # Guaranteed JSON via Constrained Decoding
        variations_data = app._chat_completion_json(
            user=f"Optimize this question: {q_dict_str}",
            system=system_prompt,
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        variations = variations_data.get("variations", [])
        if isinstance(variations, str):
            variations = [variations]
        if not isinstance(variations, list):
            variations = []
    except Exception as e:
        return json.dumps(
            {"status": "ERROR", "message": f"Optimizer Generator Failed: {str(e)}"}
        )

    variations.insert(0, q_dict["prompt"])  # include original
    test_questions: List[QuestionType] = []

    for i, var_prompt in enumerate(variations):
        key = f"var_{i}"
        if q_dict["type"] == "noul":
            test_questions.append(NoulQuestion(key=key, prompt=var_prompt))
        elif q_dict["type"] == "choice":
            test_questions.append(
                ChoiceQuestion(key=key, prompt=var_prompt, options=q_dict["options"])
            )
        elif q_dict["type"] == "score":
            test_questions.append(
                ScoreQuestion(key=key, prompt=var_prompt, labels=q_dict["labels"])
            )

    if dataset is not None:
        try:
            row_results = [
                app.provider.evaluate_batch(row["state"], test_questions)
                for row in dataset
            ]
        except Exception as e:
            return json.dumps(
                {"status": "ERROR", "message": f"Optimizer Evaluation Failed: {str(e)}"}
            )
        per_variation = [
            [res.get(q.key, {}) for res in row_results] for q in test_questions
        ]
        stats = prompt_selection.score_variations(
            per_variation, [row["expected"] for row in dataset]
        )
        best = prompt_selection.best_variation(stats)
        gain = (stats[best]["accuracy"] - stats[0]["accuracy"]) * 100
        return json.dumps(
            {
                "selected_by": "accuracy",
                "rows": len(dataset),
                "original_prompt": q_dict["prompt"],
                "original_accuracy": f"{stats[0]['accuracy'] * 100:.2f}%",
                "optimized_prompt": variations[best],
                "optimized_accuracy": f"{stats[best]['accuracy'] * 100:.2f}%",
                "improvement_points": f"{gain:.2f}",
                "recommendation": (
                    "Keep the original: no variation scored better on this dataset."
                    if best == 0
                    else f"Adopt the optimized prompt: it scored better on these {len(dataset)} rows. Confirm on rows it was not selected on."
                ),
                "variations": [
                    {
                        "prompt": variations[i],
                        "accuracy": f"{s['accuracy'] * 100:.2f}%",
                        "mean_correct_probability": round(
                            s["mean_correct_probability"], 4
                        ),
                    }
                    for i, s in enumerate(stats)
                ],
            },
            indent=2,
        )

    try:
        results = app.provider.evaluate_batch(state, test_questions)
    except Exception as e:
        return json.dumps(
            {"status": "ERROR", "message": f"Optimizer Evaluation Failed: {str(e)}"}
        )

    best_var: Dict[str, Any] | None = None
    best_conf = -1.0
    original_conf = 0.0

    for i, res in enumerate(results.values()):
        # Calculate a simple confidence metric (max prob)
        conf: float = 0.0
        if "confidence" in res:
            conf = res["confidence"]
        elif "noul" in res:
            conf = max(res["noul"], 1.0 - res["noul"])
        elif "probabilities" in res:
            conf = max(res["probabilities"].values())

        if i == 0:
            original_conf = conf

        if conf > best_conf:
            best_conf = conf
            best_var = {
                "optimized_prompt": variations[i],
                "optimized_confidence": f"{conf * 100:.2f}%",
                "raw_result": res,
            }

    if not best_var:
        return json.dumps({"error": "No variations evaluated."})
    return json.dumps(
        {
            "original_prompt": q_dict["prompt"],
            "original_confidence": f"{original_conf * 100:.2f}%",
            "optimized_prompt": best_var["optimized_prompt"],
            "optimized_confidence": best_var["optimized_confidence"],
            "improvement_points": f"{(best_conf - original_conf) * 100:.2f}",
            "selected_by": "confidence",
            "recommendation": "Selected by confidence on one state, which is not accuracy: a phrasing can be more confident and wrong. Pass `dataset` (rows of state + expected) to select by measured accuracy.",
            "all_variations_tested": len(variations),
        },
        indent=2,
    )
