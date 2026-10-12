"""Rank prompt phrasings by measured accuracy on labelled rows (used by `jev_optimize_prompt`)."""

from typing import Any, Optional

DECISION_THRESHOLD = 0.5


def _is_true_false(result: dict[str, Any], expected: Any) -> bool:
    return "noul" in result and isinstance(expected, bool)


def predicted_answer(result: dict[str, Any]) -> Optional[str]:
    """The option or label a choice/score result settled on."""
    for key in ("choice", "score"):
        if result.get(key) is not None:
            return str(result[key])
    probabilities: dict[str, float] = result.get("probabilities") or {}
    return (
        str(max(probabilities, key=lambda option: probabilities[option]))
        if probabilities
        else None
    )


def is_correct(result: dict[str, Any], expected: Any) -> bool:
    if _is_true_false(result, expected):
        return (result["noul"] > DECISION_THRESHOLD) == expected
    return predicted_answer(result) == str(expected)


def correct_probability(result: dict[str, Any], expected: Any) -> float:
    """Probability the engine gave to the labelled answer."""
    if _is_true_false(result, expected):
        return float(result["noul"] if expected else 1.0 - result["noul"])
    return float((result.get("probabilities") or {}).get(str(expected), 0.0))


def score_variations(
    per_variation: list[list[dict[str, Any]]], expected: list[Any]
) -> list[dict[str, float]]:
    """per_variation[v][r] is phrasing v's result on row r; returns accuracy stats per phrasing."""
    rows = len(expected)
    return [
        {
            "accuracy": sum(is_correct(res, exp) for res, exp in zip(results, expected))
            / rows,
            "mean_correct_probability": sum(
                correct_probability(res, exp) for res, exp in zip(results, expected)
            )
            / rows,
        }
        for results in per_variation
    ]


def best_variation(stats: list[dict[str, float]]) -> int:
    """Index of the most accurate phrasing; ties go to the earliest, so the original (index 0) wins a draw."""
    best = 0
    for i, s in enumerate(stats):
        if (s["accuracy"], s["mean_correct_probability"]) > (
            stats[best]["accuracy"],
            stats[best]["mean_correct_probability"],
        ):
            best = i
    return best
