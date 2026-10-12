import json
from unittest.mock import patch

import pytest

from jev_mcp import app, prompt_selection, server
from jev_mcp.provider import ChoiceQuestion, NoulQuestion


# --- scoring one answer against its label ---


def test_a_true_false_answer_is_correct_when_it_lands_on_the_labelled_side_of_half():
    assert prompt_selection.is_correct({"noul": 0.8}, True)
    assert prompt_selection.is_correct({"noul": 0.2}, False)
    assert not prompt_selection.is_correct({"noul": 0.8}, False)


def test_a_choice_answer_is_correct_when_the_chosen_option_is_the_label():
    result = {"choice": "Billing", "probabilities": {"Billing": 0.7, "Tech": 0.3}}

    assert prompt_selection.is_correct(result, "Billing")
    assert not prompt_selection.is_correct(result, "Tech")


def test_the_most_probable_option_is_the_answer_when_no_choice_is_reported():
    assert prompt_selection.is_correct({"probabilities": {"A": 0.2, "B": 0.8}}, "B")
    assert not prompt_selection.is_correct({"probabilities": {"A": 0.2, "B": 0.8}}, "A")
    assert not prompt_selection.is_correct({}, "A")


def test_a_score_answer_is_compared_as_text_so_numeric_labels_match():
    assert prompt_selection.is_correct({"score": "2", "probabilities": {"2": 0.9}}, 2)


def test_probability_of_the_correct_answer_for_each_question_type():
    assert prompt_selection.correct_probability({"noul": 0.8}, True) == pytest.approx(
        0.8
    )
    assert prompt_selection.correct_probability({"noul": 0.8}, False) == pytest.approx(
        0.2
    )
    assert prompt_selection.correct_probability(
        {"probabilities": {"A": 0.7, "B": 0.3}}, "B"
    ) == pytest.approx(0.3)
    assert prompt_selection.correct_probability({}, "B") == 0.0


# --- ranking phrasings ---


def test_the_more_accurate_phrasing_wins_even_when_it_is_less_confident():
    # Phrasing 0 is very sure and wrong on the second row; phrasing 1 is hesitant and right on both.
    expected = [True, False]
    per_variation = [
        [{"noul": 0.99}, {"noul": 0.99}],
        [{"noul": 0.60}, {"noul": 0.40}],
    ]

    stats = prompt_selection.score_variations(per_variation, expected)

    assert [s["accuracy"] for s in stats] == [0.5, 1.0]
    assert prompt_selection.best_variation(stats) == 1


def test_equal_accuracy_is_broken_by_probability_given_to_the_correct_answers():
    expected = [True, False]
    per_variation = [
        [{"noul": 0.6}, {"noul": 0.4}],
        [{"noul": 0.9}, {"noul": 0.1}],
    ]

    assert (
        prompt_selection.best_variation(
            prompt_selection.score_variations(per_variation, expected)
        )
        == 1
    )


def test_the_original_is_kept_when_no_phrasing_is_strictly_better():
    expected = [True, False]
    same = [{"noul": 0.9}, {"noul": 0.1}]

    assert (
        prompt_selection.best_variation(
            prompt_selection.score_variations([same, same, same], expected)
        )
        == 0
    )


# --- the tool ---


def _generator_returns(variations):
    return patch.object(
        app, "_chat_completion_json", return_value={"variations": variations}
    )


def _scores(table):
    """evaluate_batch stand-in: table[state] is the list of noul scores, one per phrasing."""

    def evaluate_batch(state, questions):
        return {q.key: {"noul": table[state][i]} for i, q in enumerate(questions)}

    return patch.object(server.provider, "evaluate_batch", side_effect=evaluate_batch)


DATASET = [
    {"state": "reset my password", "expected": True},
    {"state": "I changed my password yesterday", "expected": False},
]


def test_tool_selects_by_accuracy_on_a_labelled_dataset():
    # The original is confidently wrong on the negative row; "v1" is right on both.
    table = {
        "reset my password": [0.99, 0.70],
        "I changed my password yesterday": [0.95, 0.30],
    }
    question = NoulQuestion(key="q", prompt="Is this about passwords?")

    with _generator_returns(["v1"]), _scores(table):
        out = json.loads(server.jev_optimize_prompt(question=question, dataset=DATASET))

    assert out["selected_by"] == "accuracy"
    assert out["rows"] == 2
    assert out["optimized_prompt"] == "v1"
    assert out["original_accuracy"] == "50.00%"
    assert out["optimized_accuracy"] == "100.00%"
    assert [v["prompt"] for v in out["variations"]] == [
        "Is this about passwords?",
        "v1",
    ]


def test_tool_keeps_the_original_when_it_is_already_the_most_accurate():
    table = {
        "reset my password": [0.9, 0.2],
        "I changed my password yesterday": [0.1, 0.9],
    }
    question = NoulQuestion(key="q", prompt="Is this about passwords?")

    with _generator_returns(["v1"]), _scores(table):
        out = json.loads(server.jev_optimize_prompt(question=question, dataset=DATASET))

    assert out["optimized_prompt"] == "Is this about passwords?"
    assert "Keep the original" in out["recommendation"]


def test_tool_scores_choice_questions_against_the_labelled_option():
    dataset = [
        {"state": "refund please", "expected": "Billing"},
        {"state": "app crashes", "expected": "Tech"},
    ]
    question = ChoiceQuestion(
        key="q", prompt="Which team?", options=["Billing", "Tech"]
    )
    answers = {"refund please": ["Tech", "Billing"], "app crashes": ["Tech", "Tech"]}

    def evaluate_batch(state, questions):
        return {
            q.key: {
                "choice": answers[state][i],
                "probabilities": {answers[state][i]: 0.9},
            }
            for i, q in enumerate(questions)
        }

    with (
        _generator_returns(["v1"]),
        patch.object(server.provider, "evaluate_batch", side_effect=evaluate_batch),
    ):
        out = json.loads(server.jev_optimize_prompt(question=question, dataset=dataset))

    assert out["optimized_prompt"] == "v1"
    assert out["optimized_accuracy"] == "100.00%"


@pytest.mark.parametrize(
    "dataset", [[], [{"state": "x"}], [{"expected": True}], "not a list"]
)
def test_tool_rejects_a_dataset_without_state_and_expected_on_every_row(dataset):
    question = NoulQuestion(key="q", prompt="p?")

    with (
        _generator_returns(["v1"]),
        patch.object(server.provider, "evaluate_batch") as evaluate_batch,
    ):
        out = json.loads(server.jev_optimize_prompt(question=question, dataset=dataset))

    assert out["status"] == "ERROR"
    assert "expected" in out["message"]
    evaluate_batch.assert_not_called()


def test_tool_needs_either_a_state_or_a_dataset():
    out = json.loads(
        server.jev_optimize_prompt(question=NoulQuestion(key="q", prompt="p?"))
    )

    assert out["status"] == "ERROR"
    assert "dataset" in out["message"]


def test_tool_says_when_it_could_only_select_by_confidence():
    question = NoulQuestion(key="q", prompt="p?")

    with (
        _generator_returns(["v1"]),
        patch.object(
            server.provider,
            "evaluate_batch",
            return_value={"var_0": {"noul": 0.6}, "var_1": {"noul": 0.9}},
        ),
    ):
        out = json.loads(server.jev_optimize_prompt(state={"a": 1}, question=question))

    assert out["selected_by"] == "confidence"
    assert out["optimized_prompt"] == "v1"
    assert "not accuracy" in out["recommendation"]


def test_tool_returns_an_error_when_scoring_a_dataset_row_fails():
    question = NoulQuestion(key="q", prompt="p?")

    with (
        _generator_returns(["v1"]),
        patch.object(
            server.provider, "evaluate_batch", side_effect=RuntimeError("engine down")
        ),
    ):
        out = server.jev_optimize_prompt(question=question, dataset=DATASET)

    assert "Optimizer Evaluation Failed" in out and "engine down" in out


def test_optimize_prompt_reports_a_generator_failure_as_json():
    from unittest.mock import patch
    from jev_mcp import server
    from jev_mcp.provider import NoulQuestion

    with patch(
        "jev_mcp.app._chat_completion_json", side_effect=RuntimeError("smart tier down")
    ):
        out = json.loads(
            server.jev_optimize_prompt(
                question=NoulQuestion(key="q", prompt="Is it on?"),
                state={"a": 1},
                dataset=None,
            )
        )

    assert out["status"] == "ERROR"
    assert "smart tier down" in out["message"]


def test_optimize_prompt_reports_an_evaluation_failure_as_json():
    from unittest.mock import MagicMock, patch
    from jev_mcp import server
    from jev_mcp.provider import NoulQuestion

    provider = MagicMock()
    provider.evaluate_batch.side_effect = RuntimeError("engine down")
    with (
        patch(
            "jev_mcp.app._chat_completion_json",
            return_value={"variations": ["Is it running?"]},
        ),
        patch("jev_mcp.app.provider", provider),
    ):
        out = json.loads(
            server.jev_optimize_prompt(
                question=NoulQuestion(key="q", prompt="Is it on?"),
                state={"a": 1},
                dataset=None,
            )
        )

    assert out["status"] == "ERROR"
    assert "engine down" in out["message"]
