"""Score answers from the fast tier are mapped from option index to label."""
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import ScoreQuestion


def test_score_answer_ignores_keys_that_are_not_a_valid_label_index():
    question = ScoreQuestion(key="q", prompt="How hard?", labels=["1", "2", "3"])

    parsed = KevProvider()._parse_answer(question, {"probabilities": {"0": 0.2, "-1": 0.5, "abc": 0.3}})

    # "-1" must not wrap round to the last label.
    assert parsed["probabilities"] == {"1": 0.2, "-1": 0.5, "abc": 0.3}
    assert parsed["score"] == "-1"
