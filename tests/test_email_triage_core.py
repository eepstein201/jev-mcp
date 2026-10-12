import pytest
from jev_mcp.email_triage.core import (
    build_triage_questions,
    decide_routing,
    JevMailConfig,
    CategoryConfig,
)


def test_build_payload_matches_jev_mail_prompts():
    """Asserts exact string matching of instructions to prevent drift from proven jev-mail methodology."""
    questions = build_triage_questions()

    assert len(questions) == 3
    assert questions[0].key == "requires_action"
    assert questions[1].key == "is_important"
    assert questions[2].key == "bucket"

    # Check the Noul prompts match the core jev-mail logic
    assert "Does this email require an action from me?" in questions[0].prompt
    assert "Is this email important?" in questions[1].prompt


def test_triage_questions_are_the_shared_provider_question_types():
    # One NoulQuestion/ChoiceQuestion definition in the package, not a second pair in email_triage.
    from jev_mcp import provider
    from jev_mcp.email_triage import core

    questions = build_triage_questions(["Receipts", "Review"])

    assert isinstance(questions[0], provider.NoulQuestion)
    assert isinstance(questions[2], provider.ChoiceQuestion)
    assert questions[2].options == ["Receipts", "Review"]
    assert core.NoulQuestion is provider.NoulQuestion
    assert core.ChoiceQuestion is provider.ChoiceQuestion


def test_decide_action_required():
    """Asserts high requires_action yields Action label and tests importance threshold."""
    config = JevMailConfig(action_threshold=0.8, important_threshold=0.8)

    # High action, high importance
    decision = decide_routing(
        requires_action_score=0.9,
        is_important_score=0.9,
        bucket="Engineering",
        bucket_confidence=0.9,
        config=config,
    )

    assert decision["decision"] == "Action"
    assert decision["suggested_routing"] == "escalate_to_slack"


def test_decide_fallback_to_review():
    """Asserts ambiguous action probabilities route to Review."""
    config = JevMailConfig(action_threshold=0.8, important_threshold=0.8)

    # Ambiguous action score (below threshold)
    decision = decide_routing(
        requires_action_score=0.5,
        is_important_score=0.2,
        bucket="Newsletters",
        bucket_confidence=0.5,
        config=config,
    )

    assert decision["decision"] == "Review"
    assert decision["suggested_routing"] == "hitl_slack_block"


def test_decide_bucket():
    """Asserts low action but high category confidence routes to Bucket."""
    config = JevMailConfig(action_threshold=0.8, important_threshold=0.8)

    # Low action, high bucket confidence
    decision = decide_routing(
        requires_action_score=0.1,
        is_important_score=0.1,
        bucket="Receipts",
        bucket_confidence=0.95,
        config=config,
    )

    assert decision["decision"] == "Receipts"
    assert decision["suggested_routing"] == "archive_and_label"


def test_an_important_email_that_needs_no_action_is_reviewed_not_archived():
    # An FYI from the boss or a critical alert must not be silently filed away.
    decision = decide_routing(
        requires_action_score=0.1,
        is_important_score=0.95,
        bucket="Notifications",
        bucket_confidence=0.95,
    )

    assert decision == {
        "decision": "Review",
        "suggested_routing": "hitl_slack_block",
        "reason": "important_no_action",
    }


def test_importance_exactly_at_the_threshold_is_not_enough():
    # Strictly above, like the action threshold.
    decision = decide_routing(
        requires_action_score=0.1,
        is_important_score=0.8,
        bucket="Receipts",
        bucket_confidence=0.95,
    )

    assert decision["decision"] == "Receipts"
    assert decision["suggested_routing"] == "archive_and_label"


def test_a_high_action_score_still_escalates_when_the_email_is_also_important():
    decision = decide_routing(
        requires_action_score=0.95,
        is_important_score=0.99,
        bucket="Notifications",
        bucket_confidence=0.95,
    )

    assert decision == {
        "decision": "Action",
        "suggested_routing": "escalate_to_slack",
        "reason": "requires_action",
    }


def test_the_importance_threshold_comes_from_the_config():
    config = JevMailConfig(important_threshold=0.5)

    decision = decide_routing(
        requires_action_score=0.1,
        is_important_score=0.6,
        bucket="Receipts",
        bucket_confidence=0.95,
        config=config,
    )

    assert decision["decision"] == "Review"


def test_an_unimportant_confident_email_is_still_filed_by_bucket():
    decision = decide_routing(
        requires_action_score=0.05,
        is_important_score=0.3,
        bucket="Newsletter",
        bucket_confidence=0.9,
    )

    assert decision == {
        "decision": "Newsletter",
        "suggested_routing": "archive_and_label",
        "reason": "confident_bucket",
    }


@pytest.mark.parametrize(
    "action,important,confidence,reason",
    [
        (0.95, 0.1, 0.9, "requires_action"),
        (0.50, 0.1, 0.9, "ambiguous_action"),
        (0.50, 0.95, 0.9, "ambiguous_action"),  # ambiguity is named before importance
        (0.10, 0.95, 0.9, "important_no_action"),
        (0.10, 0.10, 0.5, "low_bucket_confidence"),
        (0.10, 0.10, 0.9, "confident_bucket"),
        (0.50, 0.95, 0.5, "ambiguous_action"),  # ambiguity wins over low confidence
        (0.10, 0.95, 0.5, "important_no_action"),  # importance wins over low confidence
    ],
)
def test_every_decision_carries_its_reason(action, important, confidence, reason):
    decision = decide_routing(action, important, "Receipts", confidence)

    assert decision["reason"] == reason


# --- thresholds from the environment ---------------------------------------------


def test_thresholds_keep_their_defaults_when_the_environment_is_silent(monkeypatch):
    from jev_mcp.email_triage.core import load_mail_config

    monkeypatch.delenv("JEV_TRIAGE_ACTION_THRESHOLD", raising=False)
    monkeypatch.delenv("JEV_TRIAGE_IMPORTANT_THRESHOLD", raising=False)

    assert load_mail_config() == JevMailConfig()


def test_thresholds_come_from_the_environment(monkeypatch):
    from jev_mcp.email_triage.core import load_mail_config

    monkeypatch.setenv("JEV_TRIAGE_ACTION_THRESHOLD", "0.6")
    monkeypatch.setenv("JEV_TRIAGE_IMPORTANT_THRESHOLD", "0.95")

    config = load_mail_config()

    assert config.action_threshold == 0.6
    assert config.important_threshold == 0.95


@pytest.mark.parametrize("value", ["0", "1", "0.0", "1.0"])
def test_a_threshold_at_either_end_of_the_range_is_accepted(monkeypatch, value):
    from jev_mcp.email_triage.core import load_mail_config

    monkeypatch.setenv("JEV_TRIAGE_IMPORTANT_THRESHOLD", value)

    assert load_mail_config().important_threshold == float(value)


@pytest.mark.parametrize("value", ["abc", "1.5", "-0.1", "nan", "inf", "80%"])
def test_an_unusable_threshold_falls_back_to_the_default_and_says_so(
    monkeypatch, caplog, value
):
    from jev_mcp.email_triage.core import load_mail_config

    monkeypatch.setenv("JEV_TRIAGE_IMPORTANT_THRESHOLD", value)

    with caplog.at_level("WARNING"):
        config = load_mail_config()

    assert config.important_threshold == JevMailConfig().important_threshold
    assert "JEV_TRIAGE_IMPORTANT_THRESHOLD" in caplog.text


def test_a_blank_threshold_is_treated_as_unset_without_a_warning(monkeypatch, caplog):
    from jev_mcp.email_triage.core import load_mail_config

    monkeypatch.setenv("JEV_TRIAGE_ACTION_THRESHOLD", "  ")

    with caplog.at_level("WARNING"):
        config = load_mail_config()

    assert config.action_threshold == JevMailConfig().action_threshold
    assert caplog.text == ""
