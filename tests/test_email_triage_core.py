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
