import json

import pytest

from jev_mcp.email_triage.slack import (
    REASON_TEXT,
    build_escalation_message,
    build_hitl_message,
)

EMAIL = {"subject": "S", "sender": "a@b.example", "id": "1"}


def test_format_hitl_block_kit():
    """Asserts that a Review decision generates interactive Slack buttons."""
    email_data = {
        "subject": "Need your opinion",
        "sender": "colleague@test.com",
        "id": "123",
    }
    decision = {"decision": "Review", "bucket": "Review"}

    payload = build_hitl_message(email_data, decision)

    assert "blocks" in payload
    # Check for interactive buttons
    blocks_str = str(payload)
    assert "Archive" in blocks_str
    assert "Mark Important" in blocks_str
    assert "Draft Reply" in blocks_str
    assert email_data["subject"] in blocks_str


def test_format_urgent_escalation():
    """Verifies the payload structure for an urgent escalation."""
    email_data = {"subject": "Server Down", "sender": "alert@test.com", "id": "123"}
    decision = {"decision": "Action", "is_important_score": 0.95}

    payload = build_escalation_message(email_data, decision)

    assert "blocks" in payload
    blocks_str = str(payload)
    assert "URGENT ACTION REQUIRED" in blocks_str
    assert email_data["subject"] in blocks_str


def test_slack_messages_escape_mrkdwn_control_characters():
    # Subject and sender come from the email, so they must not become mentions or links.
    email_data = {
        "subject": "<!channel> & <https://evil.test|click>",
        "sender": "x <a@b.test>",
        "id": "1",
    }

    for build in (build_hitl_message, build_escalation_message):
        sections = [
            b for b in build(email_data, {})["blocks"] if b["type"] == "section"
        ]
        text = sections[0]["text"]["text"]
        assert "<" not in text and ">" not in text
        assert "&lt;!channel&gt; &amp; &lt;https://evil.test|click&gt;" in text


def test_the_review_card_says_why_the_email_is_there():
    message = build_hitl_message(EMAIL, {"reason": "important_no_action"})

    assert "It looks important, though nothing is asked of you." in json.dumps(message)


def test_the_escalation_card_says_why_the_email_is_there():
    message = build_escalation_message(EMAIL, {"reason": "requires_action"})

    assert "It asks you to do something." in json.dumps(message)


@pytest.mark.parametrize("reason", sorted(REASON_TEXT))
def test_each_known_reason_adds_one_why_line(reason):
    message = build_hitl_message(EMAIL, {"reason": reason})

    contexts = [b for b in message["blocks"] if b["type"] == "context"]
    assert len(contexts) == 1
    assert contexts[0]["elements"][0]["text"] == f"*Why:* {REASON_TEXT[reason]}"


@pytest.mark.parametrize("build", [build_hitl_message, build_escalation_message])
@pytest.mark.parametrize(
    "decision", [{"reason": "<@everyone> injected"}, {"reason": 7}, {}]
)
def test_an_unknown_reason_adds_no_line_rather_than_raw_text(build, decision):
    message = build(EMAIL, decision)

    assert "injected" not in json.dumps(message)
    assert all(b["type"] != "context" for b in message["blocks"])


def test_the_reason_texts_are_the_agreed_copy():
    assert REASON_TEXT == {
        "requires_action": "It asks you to do something.",
        "ambiguous_action": "It may need an action from you; the model was not sure.",
        "important_no_action": "It looks important, though nothing is asked of you.",
        "low_bucket_confidence": "The model was not sure which label fits.",
        "confident_bucket": "Nothing is asked of you and the label is clear.",
    }
