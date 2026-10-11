from jev_mcp.email_triage.slack import build_hitl_message, build_escalation_message

def test_format_hitl_block_kit():
    """Asserts that a Review decision generates interactive Slack buttons."""
    email_data = {"subject": "Need your opinion", "sender": "colleague@test.com", "id": "123"}
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
    email_data = {"subject": "<!channel> & <https://evil.test|click>", "sender": "x <a@b.test>", "id": "1"}

    for build in (build_hitl_message, build_escalation_message):
        sections = [b for b in build(email_data, {})["blocks"] if b["type"] == "section"]
        text = sections[0]["text"]["text"]
        assert "<" not in text and ">" not in text
        assert "&lt;!channel&gt; &amp; &lt;https://evil.test|click&gt;" in text
