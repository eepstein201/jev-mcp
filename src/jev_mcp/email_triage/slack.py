def _escape(text: object) -> str:
    """Escape Slack's three mrkdwn control characters, so email text cannot become a mention or link."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# The only text the cards show about a decision: a reason that is not a key here adds nothing.
REASON_TEXT: dict[str, str] = {
    "requires_action": "It asks you to do something.",
    "ambiguous_action": "It may need an action from you; the model was not sure.",
    "important_no_action": "It looks important, though nothing is asked of you.",
    "low_bucket_confidence": "The model was not sure which label fits.",
    "confident_bucket": "Nothing is asked of you and the label is clear.",
}


def _why_blocks(decision_data: dict) -> list[dict]:
    reason = decision_data.get("reason")
    if not isinstance(reason, str) or reason not in REASON_TEXT:
        return []
    return [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"*Why:* {REASON_TEXT[reason]}"}],
        }
    ]


DEGRADED_NOTICE = (
    ":warning: *Unconfirmed:* the smart engine failed, so this routing rests on "
    "the fast engine's low-confidence answer. Please check it yourself."
)


def _degraded_blocks(decision_data: dict) -> list[dict]:
    if not decision_data.get("degraded"):
        return []
    return [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": DEGRADED_NOTICE}],
        }
    ]


def build_hitl_message(email_data: dict, decision_data: dict) -> dict:
    """Builds an interactive Slack Block Kit message for Human-in-the-Loop review."""
    subject = _escape(email_data.get("subject", "No Subject"))
    sender = _escape(email_data.get("sender", "Unknown Sender"))

    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Review Required:* {subject}\nFrom: {sender}",
                },
            },
            *_degraded_blocks(decision_data),
            *_why_blocks(decision_data),
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Archive"},
                        "value": f"archive_{email_data.get('id')}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Mark Important"},
                        "value": f"important_{email_data.get('id')}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Draft Reply"},
                        "value": f"reply_{email_data.get('id')}",
                    },
                ],
            },
        ]
    }


def build_escalation_message(email_data: dict, decision_data: dict) -> dict:
    """Builds an urgent Slack Block Kit message for high-priority Action items."""
    subject = _escape(email_data.get("subject", "No Subject"))
    sender = _escape(email_data.get("sender", "Unknown Sender"))

    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚨 URGENT ACTION REQUIRED"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{subject}*\nFrom: {sender}"},
            },
            *_degraded_blocks(decision_data),
            *_why_blocks(decision_data),
        ]
    }
