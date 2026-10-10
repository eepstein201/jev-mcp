def build_hitl_message(email_data: dict, decision_data: dict) -> dict:
    """Builds an interactive Slack Block Kit message for Human-in-the-Loop review."""
    subject = email_data.get("subject", "No Subject")
    sender = email_data.get("sender", "Unknown Sender")
    
    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Review Required:* {subject}\nFrom: {sender}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Archive"},
                        "value": f"archive_{email_data.get('id')}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Mark Important"},
                        "value": f"important_{email_data.get('id')}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Draft Reply"},
                        "value": f"reply_{email_data.get('id')}"
                    }
                ]
            }
        ]
    }

def build_escalation_message(email_data: dict, decision_data: dict) -> dict:
    """Builds an urgent Slack Block Kit message for high-priority Action items."""
    subject = email_data.get("subject", "No Subject")
    sender = email_data.get("sender", "Unknown Sender")
    
    return {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 URGENT ACTION REQUIRED"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{subject}*\nFrom: {sender}"
                }
            }
        ]
    }
