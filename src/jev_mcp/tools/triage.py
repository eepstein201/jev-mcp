"""The email triage tools."""
import json
from typing import List, Optional

from pydantic import Field

from jev_mcp.app import mcp
from jev_mcp.email_triage.mcp_tool import triage_email_content, configure_triage_labels
from jev_mcp.email_triage.gas_setup import setup_gas_workflow


@mcp.tool(
    name="triage_email_content",
    description=(
        "Evaluates an email using MLX and returns the recommended triage action.\n"
        "If 'status' == 'needs_config', you must ask the user what labels they want to track and call configure_triage_labels."
    )
)
def tool_triage_email_content(
    context_id: str = Field(description="The mailbox or context identifier."),
    subject: str = Field(description="Email subject."),
    sender: str = Field(description="Email sender."),
    body: str = Field(description="Email body content.")
) -> str:
    return triage_email_content(context_id, subject, sender, body)

@mcp.tool(
    name="configure_triage_labels",
    description="Configures dynamic labels for a specific mailbox or context ID."
)
def tool_configure_triage_labels(
    context_id: str = Field(description="The mailbox or context identifier."),
    labels: List[str] = Field(description="List of desired labels/buckets.")
) -> str:
    import json
    return json.dumps(configure_triage_labels(context_id, labels))

@mcp.tool(
    name="setup_gas_workflow",
    description="Automated CLI Setup Flow for Google Apps Script. Pass webhook_url and api_key directly when running in a non-interactive environment."
)
def tool_setup_gas_workflow(
    webhook_url: Optional[str] = Field(None, description="The Webhook URL to route emails to."),
    api_key: Optional[str] = Field(None, description="The secure API Key to protect the webhook.")
) -> str:
    return setup_gas_workflow(webhook_url=webhook_url, api_key=api_key)
