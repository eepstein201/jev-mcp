"""The email triage tools."""
import contextlib
import json
import os
import sys
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import Field

from jev_mcp.app import logger, mcp
from jev_mcp.email_triage.mcp_tool import triage_email_content, configure_triage_labels
from jev_mcp.email_triage.gas_setup import setup_gas_workflow


def _error(message: str) -> str:
    return json.dumps({"status": "ERROR", "message": message})


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
    try:
        return triage_email_content(context_id, subject, sender, body)
    except Exception as e:
        logger.error(f"triage_email_content failed: {e}")
        return _error(str(e))

@mcp.tool(
    name="configure_triage_labels",
    description="Configures dynamic labels for a specific mailbox or context ID."
)
def tool_configure_triage_labels(
    context_id: str = Field(description="The mailbox or context identifier."),
    labels: List[str] = Field(description="List of desired labels/buckets.")
) -> str:
    try:
        return json.dumps(configure_triage_labels(context_id, labels))
    except Exception as e:
        logger.error(f"configure_triage_labels failed: {e}")
        return _error(str(e))

@mcp.tool(
    name="setup_gas_workflow",
    description="Automated Setup Flow for Google Apps Script. webhook_url is required. Leave api_key unset to use the JEV_MCP_API_KEY environment variable, which keeps the key out of the conversation."
)
def tool_setup_gas_workflow(
    webhook_url: Optional[str] = Field(None, description="The Webhook URL to route emails to."),
    api_key: Optional[str] = Field(None, description="The secure API Key to protect the webhook. Defaults to the JEV_MCP_API_KEY environment variable.")
) -> str:
    # The wizard prompts on stdin for anything missing; here stdin is the JSON-RPC stream.
    parsed = urlparse(webhook_url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return _error("webhook_url must be an http:// or https:// address.")
    key = api_key or os.environ.get("JEV_MCP_API_KEY")
    if not key:
        return _error("No API key: pass api_key or set the JEV_MCP_API_KEY environment variable.")

    try:
        # The wizard prints its progress; stdout is the JSON-RPC stream, so send that to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            url = setup_gas_workflow(webhook_url=webhook_url, api_key=key)
        return json.dumps({"status": "SUCCESS", "url": url})
    except Exception as e:
        logger.error(f"setup_gas_workflow failed: {e}")
        return _error(str(e))
