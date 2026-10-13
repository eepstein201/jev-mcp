"""The email triage tools."""

import contextlib
import json
import sys
from typing import List, Optional
from urllib.parse import urlparse

from mcp.types import ToolAnnotations
from pydantic import Field

from jev_mcp.app import logger, mcp
from jev_mcp.email_triage.mcp_tool import triage_email_content, configure_triage_labels
from jev_mcp.email_triage.gas_setup import KEY_NEXT_STEP, setup_gas_workflow


def _error(message: str) -> str:
    return json.dumps({"status": "ERROR", "message": message})


@mcp.tool(
    name="jev_triage_email",
    title="Triage an email",
    description=(
        "Evaluates an email using MLX and returns the recommended triage action.\n"
        "If 'status' == 'needs_config', you must ask the user what labels they want to track and call jev_configure_triage_labels."
    ),
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def tool_triage_email_content(
    context_id: str = Field(description="The mailbox or context identifier."),
    subject: str = Field(description="Email subject."),
    sender: str = Field(description="Email sender."),
    body: str = Field(description="Email body content."),
) -> str:
    try:
        return triage_email_content(context_id, subject, sender, body)
    except Exception as e:
        logger.error(f"triage_email_content failed: {e}")
        return _error(str(e))


@mcp.tool(
    name="jev_configure_triage_labels",
    title="Set a mailbox's triage labels",
    description="Configures dynamic labels for a specific mailbox or context ID.",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def tool_configure_triage_labels(
    context_id: str = Field(description="The mailbox or context identifier."),
    labels: List[str] = Field(description="List of desired labels/buckets."),
) -> str:
    try:
        return json.dumps(configure_triage_labels(context_id, labels))
    except Exception as e:
        logger.error(f"configure_triage_labels failed: {e}")
        return _error(str(e))


@mcp.tool(
    name="jev_setup_gas_workflow",
    title="Deploy the Gmail triage script",
    description="Automated Setup Flow for Google Apps Script. webhook_url is required. The webhook key is never passed here: the deployed script reads it from the Script Property JEV_MCP_API_KEY, which the user sets once in Project Settings (see next_step in the result).",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=True,
    ),
)
def tool_setup_gas_workflow(
    webhook_url: Optional[str] = Field(
        None, description="The Webhook URL to route emails to."
    ),
) -> str:
    # The wizard prompts on stdin for anything missing; here stdin is the JSON-RPC stream.
    parsed = urlparse(webhook_url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return _error("webhook_url must be an http:// or https:// address.")

    try:
        # The wizard prints its progress; stdout is the JSON-RPC stream, so send that to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            url = setup_gas_workflow(webhook_url=webhook_url)
        return json.dumps({"status": "SUCCESS", "url": url, "next_step": KEY_NEXT_STEP})
    except Exception as e:
        logger.error(f"setup_gas_workflow failed: {e}")
        return _error(str(e))
