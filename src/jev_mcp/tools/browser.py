"""The jev_run_browser_agent tool."""

import json

from pydantic import Field

from jev_mcp.app import logger, mcp


@mcp.tool(
    name="jev_run_browser_agent",
    description="Ultrafast Browser Agent: Automates complex web interactions (clicks, text entry, navigation) to achieve a given goal in a browser.",
)
def jev_run_browser_agent(
    url: str = Field(description="The starting URL for the browser agent."),
    goal: str = Field(
        description="The natural language objective for the agent to complete."
    ),
    max_steps: int = Field(
        default=20,
        description="Stop after this many browser actions if the goal is not reached.",
    ),
) -> str:
    """
    Runs the jev-ultrafast agent in the user's Chrome until the goal is done, blocked, or the step limit.
    Every choice is made locally by the Jev-MCP Dual-Engine router via ultrafast_adapter.
    """
    from jev_mcp import browser_agent

    try:
        steps = (
            max_steps if isinstance(max_steps, int) else browser_agent.DEFAULT_MAX_STEPS
        )
        return json.dumps(
            browser_agent.run_browser_agent(url, goal, max_steps=steps), indent=2
        )
    except Exception as e:
        logger.error(f"Browser agent failed: {e}")
        return json.dumps({"status": "ERROR", "message": str(e)})
