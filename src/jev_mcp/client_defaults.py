"""Map the connecting MCP client to router default buckets.

v1 is Anthropic-only: the patterns below recognize Claude-family clients and
map them to the Anthropic tiers in ``router_config.FALLBACK_BUCKETS`` (which
the model catalog keeps current). Every other client maps to ``None`` and
seeding never happens for it.
"""
from jev_mcp.router_config import FALLBACK_BUCKETS

ANTHROPIC_CLIENT_PATTERNS: tuple[str, ...] = (
    "claude code",
    "claude desktop",
    "claude",
    "antigravity",
)


def default_buckets_for_client(client_name: str) -> dict[str, str] | None:
    """The provider tier table for a connecting client, or None if unrecognized."""
    lowered = (client_name or "").lower()
    if any(pattern in lowered for pattern in ANTHROPIC_CLIENT_PATTERNS):
        return dict(FALLBACK_BUCKETS)
    return None
