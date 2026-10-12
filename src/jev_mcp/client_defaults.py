"""Map the connecting MCP client to router default buckets.

Each entry pairs the client-name patterns of an MCP client family with the
tier table of that family's provider. Tier model ids are hand-picked from the
models.dev catalog (the same-family auto-update then keeps them current);
Google's top tiers repeat ``gemini-3.8-flash`` because its catalog currently
offers nothing newer above the flash family (``pro`` is stale at 2.5).
"""

import logging
from typing import Any

from jev_mcp.router_config import FALLBACK_BUCKETS

logger = logging.getLogger("jev-mcp")

PROVIDER_CLIENT_TABLE: tuple[dict[str, Any], ...] = (
    {
        "provider": "anthropic",
        "patterns": ("claude code", "claude desktop", "claude", "antigravity"),
        "buckets": dict(FALLBACK_BUCKETS),
    },
    {
        "provider": "openai",
        "patterns": ("opencode", "cursor", "windsurf", "continue", "zed"),
        "buckets": {
            "b1": "gpt-5.4-nano",
            "b2": "gpt-5.4-mini",
            "b3": "gpt-5.5",
            "b4": "gpt-5.5-pro",
        },
    },
    {
        "provider": "google",
        "patterns": ("gemini",),
        "buckets": {
            "b1": "gemini-3.5-flash-lite",
            "b2": "gemini-3.8-flash",
            "b3": "gemini-3.8-flash",
            "b4": "gemini-3.8-flash",
        },
    },
)

# Kept for compatibility with earlier references to the Anthropic patterns.
ANTHROPIC_CLIENT_PATTERNS: tuple[str, ...] = PROVIDER_CLIENT_TABLE[0]["patterns"]


def default_buckets_for_client(client_name: str) -> dict[str, str] | None:
    """The provider tier table for a connecting client, or None if unrecognized."""
    lowered = (client_name or "").lower()
    for entry in PROVIDER_CLIENT_TABLE:
        if any(pattern in lowered for pattern in entry["patterns"]):
            return dict(entry["buckets"])
    return None


class SeedingMiddleware:
    """Seeds router defaults from the connecting client, once per server process.

    Fires on ``notifications/initialized`` (right after the handshake, when
    ``ctx.session.client_params`` is populated). Never issues server-to-client
    requests — awaiting one inside the middleware chain deadlocks the
    handshake — and a seeding failure must never break the connection.
    """

    def __init__(self) -> None:
        import threading

        self._done = threading.Event()

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if ctx.method == "notifications/initialized" and not self._done.is_set():
            self._done.set()
            try:
                self._seed(ctx)
            except Exception as e:  # noqa: BLE001 - never break the handshake
                logger.warning(f"Client-based default seeding failed: {e}")
        return await call_next(ctx)

    def _seed(self, ctx: Any) -> None:
        params = getattr(ctx.session, "client_params", None)
        # The wire key is `clientInfo`; the SDK's parsed model exposes `client_info`.
        client_info = getattr(params, "client_info", None) or getattr(
            params, "clientInfo", None
        )
        client_name = getattr(client_info, "name", None)
        if not client_name:
            return
        from jev_mcp import router_config

        router_config.seed_defaults_for_client(client_name)
