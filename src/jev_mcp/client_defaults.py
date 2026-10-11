"""Map the connecting MCP client to router default buckets.

v1 is Anthropic-only: the patterns below recognize Claude-family clients and
map them to the Anthropic tiers in ``router_config.FALLBACK_BUCKETS`` (which
the model catalog keeps current). Every other client maps to ``None`` and
seeding never happens for it.
"""
import logging
from typing import Any

from jev_mcp.router_config import FALLBACK_BUCKETS

logger = logging.getLogger("jev-mcp")

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
        client_info = getattr(params, "clientInfo", None)
        client_name = getattr(client_info, "name", None)
        if not client_name:
            return
        from jev_mcp import router_config
        router_config.seed_defaults_for_client(client_name)
