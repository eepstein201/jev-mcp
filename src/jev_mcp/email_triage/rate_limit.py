import logging
import os
import threading
import time
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)

RATE_LIMIT_ENV_VAR = "JEV_MCP_RATE_LIMIT_PER_MINUTE"
DEFAULT_LIMIT_PER_MINUTE = 60


class SlidingWindowLimiter:
    """Allows each key at most `limit` requests in any `window_seconds`. `limit <= 0` allows all."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        # Ordered by each key's latest allowed request, oldest first: an allowed request moves
        # its key to the end, so stale keys and the key to drop are always at the front.
        self._allowed: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Counts a request. None when allowed, otherwise seconds until the next is allowed.

        A refused request is not counted, so retrying early never lengthens the wait.
        """
        if self.limit <= 0:
            return None

        with self._lock:
            now = self._clock()
            cutoff = now - self.window_seconds
            self._forget_stale_keys(cutoff)

            times = self._allowed.get(key)
            if times is None:
                times = deque()
            else:
                # A key that survived the sweep above has a request inside the window.
                while times[0] <= cutoff:
                    times.popleft()
                if len(times) >= self.limit:
                    return times[0] + self.window_seconds - now
                del self._allowed[key]

            times.append(now)
            self._allowed[key] = times
            while len(self._allowed) > self.max_keys:
                del self._allowed[next(iter(self._allowed))]
            return None

    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._allowed)

    def _forget_stale_keys(self, cutoff: float) -> None:
        while self._allowed:
            oldest = next(iter(self._allowed))
            if self._allowed[oldest][-1] > cutoff:
                return
            del self._allowed[oldest]


def limit_from_env() -> int:
    """Requests allowed per client per minute; 0 disables the limit."""
    raw = os.environ.get(RATE_LIMIT_ENV_VAR)
    if raw is None:
        return DEFAULT_LIMIT_PER_MINUTE
    try:
        limit = int(raw)
    except ValueError:
        limit = -1
    if limit < 0:
        logger.warning(
            "%s=%r is not a whole number of 0 or more; using %d.",
            RATE_LIMIT_ENV_VAR,
            raw,
            DEFAULT_LIMIT_PER_MINUTE,
        )
        return DEFAULT_LIMIT_PER_MINUTE
    return limit
