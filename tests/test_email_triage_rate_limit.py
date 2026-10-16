import asyncio
import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from jev_mcp.email_triage import api
from jev_mcp.email_triage.rate_limit import SlidingWindowLimiter, limit_from_env

client = TestClient(api.app)

ENV_VAR = "JEV_MCP_RATE_LIMIT_PER_MINUTE"
AUTH_HEADERS = {"Authorization": "Bearer test-key"}


def _counting_webhook(monkeypatch, limiter):
    """Installs a limiter, valid credentials and a triage stub that needs no engine."""
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")
    monkeypatch.setattr(api, "limiter", limiter)
    monkeypatch.setattr(
        api, "triage_email_content", lambda **kwargs: '{"status": "OK"}'
    )


def _post(headers=None):
    return client.post(
        "/api/v1/triage/email", json={"context_id": "c"}, headers=headers
    )


def test_requests_up_to_the_limit_are_allowed_and_the_next_is_refused():
    limiter = SlidingWindowLimiter(limit=3, clock=lambda: 100.0)
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    assert limiter.check("a") == pytest.approx(60.0)


def test_the_window_slides():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=1, clock=lambda: now[0])
    assert limiter.check("a") is None
    now[0] = 61.0
    assert limiter.check("a") is None


def test_the_wait_shrinks_as_the_oldest_request_ages():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, clock=lambda: now[0])
    limiter.check("a")
    now[0] = 10.0
    limiter.check("a")
    now[0] = 45.0

    assert limiter.check("a") == pytest.approx(15.0)


def test_only_requests_inside_the_window_count():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, clock=lambda: now[0])
    limiter.check("a")
    now[0] = 30.0
    limiter.check("a")
    now[0] = 70.0

    assert limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(20.0)


def test_refused_requests_do_not_extend_the_wait():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=1, clock=lambda: now[0])
    limiter.check("a")
    for second in range(1, 60):
        now[0] = float(second)
        assert limiter.check("a") is not None
    now[0] = 60.0

    assert limiter.check("a") is None


def test_clients_are_limited_independently():
    limiter = SlidingWindowLimiter(limit=1, clock=lambda: 0.0)
    assert limiter.check("a") is None and limiter.check("b") is None


def test_a_burst_of_ten_from_one_client_is_allowed_at_the_default_limit(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    limiter = SlidingWindowLimiter(limit=limit_from_env(), clock=lambda: 0.0)
    assert all(limiter.check("apps-script") is None for _ in range(10))


def test_tracked_clients_are_bounded():
    limiter = SlidingWindowLimiter(limit=1, clock=lambda: 0.0, max_keys=100)
    for i in range(1_000):
        limiter.check(f"client-{i}")
    assert limiter.tracked_keys() <= 100


def test_the_client_dropped_at_the_bound_is_the_least_recently_allowed():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, clock=lambda: now[0], max_keys=2)
    limiter.check("first")
    limiter.check("second")
    limiter.check("first")
    limiter.check("third")

    # "second" was dropped, so it starts again; "first" is still at its limit.
    assert limiter.check("first") is not None
    assert limiter.check("second") is None


def test_clients_whose_requests_have_aged_out_are_forgotten():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=5, clock=lambda: now[0])
    for i in range(50):
        limiter.check(f"client-{i}")
    now[0] = 61.0
    limiter.check("late")

    assert limiter.tracked_keys() == 1


def test_zero_disables_the_limit(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "0")
    limiter = SlidingWindowLimiter(limit=limit_from_env(), clock=lambda: 0.0)
    assert all(limiter.check("a") is None for _ in range(500))
    assert limiter.tracked_keys() == 0


def test_the_limit_defaults_to_sixty_per_minute(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)

    assert limit_from_env() == 60


def test_the_limit_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, " 5 ")

    assert limit_from_env() == 5


@pytest.mark.parametrize("value", ["abc", "", "1.5", "-1"])
def test_an_unusable_limit_logs_a_warning_and_falls_back_to_the_default(
    monkeypatch, caplog, value
):
    monkeypatch.setenv(ENV_VAR, value)

    with caplog.at_level(logging.WARNING):
        assert limit_from_env() == 60

    assert ENV_VAR in caplog.text


def test_the_webhook_answers_429_with_retry_after_past_the_limit(monkeypatch):
    _counting_webhook(monkeypatch, SlidingWindowLimiter(limit=1, clock=lambda: 0.0))

    assert _post(AUTH_HEADERS).status_code == 200
    response = _post(AUTH_HEADERS)

    assert response.status_code == 429 and int(response.headers["retry-after"]) >= 1
    assert response.headers["retry-after"] == "60"
    assert response.json() == {"detail": "Too many requests"}


def test_unauthenticated_requests_get_401_and_never_consume_the_allowance(monkeypatch):
    _counting_webhook(monkeypatch, SlidingWindowLimiter(limit=1, clock=lambda: 0.0))

    statuses = [_post().status_code for _ in range(20)]

    assert statuses == [401] * 20
    assert _post(AUTH_HEADERS).status_code == 200


def test_a_throttled_request_with_a_valid_key_never_reaches_triage(monkeypatch):
    _counting_webhook(monkeypatch, SlidingWindowLimiter(limit=1, clock=lambda: 0.0))
    _post(AUTH_HEADERS)

    def must_not_run(*args, **kwargs):
        raise AssertionError("triage ran on a throttled request")

    monkeypatch.setattr(api, "triage_email_content", must_not_run)
    response = _post(AUTH_HEADERS)

    assert response.status_code == 429


def test_retry_after_is_rounded_up_to_whole_seconds(monkeypatch):
    now = [0.0]
    _counting_webhook(monkeypatch, SlidingWindowLimiter(limit=1, clock=lambda: now[0]))
    _post(AUTH_HEADERS)
    now[0] = 59.6

    response = _post(AUTH_HEADERS)

    assert response.headers["retry-after"] == "1"


def test_the_limit_counts_only_authenticated_requests(monkeypatch):
    _counting_webhook(monkeypatch, SlidingWindowLimiter(limit=2, clock=lambda: 0.0))

    authorized = [_post(AUTH_HEADERS).status_code for _ in range(3)]
    anonymous = [_post().status_code for _ in range(3)]

    assert authorized == [200, 200, 429]
    assert anonymous == [401, 401, 401]


def test_a_request_with_no_client_address_is_still_limited(monkeypatch):
    _counting_webhook(monkeypatch, SlidingWindowLimiter(limit=1, clock=lambda: 0.0))
    anonymous = TestClient(api.app, client=None)

    first = anonymous.post(
        "/api/v1/triage/email", json={"context_id": "c"}, headers=AUTH_HEADERS
    )
    response = anonymous.post(
        "/api/v1/triage/email", json={"context_id": "c"}, headers=AUTH_HEADERS
    )

    assert first.status_code == 200
    assert response.status_code == 429


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    """Stands in for starlette's Request: the limiter must key on request.client.host."""

    def __init__(self, host: str) -> None:
        self.client = _FakeClient(host)


def test_the_limit_is_keyed_on_the_asgi_reported_client_address(monkeypatch):
    monkeypatch.setattr(
        api, "limiter", SlidingWindowLimiter(limit=1, clock=lambda: 0.0)
    )
    sender = _FakeRequest("203.0.113.7")

    asyncio.run(api.enforce_rate_limit(sender, auth={"sub": "sender"}))
    # A second address has its own allowance.
    asyncio.run(
        api.enforce_rate_limit(_FakeRequest("198.51.100.9"), auth={"sub": "other"})
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(api.enforce_rate_limit(sender, auth={"sub": "sender"}))

    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["Retry-After"] == "60"


def test_refund_gives_back_the_newest_count():
    limiter = SlidingWindowLimiter(limit=1, clock=lambda: 0.0)
    assert limiter.check("a") is None
    assert limiter.check("a") is not None

    limiter.refund("a")

    assert limiter.tracked_keys() == 0
    assert limiter.check("a") is None


def test_refund_keeps_older_counts_and_ignores_unknown_keys():
    limiter = SlidingWindowLimiter(limit=2, clock=lambda: 0.0)
    limiter.check("a")
    limiter.check("a")
    limiter.refund("a")
    limiter.refund("nobody")

    assert limiter.check("a") is None
    assert limiter.check("a") is not None
