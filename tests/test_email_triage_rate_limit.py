import logging

import pytest
from fastapi.testclient import TestClient

from jev_mcp.email_triage import api
from jev_mcp.email_triage.rate_limit import SlidingWindowLimiter, limit_from_env

client = TestClient(api.app)

ENV_VAR = "JEV_MCP_RATE_LIMIT_PER_MINUTE"


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


def test_the_webhook_answers_429_with_retry_after_before_checking_auth(monkeypatch):
    monkeypatch.setattr(
        api, "limiter", SlidingWindowLimiter(limit=1, clock=lambda: 0.0)
    )
    client.post("/api/v1/triage/email", json={"context_id": "c"})
    response = client.post("/api/v1/triage/email", json={"context_id": "c"})
    assert response.status_code == 429 and int(response.headers["retry-after"]) >= 1
    assert response.headers["retry-after"] == "60"
    assert response.json() == {"detail": "Too many requests"}


def test_a_throttled_request_with_a_valid_key_never_reaches_triage(monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")
    monkeypatch.setattr(
        api, "limiter", SlidingWindowLimiter(limit=1, clock=lambda: 0.0)
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("triage ran on a throttled request")

    client.post("/api/v1/triage/email", json={"context_id": "c"})
    monkeypatch.setattr(api, "triage_email_content", must_not_run)

    response = client.post(
        "/api/v1/triage/email",
        json={"context_id": "c"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 429


def test_retry_after_is_rounded_up_to_whole_seconds(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        api, "limiter", SlidingWindowLimiter(limit=1, clock=lambda: now[0])
    )
    client.post("/api/v1/triage/email", json={"context_id": "c"})
    now[0] = 59.6

    response = client.post("/api/v1/triage/email", json={"context_id": "c"})

    assert response.headers["retry-after"] == "1"


def test_requests_under_the_limit_still_reach_auth(monkeypatch):
    monkeypatch.setattr(
        api, "limiter", SlidingWindowLimiter(limit=2, clock=lambda: 0.0)
    )

    statuses = [
        client.post("/api/v1/triage/email", json={"context_id": "c"}).status_code
        for _ in range(3)
    ]

    assert statuses == [401, 401, 429]


def test_a_request_with_no_client_address_is_still_limited(monkeypatch):
    monkeypatch.setattr(
        api, "limiter", SlidingWindowLimiter(limit=1, clock=lambda: 0.0)
    )
    anonymous = TestClient(api.app, client=None)

    anonymous.post("/api/v1/triage/email", json={"context_id": "c"})
    response = anonymous.post("/api/v1/triage/email", json={"context_id": "c"})

    assert response.status_code == 429
