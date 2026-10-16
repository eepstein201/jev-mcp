import pytest
from fastapi.testclient import TestClient
from jev_mcp.email_triage.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    """Every test here posts from one address; the shared limiter must not count across them."""
    from jev_mcp.email_triage import api
    from jev_mcp.email_triage.rate_limit import SlidingWindowLimiter

    monkeypatch.setattr(api, "limiter", SlidingWindowLimiter(limit=0))


def test_webhook_requires_auth():
    """Asserts that calling the webhook without a valid IdP bearer token returns a 401."""
    response = client.post("/api/v1/triage/email", json={"subject": "Test"})
    assert response.status_code == 401
    assert "Missing Authorization header" in response.json()["detail"]


def test_webhook_end_to_end(monkeypatch):
    """Uses TestClient to send a synthetic email payload to the webhook and mocks MLX logic."""
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")

    # Mock the evaluate_email_with_mlx function inside api.py
    def mock_evaluate(*args, **kwargs):
        return {
            "decision": "Action",
            "requires_action_score": 0.9,
            "is_important_score": 0.9,
            "bucket": "Engineering",
            "suggested_routing": "escalate_to_slack",
        }

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.evaluate_email_with_mlx", mock_evaluate
    )

    # Needs config file so it doesn't return 'needs_config'
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "triage_configs.json"
        config_path.write_text(json.dumps({"test@example.com": ["Engineering"]}))

        monkeypatch.setattr(
            "jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_path
        )

        payload = {
            "context_id": "test@example.com",
            "subject": "Urgent Request",
            "sender": "boss@test.com",
            "body": "Fix it",
        }

        response = client.post(
            "/api/v1/triage/email",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "Action"
        assert data["slack_payload"] is not None
        assert "URGENT ACTION REQUIRED" in str(data["slack_payload"])


def test_webhook_needs_config(monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")
    # By providing no config file, it will trigger needs_config
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(
            "jev_mcp.email_triage.mcp_tool.get_config_path",
            lambda: Path(tmpdir) / "missing.json",
        )
        response = client.post(
            "/api/v1/triage/email",
            json={"context_id": "test@example.com"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 400
        assert "No labels configured" in response.json()["detail"]


def test_webhook_hitl_branch(monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")

    def mock_evaluate(*args, **kwargs):
        return {
            "decision": "Review",
            "requires_action_score": 0.5,
            "is_important_score": 0.5,
            "bucket": "Review",
            "suggested_routing": "hitl_slack_block",
        }

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.evaluate_email_with_mlx", mock_evaluate
    )
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "triage_configs.json"
        config_path.write_text(json.dumps({"test@example.com": ["Review"]}))
        monkeypatch.setattr(
            "jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_path
        )

        response = client.post(
            "/api/v1/triage/email",
            json={"context_id": "test@example.com"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        assert "Archive" in str(response.json()["slack_payload"])


def test_webhook_answers_503_when_no_engine_is_running(monkeypatch, tmp_path):
    import json
    from jev_mcp.provider import EngineUnavailableError

    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")
    config_path = tmp_path / "triage_configs.json"
    config_path.write_text(json.dumps({"me@example.com": ["Engineering"]}))
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_path
    )

    def down(*args, **kwargs):
        raise EngineUnavailableError("No engine answered: requires_action")

    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.evaluate_email_with_mlx", down)

    response = client.post(
        "/api/v1/triage/email",
        json={
            "context_id": "me@example.com",
            "subject": "Hi",
            "sender": "a@b.c",
            "body": "text",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503


def test_webhook_rejects_an_oversized_body_before_any_engine_work(monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")

    def must_not_run(*args, **kwargs):
        raise AssertionError("triage ran on an oversized payload")

    monkeypatch.setattr("jev_mcp.email_triage.api.triage_email_content", must_not_run)

    response = client.post(
        "/api/v1/triage/email",
        json={"context_id": "ctx", "body": "x" * 1_000_001},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422


def _post_raw(test_client, body: bytes):
    return test_client.post(
        "/api/v1/triage/email",
        content=body,
        headers={"content-type": "application/json"},
    )


def test_webhook_rejects_a_body_over_the_request_cap_before_auth():
    from jev_mcp.email_triage import api

    response = _post_raw(client, b"x" * (api.MAX_REQUEST_BYTES + 1))

    assert response.status_code == 413


def test_body_limit_also_stops_a_streamed_body_with_no_content_length():
    from jev_mcp.email_triage.api import BodySizeLimitMiddleware, app

    limited = TestClient(BodySizeLimitMiddleware(app, max_bytes=50))

    def chunks():
        for _ in range(10):
            yield b"y" * 20

    response = limited.post(
        "/api/v1/triage/email",
        content=chunks(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_a_body_under_the_cap_still_reaches_auth():
    response = client.post("/api/v1/triage/email", json={"context_id": "c"})

    assert response.status_code == 401


# --- degraded triage: bounded retry (X-Triage-Attempt) --------------------------------------

import json as _json

from jev_mcp.email_triage import api as _api

AUTH = {"Authorization": "Bearer test-key"}
EMAIL_BODY = {"context_id": "c", "subject": "S", "sender": "a@b.c", "id": "m1"}


def _stub_triage(monkeypatch, routing, degraded):
    monkeypatch.setenv("JEV_MCP_API_KEY", "test-key")
    result = {
        "status": "DEGRADED" if degraded else "success",
        "decision": "Review",
        "suggested_routing": routing,
        "reason": "requires_action",
    }
    if degraded:
        result.update(degraded=True, escalation_failed=["bucket"])
    monkeypatch.setattr(
        _api, "triage_email_content", lambda **kwargs: _json.dumps(result)
    )


def _post(attempt=None):
    headers = dict(AUTH)
    if attempt is not None:
        headers["X-Triage-Attempt"] = attempt
    return client.post("/api/v1/triage/email", json=EMAIL_BODY, headers=headers)


def test_retry_constants():
    assert _api.MAX_DEGRADED_ATTEMPTS == 3
    assert 1 <= _api.DEGRADED_RETRY_AFTER_SECONDS <= 30


@pytest.mark.parametrize("attempt", ["1", "2", " 2 "])
def test_degraded_early_attempt_answers_503_with_retry_after_and_no_slack(
    monkeypatch, attempt
):
    _stub_triage(monkeypatch, "hitl_slack_block", degraded=True)

    response = _post(attempt)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(_api.DEGRADED_RETRY_AFTER_SECONDS)
    assert "slack_payload" not in response.json()
    assert "detail" in response.json()


def test_degraded_early_attempt_builds_no_slack_message(monkeypatch):
    _stub_triage(monkeypatch, "escalate_to_slack", degraded=True)
    built = []
    monkeypatch.setattr(
        _api, "build_escalation_message", lambda *a: built.append(a) or {}
    )
    monkeypatch.setattr(_api, "build_hitl_message", lambda *a: built.append(a) or {})

    assert _post("1").status_code == 503
    assert built == []


@pytest.mark.parametrize(
    "attempt", [None, "", "  ", "abc", "0", "-1", "1.5", "3", "4", "99"]
)
def test_degraded_final_or_unreadable_attempt_answers_200_with_the_notice(
    monkeypatch, attempt
):
    _stub_triage(monkeypatch, "hitl_slack_block", degraded=True)

    response = _post(attempt)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DEGRADED"
    assert body["suggested_routing"] == "hitl_slack_block"
    assert "Unconfirmed" in str(body["slack_payload"])


def test_degraded_final_attempt_that_was_an_archive_is_a_review_card(monkeypatch):
    # The routing itself is fixed where the triage is produced (mcp_tool); here the card of a
    # review routing is built and carries the notice.
    _stub_triage(monkeypatch, "escalate_to_slack", degraded=True)

    body = _post("3").json()

    assert body["suggested_routing"] == "escalate_to_slack"
    assert "Unconfirmed" in str(body["slack_payload"])


@pytest.mark.parametrize("attempt", [None, "1", "2", "3", "junk"])
def test_clean_result_ignores_the_header(monkeypatch, attempt):
    _stub_triage(monkeypatch, "hitl_slack_block", degraded=False)

    response = _post(attempt)

    assert response.status_code == 200
    assert "Unconfirmed" not in str(response.json()["slack_payload"])


def test_retried_degraded_attempts_do_not_use_up_the_rate_limit(monkeypatch):
    from jev_mcp.email_triage.rate_limit import SlidingWindowLimiter

    _stub_triage(monkeypatch, "hitl_slack_block", degraded=True)
    monkeypatch.setattr(_api, "limiter", SlidingWindowLimiter(limit=2))

    assert [_post("1").status_code for _ in range(5)] == [503] * 5
    assert _post("3").status_code == 200
    assert _post("3").status_code == 200
    assert _post("3").status_code == 429
