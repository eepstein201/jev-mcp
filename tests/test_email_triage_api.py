import pytest
from fastapi.testclient import TestClient
from jev_mcp.email_triage.api import app

client = TestClient(app)


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
