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
            "suggested_routing": "escalate_to_slack"
        }
    
    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.evaluate_email_with_mlx", mock_evaluate)
    
    # Needs config file so it doesn't return 'needs_config'
    import tempfile
    import json
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "triage_configs.json"
        config_path.write_text(json.dumps({"test@example.com": ["Engineering"]}))
        
        monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_path)
        
        payload = {
            "context_id": "test@example.com",
            "subject": "Urgent Request",
            "sender": "boss@test.com",
            "body": "Fix it"
        }
        
        response = client.post(
            "/api/v1/triage/email", 
            json=payload,
            headers={"Authorization": "Bearer test-key"}
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
        monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: Path(tmpdir) / "missing.json")
        response = client.post(
            "/api/v1/triage/email", 
            json={"context_id": "test@example.com"},
            headers={"Authorization": "Bearer test-key"}
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
            "suggested_routing": "hitl_slack_block"
        }
    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.evaluate_email_with_mlx", mock_evaluate)
    import tempfile
    import json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "triage_configs.json"
        config_path.write_text(json.dumps({"test@example.com": ["Review"]}))
        monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_path)
        
        response = client.post(
            "/api/v1/triage/email", 
            json={"context_id": "test@example.com"},
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        assert "Archive" in str(response.json()["slack_payload"])
