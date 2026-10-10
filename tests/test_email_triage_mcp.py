import pytest
import os
import json
from pathlib import Path
from jev_mcp.email_triage.mcp_tool import (
    configure_triage_labels,
    triage_email_content,
    get_config_path
)

def test_get_config_path(monkeypatch, tmp_path):
    """Test get_config_path creates dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = get_config_path()
    assert path.name == "triage_configs.json"
    assert path.parent.name == ".jev-mcp"
    assert path.parent.exists()

def test_configure_triage_labels_persists(tmp_path, monkeypatch):
    """Asserts the config tool correctly writes dynamic categories to disk."""
    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: tmp_path / "triage_configs.json")
    
    result = configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])
    assert result["status"] == "success"
    
    # Verify disk
    config_file = tmp_path / "triage_configs.json"
    assert config_file.exists()
    
    # Test updating an existing config file to hit line 14
    result = configure_triage_labels("test2@mailbox.com", ["HR"])
    assert result["status"] == "success"
    
    data = json.loads(config_file.read_text())
    assert "test@mailbox.com" in data
    assert "test2@mailbox.com" in data

def test_triage_mcp_tool_missing_config_no_file(tmp_path, monkeypatch):
    """Asserts the tool returns needs_config if config file is missing."""
    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: tmp_path / "triage_configs.json")
    
    response_json = triage_email_content("unknown@mailbox.com", "Test", "bob@test.com", "Hello")
    response = json.loads(response_json)
    assert response["status"] == "needs_config"

def test_triage_mcp_tool_missing_config_in_file(tmp_path, monkeypatch):
    """Asserts the tool returns needs_config if config file exists but missing context_id."""
    config_file = tmp_path / "triage_configs.json"
    config_file.write_text(json.dumps({"other@mailbox.com": ["Sales"]}))
    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_file)
    
    response_json = triage_email_content("unknown@mailbox.com", "Test", "bob@test.com", "Hello")
    response = json.loads(response_json)
    assert response["status"] == "needs_config"

def test_triage_mcp_tool_execution(tmp_path, monkeypatch):
    """Asserts the tool executes normally when config is present."""
    monkeypatch.setattr("jev_mcp.email_triage.mcp_tool.get_config_path", lambda: tmp_path / "triage_configs.json")
    configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])
    
    from unittest.mock import MagicMock
    mock_provider_instance = MagicMock()
    
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.9}},
        "is_important": {"probabilities": {"true": 0.9}},
        "bucket": {"probabilities": {"Engineering": 0.8, "Sales": 0.2}}
    }
    
    monkeypatch.setattr("jev_mcp.routing_provider.RoutingProvider", lambda: mock_provider_instance)
    
    response_json = triage_email_content("test@mailbox.com", "Urgent Bug", "boss@test.com", "Fix it")
    
    response = json.loads(response_json)
    assert response["status"] == "success"
    assert response["decision"] == "Action"
    assert response["bucket"] == "Engineering"
    assert response["requires_action_score"] == 0.9
    
    # Also test the Review branch for complete coverage
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.5}}, # Between 0.3 and 0.8 -> Review
        "is_important": {"probabilities": {"true": 0.2}},
        "bucket": {"probabilities": {"Engineering": 0.5, "Sales": 0.2}} # Confidence < 0.6 -> Review
    }
    
    response_json_2 = triage_email_content("test@mailbox.com", "Weird Bug", "boss@test.com", "Fix it maybe")
    response_2 = json.loads(response_json_2)
    assert response_2["status"] == "success"
    assert response_2["decision"] == "Review"
