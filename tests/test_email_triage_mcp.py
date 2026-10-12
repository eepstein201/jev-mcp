import pytest
import os
import json
from pathlib import Path
from jev_mcp.email_triage.mcp_tool import (
    configure_triage_labels,
    triage_email_content,
    get_config_path,
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
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )

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
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )

    response_json = triage_email_content(
        "unknown@mailbox.com", "Test", "bob@test.com", "Hello"
    )
    response = json.loads(response_json)
    assert response["status"] == "needs_config"


def test_triage_mcp_tool_missing_config_in_file(tmp_path, monkeypatch):
    """Asserts the tool returns needs_config if config file exists but missing context_id."""
    config_file = tmp_path / "triage_configs.json"
    config_file.write_text(json.dumps({"other@mailbox.com": ["Sales"]}))
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_file
    )

    response_json = triage_email_content(
        "unknown@mailbox.com", "Test", "bob@test.com", "Hello"
    )
    response = json.loads(response_json)
    assert response["status"] == "needs_config"


def test_triage_mcp_tool_execution(tmp_path, monkeypatch):
    """Asserts the tool executes normally when config is present."""
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])

    from unittest.mock import MagicMock

    mock_provider_instance = MagicMock()

    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.9}},
        "is_important": {"probabilities": {"true": 0.9}},
        "bucket": {"probabilities": {"Engineering": 0.8, "Sales": 0.2}},
    }

    monkeypatch.setattr(
        "jev_mcp.routing_provider.RoutingProvider", lambda: mock_provider_instance
    )

    response_json = triage_email_content(
        "test@mailbox.com", "Urgent Bug", "boss@test.com", "Fix it"
    )

    response = json.loads(response_json)
    assert response["status"] == "success"
    assert response["decision"] == "Action"
    assert response["bucket"] == "Engineering"
    assert response["requires_action_score"] == 0.9

    # Also test the Review branch for complete coverage
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {
            "probabilities": {"true": 0.5}
        },  # Between 0.3 and 0.8 -> Review
        "is_important": {"probabilities": {"true": 0.2}},
        "bucket": {
            "probabilities": {"Engineering": 0.5, "Sales": 0.2}
        },  # Confidence < 0.6 -> Review
    }

    response_json_2 = triage_email_content(
        "test@mailbox.com", "Weird Bug", "boss@test.com", "Fix it maybe"
    )
    response_2 = json.loads(response_json_2)
    assert response_2["status"] == "success"
    assert response_2["decision"] == "Review"


def test_triage_mcp_tool_execution_with_long_body(tmp_path, monkeypatch):
    """Asserts the tool executes compaction when body is long."""
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])

    from unittest.mock import MagicMock

    mock_provider_instance = MagicMock()
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.9}},
        "is_important": {"probabilities": {"true": 0.9}},
        "bucket": {"probabilities": {"Engineering": 0.8, "Sales": 0.2}},
    }

    monkeypatch.setattr(
        "jev_mcp.routing_provider.RoutingProvider", lambda: mock_provider_instance
    )

    def mock_compact(state, goal, confidence_threshold):
        import json

        return json.dumps({"compacted_state": "Compressed body"})

    monkeypatch.setattr("jev_mcp.tools.files.jev_compact_context", mock_compact)

    long_body = "A" * 2001
    response_json = triage_email_content(
        "test@mailbox.com", "Urgent Bug", "boss@test.com", long_body
    )

    response = json.loads(response_json)
    assert response["status"] == "success"

    # Also test failure mode of jev_compact_context
    def mock_compact_fail(state, goal, confidence_threshold):
        raise ValueError("Failed to compact")

    monkeypatch.setattr("jev_mcp.tools.files.jev_compact_context", mock_compact_fail)
    response_json_2 = triage_email_content(
        "test@mailbox.com", "Urgent Bug", "boss@test.com", long_body
    )
    response_2 = json.loads(response_json_2)
    assert response_2["status"] == "success"


def test_triage_evaluates_the_compacted_body_returned_by_the_compact_tool(
    tmp_path, monkeypatch
):
    # Arrange
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])

    from unittest.mock import MagicMock

    mock_provider_instance = MagicMock()
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.9}},
        "is_important": {"probabilities": {"true": 0.9}},
        "bucket": {"probabilities": {"Engineering": 0.8, "Sales": 0.2}},
    }
    monkeypatch.setattr(
        "jev_mcp.routing_provider.RoutingProvider", lambda: mock_provider_instance
    )

    # The real return shape of jev_compact_context (see server.py).
    def mock_compact(state, goal, confidence_threshold):
        return json.dumps(
            {"status": "COMPACTION_COMPLETE", "compacted_state": "Compressed body"}
        )

    monkeypatch.setattr("jev_mcp.tools.files.jev_compact_context", mock_compact)

    # Act
    triage_email_content("test@mailbox.com", "Urgent Bug", "boss@test.com", "A" * 2001)

    # Assert
    evaluated_state = mock_provider_instance.evaluate_batch.call_args[0][0]
    assert evaluated_state["email_body"] == "Compressed body"


def test_triage_compaction_keeps_a_paragraph_the_fast_engine_scores_as_moderately_relevant(
    tmp_path, monkeypatch
):
    # Measured: relevant chunks score roughly 0.32-0.90 on the fast engine and irrelevant ones
    # 0.04-0.35, so a paragraph at 0.4 must survive while one at 0.1 is dropped.
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])
    mock_provider_instance = MagicMock()
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.9}},
        "is_important": {"probabilities": {"true": 0.9}},
        "bucket": {"probabilities": {"Engineering": 0.8, "Sales": 0.2}},
    }
    monkeypatch.setattr(
        "jev_mcp.routing_provider.RoutingProvider", lambda: mock_provider_instance
    )

    def fake_kev(req, timeout=None):
        body = json.loads(req.data)
        noul = 0.4 if "REFUND" in body["state"] else 0.1
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"answers": {k: {"noul": noul} for k in body["questions"]}}
        ).encode("utf-8")
        context = MagicMock()
        context.__enter__.return_value = response
        return context

    long_body = ("REFUND " * 200).strip() + "\n\n" + ("NEWSLETTER " * 150).strip()

    # Act
    with patch("urllib.request.urlopen", side_effect=fake_kev):
        triage_email_content(
            "test@mailbox.com", "Order issue", "customer@test.com", long_body
        )

    # Assert
    evaluated_body = mock_provider_instance.evaluate_batch.call_args[0][0]["email_body"]
    assert "REFUND" in evaluated_body
    assert "NEWSLETTER" not in evaluated_body


def test_triage_logs_a_warning_when_compaction_fails(tmp_path, monkeypatch, caplog):
    # Arrange
    import logging
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("test@mailbox.com", ["Engineering", "Sales"])
    mock_provider_instance = MagicMock()
    mock_provider_instance.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.9}},
        "is_important": {"probabilities": {"true": 0.9}},
        "bucket": {"probabilities": {"Engineering": 0.8, "Sales": 0.2}},
    }
    monkeypatch.setattr(
        "jev_mcp.routing_provider.RoutingProvider", lambda: mock_provider_instance
    )

    def mock_compact_fail(state, goal, confidence_threshold):
        raise ValueError("boom")

    monkeypatch.setattr("jev_mcp.tools.files.jev_compact_context", mock_compact_fail)

    # Act
    with caplog.at_level(logging.WARNING, logger="jev_mcp.email_triage.mcp_tool"):
        response = json.loads(
            triage_email_content(
                "test@mailbox.com", "Urgent Bug", "boss@test.com", "A" * 2001
            )
        )

    # Assert: the original body is still triaged, and the failure is visible.
    assert response["status"] == "success"
    assert any(
        "compaction failed" in record.getMessage().lower() for record in caplog.records
    )


def test_configure_triage_labels_keeps_the_old_file_when_the_write_fails(
    tmp_path, monkeypatch
):
    import os
    import pytest

    config_file = tmp_path / "triage_configs.json"
    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path", lambda: config_file
    )
    configure_triage_labels("a@mailbox.com", ["Sales"])
    before = config_file.read_text()

    def failing_replace(src, dst):
        raise OSError("rename blocked")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        configure_triage_labels("b@mailbox.com", ["HR"])
    assert config_file.read_text() == before
    assert os.listdir(tmp_path) == ["triage_configs.json"]


def test_triage_sends_an_important_email_that_needs_no_action_to_review(
    tmp_path, monkeypatch
):
    """The importance answer the engines return now reaches the routing decision."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("boss@mailbox.com", ["Notifications", "Receipts"])
    provider = MagicMock()
    provider.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": 0.05}},
        "is_important": {"probabilities": {"true": 0.93}},
        "bucket": {"probabilities": {"Notifications": 0.9, "Receipts": 0.1}},
    }
    monkeypatch.setattr("jev_mcp.routing_provider.RoutingProvider", lambda: provider)

    response = json.loads(
        triage_email_content(
            "boss@mailbox.com", "FYI: board notes", "ceo@test.com", "For your records."
        )
    )

    assert response["decision"] == "Review"
    assert response["suggested_routing"] == "hitl_slack_block"
    assert response["is_important_score"] == 0.93
    assert response["bucket"] == "Notifications"


def _triage_with_scores(tmp_path, monkeypatch, action, important):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("tune@mailbox.com", ["Notifications", "Receipts"])
    provider = MagicMock()
    provider.evaluate_batch.return_value = {
        "requires_action": {"probabilities": {"true": action}},
        "is_important": {"probabilities": {"true": important}},
        "bucket": {"probabilities": {"Notifications": 0.9, "Receipts": 0.1}},
    }
    monkeypatch.setattr("jev_mcp.routing_provider.RoutingProvider", lambda: provider)
    return json.loads(
        triage_email_content("tune@mailbox.com", "FYI", "ceo@test.com", "For you.")
    )


def test_triage_honours_an_importance_threshold_set_in_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_TRIAGE_IMPORTANT_THRESHOLD", "0.99")

    response = _triage_with_scores(tmp_path, monkeypatch, action=0.05, important=0.93)

    # 0.93 is over the 0.8 default but under the configured 0.99, so it is filed, not reviewed.
    assert response["decision"] == "Notifications"
    assert response["suggested_routing"] == "archive_and_label"


def test_triage_honours_an_action_threshold_set_in_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JEV_TRIAGE_ACTION_THRESHOLD", "0.5")

    response = _triage_with_scores(tmp_path, monkeypatch, action=0.6, important=0.1)

    # 0.6 is "ambiguous" under the 0.8 default but an Action under the configured 0.5.
    assert response["decision"] == "Action"
    assert response["suggested_routing"] == "escalate_to_slack"


def test_the_triage_result_includes_the_reason(tmp_path, monkeypatch):
    response = _triage_with_scores(tmp_path, monkeypatch, action=0.05, important=0.93)

    assert response["reason"] == "important_no_action"
