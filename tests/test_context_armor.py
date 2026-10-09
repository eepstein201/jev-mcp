import pytest
import json
from unittest.mock import patch, MagicMock
from jev_mcp.server import _safe_truncate, jev_read_file, jev_scan_repo, jev_compact_context

def test_safe_truncate():
    content = "a" * 25000
    safe, trunc = _safe_truncate(content, max_length=24000)
    assert trunc is True
    assert len(safe) == 24000
    assert "[TRUNCATED" in safe

    content = "a" * 100
    safe, trunc = _safe_truncate(content, max_length=24000)
    assert trunc is False
    assert len(safe) == 100

@patch("jev_mcp.routing_provider.RoutingProvider")
def test_read_file_truncation(mock_provider_class):
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {"file_eval": {"probabilities": {"true": 0.99}}, "chunk_0": {"probabilities": {"true": 0.99}}}
    
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "a" * 30000
        mock_open.return_value.__enter__.return_value = mock_file
        
        with patch("jev_mcp.security.is_safe_path", return_value=True):
            with patch("os.path.exists", return_value=True):
                res = jev_read_file("/fake/file.txt", "test", filter_by_chunk=False)
                parsed = json.loads(res)
                assert parsed["status"] == "SUCCESS"
                assert len(parsed["content"]) == 24000
                
                res2 = jev_read_file("/fake/file.txt", "test", filter_by_chunk=True)
                parsed2 = json.loads(res2)
                assert parsed2["status"] == "SUCCESS"
                assert len(parsed2["content"]) <= 24100

@patch("jev_mcp.routing_provider.RoutingProvider")
def test_scan_repo_truncation(mock_provider_class):
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {"chunk_0": {"probabilities": {"true": 0.99}}}
    
    with patch("jev_mcp.security.is_safe_path", return_value=True):
        with patch("os.path.isdir", return_value=True):
            with patch("jev_mcp.scanner.walk_repository", return_value=["/fake/file.txt"]):
                with patch("builtins.open", MagicMock()) as mock_open:
                    mock_file = MagicMock()
                    mock_file.read.return_value = "a" * 26000
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    res = jev_scan_repo("/fake", "test")
                    parsed = json.loads(res)
                    assert parsed["status"] == "SUCCESS"
                    assert len(parsed["content"]) <= 24100

@patch("jev_mcp.routing_provider.RoutingProvider")
def test_compact_truncation(mock_provider_class):
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    mock_provider.evaluate_batch.return_value = {"chunk_0": {"noul": 0.99}}
    
    res = jev_compact_context({"text": "a" * 25000}, "test")
    parsed = json.loads(res)
    assert "TRUNCATED" in parsed["status"]
    assert len(parsed["compacted_state"]) <= 24100
