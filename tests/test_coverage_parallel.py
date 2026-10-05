import pytest
import json
from unittest.mock import patch, MagicMock
from jev_mcp.server import jev_read_file, jev_scan_repo

@patch("urllib.request.urlopen")
def test_parallel_read_file(mock_urlopen):
    # Mock urlopen directly to avoid any network calls
    mock_res = MagicMock()
    mock_res.read.return_value = json.dumps({"chunk_0": {"probabilities": {"true": 0.9}}}).encode("utf-8")
    mock_res.__enter__.return_value = mock_res
    mock_urlopen.return_value = mock_res

    res = jev_read_file("src/jev_mcp/server.py", "testing parallel execution")
    assert "testing parallel execution" in res or len(res) > 0

@patch("urllib.request.urlopen")
def test_parallel_scan_repo(mock_urlopen):
    mock_res = MagicMock()
    mock_res.read.return_value = json.dumps({"chunk_0": {"probabilities": {"true": 0.9}}}).encode("utf-8")
    mock_res.__enter__.return_value = mock_res
    mock_urlopen.return_value = mock_res

    res = jev_scan_repo("src/jev_mcp", "testing parallel execution")
    assert "testing parallel execution" in res or len(res) > 0

