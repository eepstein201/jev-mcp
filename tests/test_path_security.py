import json
from unittest.mock import patch

from jev_mcp.security import is_safe_path
from jev_mcp.server import jev_read_file, jev_scan_repo


def test_read_file_blocks_system_paths():
    res = json.loads(jev_read_file("/etc/passwd", "find secrets"))
    assert res["status"] == "BLOCKED"
    assert "sensitive" in res["message"].lower()


def test_read_file_blocks_tilde_credential_paths():
    res = json.loads(jev_read_file("~/.ssh/id_rsa", "read the key"))
    assert res["status"] == "BLOCKED"


def test_scan_repo_blocks_system_paths():
    res = json.loads(jev_scan_repo("/etc/ssh", "find config"))
    assert res["status"] == "BLOCKED"
    assert "sensitive" in res["message"].lower()


def test_is_safe_path_tilde_expansion():
    assert is_safe_path("~/.ssh/id_rsa") is False
    assert is_safe_path("~/.aws/credentials") is False


def test_is_safe_path_allows_user_temp():
    import os
    import tempfile

    assert is_safe_path(os.path.join(tempfile.gettempdir(), "scratch.py")) is True


@patch("jev_mcp.routing_provider.RoutingProvider.evaluate_batch")
def test_read_file_allows_user_temp(mock_eval, tmp_path):
    mock_eval.return_value = {"file_eval": {"probabilities": {"true": 0.99}}}
    f = tmp_path / "note.py"
    f.write_text("def x(): pass")

    res = jev_read_file(str(f), "find x", filter_by_chunk=False)
    assert "passed relevance check" in res
