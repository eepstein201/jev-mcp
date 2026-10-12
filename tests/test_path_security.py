import json
from unittest.mock import patch

import pytest

from jev_mcp.security import is_safe_path
from jev_mcp.server import jev_read_file, jev_scan_repo


@pytest.fixture(autouse=True)
def _root_the_fabricated_paths(monkeypatch):
    # The fabricated /Users/test paths below must sit inside an allowed root, so a False
    # from is_safe_path in this file means the denylist fired, not "outside the roots".
    monkeypatch.setenv("JEV_ALLOWED_ROOTS", "/Users/test")


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


def test_is_safe_path_blocks_credentials_under_temp():
    import os
    import tempfile

    assert is_safe_path(os.path.join(tempfile.gettempdir(), ".ssh", "id_rsa")) is False
    assert (
        is_safe_path(os.path.join(tempfile.gettempdir(), "sub", ".aws", "credentials"))
        is False
    )


def test_is_safe_path_blocks_netrc_and_git_credentials():
    assert is_safe_path("/Users/test/.netrc") is False
    assert is_safe_path("/Users/test/.git-credentials") is False


@patch("jev_mcp.routing_provider.RoutingProvider.evaluate_batch")
def test_read_file_expands_tilde_paths(mock_eval, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    f = tmp_path / "note.py"
    f.write_text("def x(): pass")
    mock_eval.return_value = {"file_eval": {"probabilities": {"true": 0.99}}}

    res = jev_read_file("~/note.py", "find x", filter_by_chunk=False)
    assert "passed relevance check" in res


@patch("jev_mcp.routing_provider.RoutingProvider.evaluate_batch")
def test_read_file_allows_user_temp(mock_eval, tmp_path):
    mock_eval.return_value = {"file_eval": {"probabilities": {"true": 0.99}}}
    f = tmp_path / "note.py"
    f.write_text("def x(): pass")

    res = jev_read_file(str(f), "find x", filter_by_chunk=False)
    assert "passed relevance check" in res


@pytest.mark.parametrize(
    "path",
    [
        "/Users/test/.SSH/id_rsa",
        "/Users/test/.Aws/credentials",
        "/Users/test/.CONFIG/gh/hosts.yml",
        "/Users/test/.ZSH_HISTORY",
    ],
)
def test_is_safe_path_blocks_credential_names_in_any_letter_case(path):
    # macOS filesystems are case-insensitive, so ~/.SSH opens ~/.ssh.
    assert is_safe_path(path) is False


@pytest.mark.parametrize(
    "path",
    [
        "/Users/test/.docker/config.json",
        "/Users/test/.pgpass",
        "/Users/test/.pypirc",
        "/Users/test/.jev-mcp/client_secrets.json",
        "/Users/test/Library/Keychains/login.keychain-db",
        "/Users/test/Library/Application Support/Google/Chrome/Default/Cookies",
        "/Users/test/Library/Cookies/Cookies.binarycookies",
        "/Users/test/project/.env",
        "/Users/test/project/.env.production",
        "/Users/test/backup/id_ed25519",
        "/Users/test/.claude/.credentials.json",
    ],
)
def test_is_safe_path_blocks_credential_stores(path):
    assert is_safe_path(path) is False


@pytest.mark.parametrize(
    "path",
    [
        "/Users/test/project/.env.example",
        "/Users/test/project/environment.py",
        "/Users/test/project/library/keychains.py",
        "/Users/test/Library/Developer/notes.md",
    ],
)
def test_is_safe_path_still_allows_ordinary_project_files(path):
    assert is_safe_path(path) is True


# --- allowlist of roots -------------------------------------------------------

import os


@pytest.fixture
def no_extra_roots(monkeypatch):
    monkeypatch.delenv("JEV_ALLOWED_ROOTS", raising=False)
    monkeypatch.setenv("HOME", "/Users/test")
    monkeypatch.setattr(os, "getcwd", lambda: "/Users/test/project")


def test_a_path_outside_every_root_is_refused(no_extra_roots):
    assert is_safe_path("/Users/test/documents/notes.md") is False
    assert is_safe_path("/Volumes/backup/notes.md") is False


def test_the_working_directory_is_a_root(no_extra_roots):
    assert is_safe_path("/Users/test/project/src/app.py") is True
    assert is_safe_path("/Users/test/project") is True


def test_a_sibling_that_shares_the_root_name_prefix_is_refused(no_extra_roots):
    assert is_safe_path("/Users/test/project-evil/app.py") is False


def test_the_jev_directory_is_a_root(no_extra_roots):
    assert is_safe_path("/Users/test/.jev/router_config.json") is True
    assert is_safe_path("~/.jev/jev.log") is True


def test_the_jev_mcp_credential_directory_is_still_refused(no_extra_roots):
    assert is_safe_path("/Users/test/.jev-mcp/client_secrets.json") is False


def test_extra_roots_come_from_the_environment(no_extra_roots, monkeypatch):
    monkeypatch.setenv(
        "JEV_ALLOWED_ROOTS", os.pathsep.join(["/Users/test/work", "~/notes"])
    )

    assert is_safe_path("/Users/test/work/a.py") is True
    assert is_safe_path("/Users/test/notes/b.md") is True
    assert is_safe_path("/Users/test/other/c.py") is False


def test_the_denylist_still_wins_inside_an_allowed_root(no_extra_roots, monkeypatch):
    monkeypatch.setenv("JEV_ALLOWED_ROOTS", "/Users/test")

    assert is_safe_path("/Users/test/.ssh/id_rsa") is False
    assert is_safe_path("/Users/test/project/.env") is False


def test_an_explicit_root_may_sit_under_a_blocked_system_prefix(
    no_extra_roots, monkeypatch
):
    monkeypatch.setenv("JEV_ALLOWED_ROOTS", "/opt/projects")

    assert is_safe_path("/opt/projects/app/main.py") is True
    assert is_safe_path("/opt/homebrew/etc/passwd") is False


def test_the_filesystem_root_is_never_a_root(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/test")
    monkeypatch.setenv("JEV_ALLOWED_ROOTS", "/")
    monkeypatch.setattr(os, "getcwd", lambda: "/")

    assert is_safe_path("/Users/test/documents/notes.md") is False
    assert is_safe_path("/etc/passwd") is False


def test_roots_match_regardless_of_letter_case(no_extra_roots):
    # macOS filesystems are case-insensitive, like the name denylist assumes.
    assert is_safe_path("/users/TEST/Project/app.py") is True


def test_read_file_names_the_allowed_roots_when_refusing_an_outside_path(
    no_extra_roots,
):
    res = json.loads(jev_read_file("/Users/test/documents/notes.md", "any task"))

    assert res["status"] == "BLOCKED"
    assert "JEV_ALLOWED_ROOTS" in res["message"]
    assert "sensitive" not in res["message"].lower()


def test_scan_repo_names_the_allowed_roots_when_refusing_an_outside_path(
    no_extra_roots,
):
    res = json.loads(jev_scan_repo("/Users/test/documents", "any task"))

    assert res["status"] == "BLOCKED"
    assert "JEV_ALLOWED_ROOTS" in res["message"]
