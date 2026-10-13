import importlib.metadata
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp.doctor import collect_report, main, problems, render


@pytest.fixture
def urlopen_ok():
    """The engines answer GET /health at the HTTP boundary."""

    def _urlopen(request, timeout=None):
        response = MagicMock()
        response.read.return_value = b'{"status": "ok"}'
        context = MagicMock()
        context.__enter__.return_value = response
        return context

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        yield


@pytest.fixture
def urlopen_refused():
    """No engine answers at the HTTP boundary."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        yield


@pytest.fixture
def tmp_router_config():
    from jev_mcp import router_config

    router_config.write_json_atomic(
        router_config.ROUTER_CONFIG_PATH, {"buckets": {}, "rules": []}
    )
    return router_config.ROUTER_CONFIG_PATH


# --- the plan's Step 1 tests --------------------------------------------------


def test_an_engine_that_does_not_answer_is_reported_down_and_is_a_problem(
    urlopen_refused,
):
    report = collect_report(env={})
    assert report["engines"]["fast"]["status"] == "down"
    assert any("fast engine" in p for p in problems(report))


def test_both_engines_up_and_nothing_else_wrong_is_no_problem(
    urlopen_ok, tmp_router_config
):
    assert problems(collect_report(env={"JEV_ALLOWED_ROOTS": "/tmp/proj"})) == []


def test_jwks_without_an_audience_is_a_problem(urlopen_ok):
    report = collect_report(env={"JEV_MCP_JWKS_URL": "https://idp.example/jwks.json"})
    assert any("JEV_MCP_JWT_AUDIENCE" in p for p in problems(report))


def test_no_secret_value_appears_in_any_output(urlopen_ok, capsys):
    env = {
        "JEV_MCP_API_KEY": "s3cr3t-value-xyz",
        "JEV_MCP_JWT_SECRET": "another-s3cr3t-value-0123456789abcdef",
    }
    report = collect_report(env=env)
    text = render(report) + json.dumps(report)
    assert "s3cr3t" not in text
    assert report["webhook_auth"]["set"] == ["JEV_MCP_API_KEY", "JEV_MCP_JWT_SECRET"]


def test_a_home_working_directory_with_no_explicit_root_is_a_problem(
    urlopen_ok, monkeypatch
):
    monkeypatch.setenv("HOME", "/Users/test")
    monkeypatch.setattr(os, "getcwd", lambda: "/Users/test")
    monkeypatch.delenv("JEV_ALLOWED_ROOTS", raising=False)
    assert any("JEV_ALLOWED_ROOTS" in p for p in problems(collect_report(env={})))


def test_main_exits_one_when_there_is_a_problem(urlopen_refused, capsys):
    assert main([]) == 1


def test_json_output_parses(urlopen_ok, capsys):
    main(["--json"])
    json.loads(capsys.readouterr().out)


# --- the review-focus pin: no secret in --json either ------------------------


def test_json_output_never_contains_a_secret_value(urlopen_ok, capsys, monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "s3cr3t-value-xyz")
    main(["--json"])
    assert "s3cr3t" not in capsys.readouterr().out


# --- the remaining problem and report branches -------------------------------


def test_a_router_config_that_is_present_but_not_valid_json_is_a_problem(urlopen_ok):
    from jev_mcp import router_config

    path = Path(router_config.ROUTER_CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    report = collect_report(env={})

    assert report["router_config"]["exists"] is True
    assert report["router_config"]["valid_json"] is False
    assert any("not valid JSON" in p for p in problems(report))


def test_render_lists_the_problems_and_an_empty_auth_section(urlopen_refused):
    report = collect_report(env={})
    text = render(report)
    assert "problems" in text
    assert all(problem in text for problem in problems(report))
    assert "set : none" in text  # the auth section names variables, so never values


def test_an_unusable_port_value_falls_back_to_the_default(urlopen_ok):
    report = collect_report(env={"JEV_FAST_PORT": "not-a-number"})
    assert report["engines"]["fast"]["port"] == 8080


def test_versions_omit_a_distribution_that_is_not_installed(urlopen_ok):
    real_version = importlib.metadata.version

    def _missing(name, *args, **kwargs):
        if name == "torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name, *args, **kwargs)

    with patch("importlib.metadata.version", side_effect=_missing):
        report = collect_report(env={})

    assert "torch" not in report["versions"]
    assert "mcp" in report["versions"]


# --- the server dispatches the subcommand ------------------------------------


def test_server_dispatches_the_doctor_subcommand(urlopen_refused, capsys, monkeypatch):
    import sys

    from jev_mcp import server

    monkeypatch.setattr(sys, "argv", ["jev-mcp", "doctor", "--json"])
    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 1
    json.loads(capsys.readouterr().out)
