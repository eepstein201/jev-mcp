"""The email triage tools as MCP tools: JSON results, nothing on stdout, no prompts."""

import json

import pytest

from jev_mcp import server
from jev_mcp.provider import EngineUnavailableError
from jev_mcp.tools import triage


@pytest.fixture
def setup_calls(monkeypatch):
    calls = []

    def fake_setup(webhook_url, api_key):
        print("Welcome to the wizard")  # the real wizard prints progress
        calls.append({"webhook_url": webhook_url, "api_key": api_key})
        return "https://script.google.com/d/abc/edit"

    monkeypatch.setattr(triage, "setup_gas_workflow", fake_setup)
    return calls


def test_setup_tool_returns_json_and_keeps_stdout_clean(setup_calls, capsys):
    res = json.loads(
        server.tool_setup_gas_workflow("https://hook.example/api", "secret")
    )

    assert res == {"status": "SUCCESS", "url": "https://script.google.com/d/abc/edit"}
    assert capsys.readouterr().out == ""  # stdout is the JSON-RPC stream


def test_setup_tool_refuses_to_prompt_when_the_webhook_url_is_missing(setup_calls):
    res = json.loads(server.tool_setup_gas_workflow(None, "secret"))

    assert res["status"] == "ERROR"
    assert "webhook_url" in res["message"]
    assert setup_calls == []


def test_setup_tool_rejects_a_webhook_url_that_is_not_http(setup_calls):
    res = json.loads(server.tool_setup_gas_workflow("javascript:alert(1)", "secret"))

    assert res["status"] == "ERROR"
    assert setup_calls == []


def test_setup_tool_takes_the_api_key_from_the_environment(setup_calls, monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "env-secret")

    res = json.loads(server.tool_setup_gas_workflow("https://hook.example/api", None))

    assert res["status"] == "SUCCESS"
    assert setup_calls[0]["api_key"] == "env-secret"


def test_setup_tool_refuses_to_prompt_when_no_api_key_is_available(
    setup_calls, monkeypatch
):
    monkeypatch.delenv("JEV_MCP_API_KEY", raising=False)

    res = json.loads(server.tool_setup_gas_workflow("https://hook.example/api", None))

    assert res["status"] == "ERROR"
    assert "JEV_MCP_API_KEY" in res["message"]
    assert setup_calls == []


def test_setup_tool_reports_a_failure_as_json(monkeypatch):
    def failing_setup(webhook_url, api_key):
        raise FileNotFoundError("Missing OAuth credentials")

    monkeypatch.setattr(triage, "setup_gas_workflow", failing_setup)

    res = json.loads(
        server.tool_setup_gas_workflow("https://hook.example/api", "secret")
    )

    assert res == {"status": "ERROR", "message": "Missing OAuth credentials"}


def test_triage_tool_reports_an_unavailable_engine_as_json(monkeypatch):
    def down(*args):
        raise EngineUnavailableError("no engine answered")

    monkeypatch.setattr(triage, "triage_email_content", down)

    res = json.loads(
        server.tool_triage_email_content("ctx", "subject", "sender", "body")
    )

    assert res["status"] == "ERROR"
    assert "no engine answered" in res["message"]


def test_configure_labels_tool_reports_a_failure_as_json(monkeypatch):
    def broken(context_id, labels):
        raise ValueError("config is corrupt")

    monkeypatch.setattr(triage, "configure_triage_labels", broken)

    res = json.loads(server.tool_configure_triage_labels("ctx", ["a"]))

    assert res == {"status": "ERROR", "message": "config is corrupt"}
