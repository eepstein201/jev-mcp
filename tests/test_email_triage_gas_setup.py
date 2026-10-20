import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp.email_triage.gas_setup import setup_gas_workflow

TEMPLATE_PATH = Path(__file__).parent.parent / "examples" / "gas_triage.js"


def _mock_google(mock_flow, mock_build):
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = "mock_creds"
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance

    mock_service = MagicMock()
    mock_build.return_value = mock_service
    mock_service.projects().create.return_value.execute.return_value = {
        "scriptId": "12345"
    }
    return mock_service


def _uploaded_source(mock_service):
    call = mock_service.projects().updateContent.call_args[1]
    return call["body"]["files"][0]["source"]


@patch("jev_mcp.email_triage.gas_setup.input")
@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_prompts_for_the_webhook_url_only(
    mock_flow, mock_exists, mock_build, mock_input
):
    mock_input.return_value = "https://test.ngrok.app/api/v1/triage/email"
    mock_service = _mock_google(mock_flow, mock_build)

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "const JEV_MCP_WEBHOOK_URL = 'TEMPLATE';"
        mock_open.return_value.__enter__.return_value = mock_file

        script_url = setup_gas_workflow()

    assert "script.google.com" in script_url
    assert "12345" in script_url
    mock_input.assert_called_once()
    mock_service.projects().create.assert_called_once_with(
        body={"title": "Jev-Mail Triage Worker"}
    )
    mock_service.projects().updateContent.assert_called_once()
    assert "https://test.ngrok.app/api/v1/triage/email" in _uploaded_source(
        mock_service
    )


@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow", None)
def test_gas_setup_missing_dependencies():
    """Asserts ImportError is raised if dependencies are missing."""
    with pytest.raises(ImportError, match="Please install google-api-python-client"):
        setup_gas_workflow()


@patch("jev_mcp.email_triage.gas_setup.input")
@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_fallback_prepends_the_webhook_url(
    mock_flow, mock_exists, mock_build, mock_input
):
    """Asserts that the URL is prepended when the template has no placeholder."""
    mock_input.return_value = "https://test.ngrok.app"
    mock_service = _mock_google(mock_flow, mock_build)

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "const SOME_OTHER_CODE = true;"
        mock_open.return_value.__enter__.return_value = mock_file

        setup_gas_workflow()

    code_content = _uploaded_source(mock_service)
    assert 'const JEV_MCP_WEBHOOK_URL = "https://test.ngrok.app";' in code_content
    assert "JEV_MCP_API_KEY" not in code_content


@patch("jev_mcp.email_triage.gas_setup.input")
@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_file_not_found(mock_flow, mock_exists, mock_build, mock_input):
    """Asserts that it handles FileNotFoundError when reading the template."""
    mock_input.return_value = "https://test.ngrok.app"
    mock_service = _mock_google(mock_flow, mock_build)

    with patch("builtins.open", side_effect=FileNotFoundError):
        setup_gas_workflow()

    assert "https://test.ngrok.app" in _uploaded_source(mock_service)


@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=False)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_missing_client_secrets_names_where_to_put_the_file(mock_flow, mock_exists):
    with pytest.raises(FileNotFoundError, match="client_secrets.json"):
        setup_gas_workflow(webhook_url="https://test.ngrok.app")


@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_with_args(mock_flow, mock_exists, mock_build):
    """Asserts that the CLI skips the prompt when the URL is provided."""
    mock_service = _mock_google(mock_flow, mock_build)

    with patch("builtins.open", side_effect=FileNotFoundError):
        setup_gas_workflow(webhook_url="https://args.ngrok.app")

    assert "https://args.ngrok.app" in _uploaded_source(mock_service)


def test_gas_setup_import_error():
    import importlib
    import sys

    import jev_mcp.email_triage.gas_setup

    with patch.dict(sys.modules, {"google_auth_oauthlib.flow": None}):
        importlib.reload(jev_mcp.email_triage.gas_setup)
        assert jev_mcp.email_triage.gas_setup.InstalledAppFlow is None

    importlib.reload(jev_mcp.email_triage.gas_setup)


@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_writes_the_url_as_an_escaped_js_string_literal(
    mock_flow, mock_exists, mock_build
):
    """A quote in the URL must not end the JavaScript string it is written into."""
    mock_service = _mock_google(mock_flow, mock_build)
    hostile_url = (
        "https://hook.example/a'; UrlFetchApp.fetch('https://evil.example'); '"
    )

    with patch("builtins.open", side_effect=FileNotFoundError):
        setup_gas_workflow(webhook_url=hostile_url)

    code = _uploaded_source(mock_service)
    assert json.dumps(hostile_url) in code
    assert f"'{hostile_url}'" not in code


@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_the_webhook_key_never_enters_the_uploaded_source(
    mock_flow, mock_exists, mock_build, monkeypatch
):
    monkeypatch.setenv("JEV_MCP_API_KEY", "env-key-must-not-be-uploaded")
    mock_service = _mock_google(mock_flow, mock_build)

    setup_gas_workflow(webhook_url="https://hook.example/api")

    body = mock_service.projects().updateContent.call_args[1]["body"]
    assert "env-key-must-not-be-uploaded" not in json.dumps(body)


@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.Path.exists", return_value=True)
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_tells_the_user_to_set_the_script_property(
    mock_flow, mock_exists, mock_build, capsys
):
    _mock_google(mock_flow, mock_build)

    setup_gas_workflow(webhook_url="https://hook.example/api")

    out = capsys.readouterr().out
    assert "Script Properties" in out
    assert "JEV_MCP_API_KEY" in out


def test_gas_setup_no_longer_accepts_an_api_key():
    with pytest.raises(TypeError):
        setup_gas_workflow(webhook_url="https://hook.example/api", api_key="k")


def test_the_template_reads_the_key_from_script_properties_not_source():
    source = TEMPLATE_PATH.read_text()

    assert "PropertiesService.getScriptProperties()" in source
    assert "const JEV_MCP_API_KEY" not in source


def test_the_template_sends_the_attempt_header_and_retries_on_503():
    source = TEMPLATE_PATH.read_text()

    assert "X-Triage-Attempt" in source
    assert "MAX_DEGRADED_ATTEMPTS = 3" in source
    assert "503" in source
    assert "Utilities.sleep" in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    "codes, expected_calls, expected_attempts, marked_read",
    [
        ([200], 1, [1], True),
        ([503, 200], 2, [1, 2], True),
        ([503, 503, 200], 3, [1, 2, 3], True),
        ([503, 503, 503, 200], 3, [1, 2, 3], False),
        ([500], 1, [1], False),
    ],
)
def test_the_template_retries_a_503_up_to_the_cap_and_stops_on_other_answers(
    tmp_path, codes, expected_calls, expected_attempts, marked_read
):
    harness = tmp_path / "harness.js"
    harness.write_text(
        f"""
const vm = require('vm');
const codes = {json.dumps(codes)};
const attempts = [];
let sleeps = 0;
let marked = false;
const message = {{
  isUnread: () => true, getSubject: () => 's', getFrom: () => 'f',
  getPlainBody: () => 'b', getId: () => 'id', markRead: () => {{ marked = true; }},
}};
const sandbox = {{
  Logger: {{ log: () => {{}} }},
  Utilities: {{ sleep: () => {{ sleeps += 1; }} }},
  PropertiesService: {{ getScriptProperties: () => ({{ getProperty: () => '' }}) }},
  GmailApp: {{ search: () => [{{ getMessages: () => [message] }}] }},
  UrlFetchApp: {{ fetch: (url, options) => {{
    attempts.push(Number(options.headers['X-Triage-Attempt']));
    const code = codes[Math.min(attempts.length, codes.length) - 1];
    return {{ getResponseCode: () => code, getContentText: () => '' }};
  }} }},
}};
vm.createContext(sandbox);
vm.runInContext(require('fs').readFileSync(process.argv[2], 'utf8')
  .replace("'TEMPLATE'", "'https://hook.example/api'"), sandbox);
sandbox.processUnreadEmails();
console.log(JSON.stringify({{ attempts, sleeps, marked }}));
"""
    )

    out = subprocess.run(
        ["node", str(harness), str(TEMPLATE_PATH)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    seen = json.loads(out)
    assert seen["attempts"] == expected_attempts
    assert len(seen["attempts"]) == expected_calls
    assert seen["sleeps"] == expected_calls - 1
    assert seen["marked"] is marked_read


def test_the_template_attempt_cap_matches_the_server():
    from jev_mcp.email_triage.api import MAX_DEGRADED_ATTEMPTS

    assert (
        f"const MAX_DEGRADED_ATTEMPTS = {MAX_DEGRADED_ATTEMPTS};"
        in TEMPLATE_PATH.read_text()
    )
