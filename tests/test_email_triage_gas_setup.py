import pytest
from unittest.mock import patch, MagicMock
from jev_mcp.email_triage.gas_setup import setup_gas_workflow

@patch("jev_mcp.email_triage.gas_setup.getpass.getpass")
@patch("jev_mcp.email_triage.gas_setup.input")
@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_prompts(mock_flow, mock_build, mock_input, mock_getpass):
    """Asserts the CLI uses getpass to securely prompt for the API key."""
    # Mock inputs
    mock_input.return_value = "https://test.ngrok.app/api/v1/triage/email"
    mock_getpass.return_value = "test-secret-key"
    
    # Mock OAuth flow
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = "mock_creds"
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance
    
    # Mock Google API Client
    mock_service = MagicMock()
    mock_build.return_value = mock_service
    # We setup the return value on the execution of the call
    mock_create = MagicMock()
    mock_create.execute.return_value = {"scriptId": "12345"}
    mock_service.projects().create.return_value = mock_create
    
    # Mock file reading
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "const WEBHOOK = 'TEMPLATE'; const API_KEY = 'TEMPLATE';"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Run
        script_url = setup_gas_workflow()
        
    assert "script.google.com" in script_url
    assert "12345" in script_url
    
    # Verify getpass was called
    mock_getpass.assert_called_once_with("Create a new secure password (API Key) to protect your local webhook. Type or paste it now (input will be hidden): ")
    
    # Verify Google API was called
    mock_service.projects().create.assert_called_once_with(body={'title': 'Jev-Mail Triage Worker'})
    mock_service.projects().updateContent.assert_called_once()
    
    # Verify the code payload injected the variables
    update_call_args = mock_service.projects().updateContent.call_args[1]
    files = update_call_args["body"]["files"]
    code_content = files[0]["source"]
    assert "test-secret-key" in code_content
    assert "https://test.ngrok.app/api/v1/triage/email" in code_content

@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow", None)
def test_gas_setup_missing_dependencies():
    """Asserts ImportError is raised if dependencies are missing."""
    with pytest.raises(ImportError, match="Please install google-api-python-client"):
        setup_gas_workflow()

@patch("jev_mcp.email_triage.gas_setup.getpass.getpass")
@patch("jev_mcp.email_triage.gas_setup.input")
@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_fallback_prepend(mock_flow, mock_build, mock_input, mock_getpass):
    """Asserts that code correctly prepends variables if replacement fails."""
    mock_input.return_value = "https://test.ngrok.app"
    mock_getpass.return_value = "test-secret"
    
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = "mock_creds"
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance
    
    mock_service = MagicMock()
    mock_build.return_value = mock_service
    mock_create = MagicMock()
    mock_create.execute.return_value = {"scriptId": "12345"}
    mock_service.projects().create.return_value = mock_create
    
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "const SOME_OTHER_CODE = true;"
        mock_open.return_value.__enter__.return_value = mock_file
        
        setup_gas_workflow()
        
    update_call_args = mock_service.projects().updateContent.call_args[1]
    code_content = update_call_args["body"]["files"][0]["source"]
    assert "const JEV_MCP_WEBHOOK_URL = 'https://test.ngrok.app';" in code_content
    assert "const JEV_MCP_API_KEY = 'test-secret';" in code_content

@patch("jev_mcp.email_triage.gas_setup.getpass.getpass")
@patch("jev_mcp.email_triage.gas_setup.input")
@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_file_not_found(mock_flow, mock_build, mock_input, mock_getpass):
    """Asserts that it handles FileNotFoundError when reading the template."""
    mock_input.return_value = "https://test.ngrok.app"
    mock_getpass.return_value = "test-secret"
    
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = "mock_creds"
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance
    
    mock_service = MagicMock()
    mock_build.return_value = mock_service
    mock_create = MagicMock()
    mock_create.execute.return_value = {"scriptId": "12345"}
    mock_service.projects().create.return_value = mock_create
    
    with patch("builtins.open", side_effect=FileNotFoundError):
        setup_gas_workflow()
        
    update_call_args = mock_service.projects().updateContent.call_args[1]
    code_content = update_call_args["body"]["files"][0]["source"]
    assert "test-secret" in code_content

@patch("jev_mcp.email_triage.gas_setup.build")
@patch("jev_mcp.email_triage.gas_setup.InstalledAppFlow")
def test_gas_setup_with_args(mock_flow, mock_build):
    """Asserts that the CLI skips prompts when arguments are provided."""
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = "mock_creds"
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance
    
    mock_service = MagicMock()
    mock_build.return_value = mock_service
    mock_create = MagicMock()
    mock_create.execute.return_value = {"scriptId": "12345"}
    mock_service.projects().create.return_value = mock_create
    
    with patch("builtins.open", side_effect=FileNotFoundError):
        setup_gas_workflow(webhook_url="https://args.ngrok.app", api_key="args-secret")
        
    update_call_args = mock_service.projects().updateContent.call_args[1]
    code_content = update_call_args["body"]["files"][0]["source"]
    assert "https://args.ngrok.app" in code_content
    assert "args-secret" in code_content

def test_gas_setup_import_error():
    import sys
    import importlib
    import jev_mcp.email_triage.gas_setup
    
    with patch.dict(sys.modules, {"google_auth_oauthlib.flow": None}):
        importlib.reload(jev_mcp.email_triage.gas_setup)
        assert jev_mcp.email_triage.gas_setup.InstalledAppFlow is None
        
    importlib.reload(jev_mcp.email_triage.gas_setup)
