import pytest
from fastapi import HTTPException
from jev_mcp.email_triage.auth import verify_auth_token

def test_verify_auth_token_static_key(monkeypatch):
    """Asserts that a correct static API key passes authentication."""
    monkeypatch.setenv("JEV_MCP_API_KEY", "my-super-secret")
    
    # Valid key
    assert verify_auth_token("Bearer my-super-secret") == {"type": "static_api_key"}
    
    # Invalid key
    with pytest.raises(HTTPException) as exc:
        verify_auth_token("Bearer wrong-key")
    assert exc.value.status_code == 401

def test_verify_auth_token_missing_header():
    with pytest.raises(HTTPException) as exc:
        verify_auth_token(None)
    assert exc.value.status_code == 401
    assert "Missing Authorization header" in exc.value.detail

def test_verify_auth_token_jwt(monkeypatch):
    """Asserts that a valid JWT token passes authentication if static key is not matched."""
    monkeypatch.setenv("JEV_MCP_API_KEY", "my-super-secret")
    
    def mock_decode(token, *args, **kwargs):
        if token == "valid-jwt":
            return {"sub": "user123", "permissions": ["triage:write"]}
        raise ValueError("Invalid token")
        
    # We patch the jwt module that will be used inside auth.py
    monkeypatch.setattr("jev_mcp.email_triage.auth.jwt_decode", mock_decode)
    
    # Valid JWT
    payload = verify_auth_token("Bearer valid-jwt")
    assert payload["type"] == "jwt"
    assert payload["permissions"] == ["triage:write"]
    
    # Invalid JWT
    with pytest.raises(HTTPException) as exc:
        verify_auth_token("Bearer invalid-jwt")
    assert exc.value.status_code == 401
