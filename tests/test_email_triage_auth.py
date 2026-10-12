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


# Obviously fake signing secrets, long enough for HS256 (32+ bytes).
FAKE_SECRET = "test-only-signing-secret-not-a-real-key-0001"
OTHER_FAKE_SECRET = "test-only-signing-secret-not-a-real-key-0002"


def _token(secret=FAKE_SECRET, lifetime=300, **claims):
    import time
    import jwt

    return jwt.encode(
        {"sub": "user123", "exp": int(time.time()) + lifetime, **claims},
        secret,
        algorithm="HS256",
    )


@pytest.fixture
def jwt_env(monkeypatch):
    for name in (
        "JEV_MCP_API_KEY",
        "JEV_MCP_JWT_SECRET",
        "JEV_MCP_JWT_AUDIENCE",
        "JEV_MCP_JWT_ISSUER",
        "JEV_MCP_JWKS_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEV_MCP_JWT_SECRET", FAKE_SECRET)
    return monkeypatch


def _rejected(token):
    with pytest.raises(HTTPException) as exc:
        verify_auth_token(f"Bearer {token}")
    return exc.value.status_code


def test_jwt_signed_with_the_configured_secret_is_accepted(jwt_env):
    payload = verify_auth_token(f"Bearer {_token()}")

    assert payload["type"] == "jwt"
    assert payload["sub"] == "user123"


def test_jwt_signed_with_another_secret_is_rejected(jwt_env):
    assert _rejected(_token(secret=OTHER_FAKE_SECRET)) == 401


def test_expired_jwt_is_rejected(jwt_env):
    assert _rejected(_token(lifetime=-60)) == 401


def test_jwt_without_an_expiry_is_rejected(jwt_env):
    import jwt

    assert (
        _rejected(jwt.encode({"sub": "user123"}, FAKE_SECRET, algorithm="HS256")) == 401
    )


def test_unsigned_jwt_is_rejected(jwt_env):
    import jwt

    assert (
        _rejected(
            jwt.encode({"sub": "user123", "exp": 9999999999}, None, algorithm="none")
        )
        == 401
    )


def test_jwt_audience_and_issuer_are_enforced_when_configured(jwt_env):
    jwt_env.setenv("JEV_MCP_JWT_AUDIENCE", "jev-triage")
    jwt_env.setenv("JEV_MCP_JWT_ISSUER", "https://idp.example.test/")

    good = _token(aud="jev-triage", iss="https://idp.example.test/")

    assert verify_auth_token(f"Bearer {good}")["type"] == "jwt"
    assert _rejected(_token(aud="someone-else", iss="https://idp.example.test/")) == 401
    assert _rejected(_token(aud="jev-triage", iss="https://evil.example.test/")) == 401


def test_jwt_is_rejected_when_no_verification_key_is_configured(jwt_env):
    jwt_env.delenv("JEV_MCP_JWT_SECRET")

    assert _rejected(_token()) == 401


def test_jwt_is_verified_against_the_identity_providers_jwks(jwt_env):
    # RS256 token checked against the signing key the JWKS client returns for it.
    import time
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from unittest.mock import MagicMock, patch

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode(
        {"sub": "idp-user", "exp": int(time.time()) + 300, "aud": "jev-triage"},
        private_key,
        algorithm="RS256",
    )
    jwt_env.delenv("JEV_MCP_JWT_SECRET")
    jwt_env.setenv("JEV_MCP_JWKS_URL", "https://idp.example.test/.well-known/jwks.json")
    jwt_env.setenv("JEV_MCP_JWT_AUDIENCE", "jev-triage")
    signing_key = MagicMock(key=private_key.public_key())

    with patch("jwt.PyJWKClient") as client_cls:
        client_cls.return_value.get_signing_key_from_jwt.return_value = signing_key
        payload = verify_auth_token(f"Bearer {token}")

    assert payload["sub"] == "idp-user"
    client_cls.assert_called_once_with("https://idp.example.test/.well-known/jwks.json")


def test_static_api_key_is_compared_in_constant_time(jwt_env):
    from unittest.mock import patch

    jwt_env.setenv("JEV_MCP_API_KEY", "fake-static-key-for-tests")

    with patch("hmac.compare_digest", return_value=True) as compare:
        assert verify_auth_token("Bearer fake-static-key-for-tests") == {
            "type": "static_api_key"
        }

    compare.assert_called_once()


def test_only_the_leading_bearer_prefix_is_stripped(monkeypatch):
    # Stripping every "Bearer " would corrupt a key that happens to contain the text.
    monkeypatch.setenv("JEV_MCP_API_KEY", "tokBearer rest")

    assert verify_auth_token("Bearer tokBearer rest") == {"type": "static_api_key"}


def test_a_shared_secret_under_32_bytes_is_refused(jwt_env):
    short = "short-secret"
    jwt_env.setenv("JEV_MCP_JWT_SECRET", short)

    assert _rejected(_token(secret=short)) == 401
