"""Verify the webhook JWT path against a real (local) identity provider.

No mocks: a real RSA-2048 keypair, a real JWKS endpoint served over loopback
HTTP, a really-signed RS256 token, and the production PyJWKClient fetch. This
is the same shape as Auth0/Google — a real provider is just a JWKS URL plus
RS256 — so passing here means the IdP integration works, not a simulation of
it. (This is a verification test for existing behavior; it is expected to
pass on first run.)
"""
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from jev_mcp.email_triage.auth import verify_auth_token


def _b64url_uint(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


@pytest.fixture
def jwks_server():
    """A loopback identity provider publishing one RSA key."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = private_key.public_key().public_numbers()
    jwks = {
        "keys": [{
            "kty": "RSA", "alg": "RS256", "use": "sig", "kid": "test-key-1",
            "n": _b64url_uint(pub.n), "e": _b64url_uint(pub.e),
        }]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(jwks).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/jwks.json", private_key
    finally:
        server.shutdown()


@pytest.fixture
def jwks_env(monkeypatch, jwks_server):
    url, _ = jwks_server
    monkeypatch.setenv("JEV_MCP_JWKS_URL", url)
    monkeypatch.delenv("JEV_MCP_API_KEY", raising=False)
    monkeypatch.delenv("JEV_MCP_JWT_SECRET", raising=False)
    monkeypatch.setenv("JEV_MCP_JWT_AUDIENCE", "jev-webhook")
    monkeypatch.delenv("JEV_MCP_JWT_ISSUER", raising=False)
    return jwks_server


def test_accepts_real_rs256_token_from_jwks_idp(jwks_env):
    _, signing_key = jwks_env
    token = jwt.encode(
        {"sub": "user-1", "exp": int(time.time()) + 60, "aud": "jev-webhook"},
        signing_key,
        algorithm="RS256",
        headers={"kid": "test-key-1"},
    )
    claims = verify_auth_token(f"Bearer {token}")
    assert claims["type"] == "jwt"
    assert claims["sub"] == "user-1"


def test_rejects_token_signed_by_a_different_key(jwks_env):
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode(
        {"sub": "attacker", "exp": int(time.time()) + 60, "aud": "jev-webhook"},
        attacker_key,
        algorithm="RS256",
        headers={"kid": "test-key-1"},
    )
    with pytest.raises(Exception) as exc_info:
        verify_auth_token(f"Bearer {token}")
    assert getattr(exc_info.value, "status_code", None) == 401


def test_rejects_expired_token(jwks_env):
    _, signing_key = jwks_env
    token = jwt.encode(
        {"sub": "user-1", "exp": int(time.time()) - 10, "aud": "jev-webhook"},
        signing_key,
        algorithm="RS256",
        headers={"kid": "test-key-1"},
    )
    with pytest.raises(Exception) as exc_info:
        verify_auth_token(f"Bearer {token}")
    assert getattr(exc_info.value, "status_code", None) == 401


def test_enforces_audience_when_configured(jwks_env, monkeypatch):
    _, signing_key = jwks_env
    monkeypatch.setenv("JEV_MCP_JWT_AUDIENCE", "jev-webhook")
    token = jwt.encode(
        {"sub": "user-1", "exp": int(time.time()) + 60, "aud": "someone-else"},
        signing_key,
        algorithm="RS256",
        headers={"kid": "test-key-1"},
    )
    with pytest.raises(Exception) as exc_info:
        verify_auth_token(f"Bearer {token}")
    assert getattr(exc_info.value, "status_code", None) == 401


def test_jwks_mode_refuses_tokens_when_no_audience_is_configured(jwks_env, monkeypatch):
    # A shared identity provider signs tokens for every app; without an expected
    # audience, a token issued to someone else's app would be accepted here.
    _, signing_key = jwks_env
    monkeypatch.delenv("JEV_MCP_JWT_AUDIENCE", raising=False)
    token = jwt.encode(
        {"sub": "user-1", "exp": int(time.time()) + 60, "aud": "someone-elses-app"},
        signing_key,
        algorithm="RS256",
        headers={"kid": "test-key-1"},
    )
    with pytest.raises(Exception) as exc_info:
        verify_auth_token(f"Bearer {token}")
    assert getattr(exc_info.value, "status_code", None) == 401
