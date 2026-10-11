import hmac
import logging
import os
from functools import lru_cache

import jwt
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Fixed per key type, so a token can never pick its own algorithm (e.g. "none").
SHARED_SECRET_ALGORITHMS = ["HS256"]
JWKS_ALGORITHMS = ["RS256", "ES256"]


@lru_cache(maxsize=4)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    # One client per URL: it caches the provider's keys instead of fetching them on every request.
    return jwt.PyJWKClient(url)


def jwt_decode(token: str) -> dict:
    """Verify a JWT's signature, expiry, and (when configured) audience and issuer.

    The key comes from JEV_MCP_JWKS_URL (the identity provider's published keys) or,
    failing that, JEV_MCP_JWT_SECRET (a shared HS256 secret). With a JWKS URL the
    audience is required, not optional.
    """
    jwks_url = os.environ.get("JEV_MCP_JWKS_URL")
    secret = os.environ.get("JEV_MCP_JWT_SECRET")
    audience = os.environ.get("JEV_MCP_JWT_AUDIENCE")
    if jwks_url:
        # An identity provider signs tokens for every app it serves, so its signature
        # alone does not show the token was issued for this webhook.
        if not audience:
            raise ValueError("JEV_MCP_JWT_AUDIENCE must be set when JEV_MCP_JWKS_URL is used")
        key =_jwks_client(jwks_url).get_signing_key_from_jwt(token).key
        algorithms = JWKS_ALGORITHMS
    elif secret:
        key = secret
        algorithms = SHARED_SECRET_ALGORITHMS
    else:
        raise ValueError("No JWT verification key configured (JEV_MCP_JWKS_URL or JEV_MCP_JWT_SECRET)")

    return jwt.decode(
        token,
        key,
        algorithms=algorithms,
        audience=audience,
        issuer=os.environ.get("JEV_MCP_JWT_ISSUER"),
        options={"require": ["exp"], "verify_aud": bool(audience)},
    )


def verify_auth_token(auth_header: str | None) -> dict:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth_header.replace("Bearer ", "")
    static_key = os.environ.get("JEV_MCP_API_KEY")

    if static_key and hmac.compare_digest(token.encode(), static_key.encode()):
        return {"type": "static_api_key"}

    try:
        payload = jwt_decode(token)
        return {**payload, "type": "jwt"}
    except Exception as e:
        # The reason stays in the server log; the caller only learns that it failed.
        logger.warning("Rejected bearer token: %s", e)
        raise HTTPException(status_code=401, detail="Invalid authentication token")
