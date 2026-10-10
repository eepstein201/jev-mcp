import os
from fastapi import HTTPException

def jwt_decode(token: str):
    # This is a stub for an actual JWT library (e.g. PyJWT).
    # In a real implementation, it would fetch JWKS from the IdP and verify the signature.
    raise ValueError("Not implemented in stub")

def verify_auth_token(auth_header: str | None) -> dict:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
        
    token = auth_header.replace("Bearer ", "")
    static_key = os.environ.get("JEV_MCP_API_KEY")
    
    if static_key and token == static_key:
        return {"type": "static_api_key"}
        
    try:
        payload = jwt_decode(token)
        payload["type"] = "jwt"
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
