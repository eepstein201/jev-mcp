import json
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from jev_mcp.email_triage.auth import verify_auth_token
from jev_mcp.email_triage.mcp_tool import triage_email_content
from jev_mcp.email_triage.slack import build_escalation_message, build_hitl_message
from jev_mcp.provider import EngineUnavailableError

app = FastAPI()

# A long body is compacted chunk by chunk, one engine call each, so its length bounds the work one request can cause.
MAX_BODY_CHARS = 1_000_000
MAX_SUBJECT_CHARS = 2_000
MAX_SENDER_CHARS = 1_000
MAX_ID_CHARS = 200

class EmailPayload(BaseModel):
    context_id: str = Field(max_length=MAX_ID_CHARS)
    subject: str = Field("", max_length=MAX_SUBJECT_CHARS)
    sender: str = Field("", max_length=MAX_SENDER_CHARS)
    body: str = Field("", max_length=MAX_BODY_CHARS)
    id: str = Field("unknown", max_length=MAX_ID_CHARS)

def get_auth(authorization: Optional[str] = Header(None)):
    return verify_auth_token(authorization)

@app.post("/api/v1/triage/email")
def triage_email(payload: EmailPayload, auth: dict = Depends(get_auth)):
    try:
        result_str = triage_email_content(
            context_id=payload.context_id,
            subject=payload.subject,
            sender=payload.sender,
            body=payload.body
        )
    except EngineUnavailableError:
        # Without an engine there is no decision; a default one would look like a real triage.
        raise HTTPException(status_code=503, detail="The triage engine is not running.")
    
    result = json.loads(result_str)
    
    if result.get("status") == "needs_config":
        raise HTTPException(status_code=400, detail=result["message"])
        
    decision = result.get("decision")
    suggested = result.get("suggested_routing")
    
    slack_payload = None
    if suggested == "escalate_to_slack":
        slack_payload = build_escalation_message(payload.model_dump(), result)
    elif suggested == "hitl_slack_block":
        slack_payload = build_hitl_message(payload.model_dump(), result)
        
    result["slack_payload"] = slack_payload
    return result
