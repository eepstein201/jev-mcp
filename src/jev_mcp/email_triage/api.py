import json
import threading
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from jev_mcp.email_triage.auth import verify_auth_token
from jev_mcp.email_triage.mcp_tool import triage_email_content
from jev_mcp.email_triage.slack import build_escalation_message, build_hitl_message

app = FastAPI()
triage_lock = threading.Lock()


class EmailPayload(BaseModel):
    context_id: str
    subject: str = ""
    sender: str = ""
    body: str = ""
    id: str = "unknown"

def get_auth(authorization: Optional[str] = Header(None)):
    return verify_auth_token(authorization)

@app.post("/api/v1/triage/email")
def triage_email(payload: EmailPayload, auth: dict = Depends(get_auth)):
    with triage_lock:
        result_str = triage_email_content(
            context_id=payload.context_id,
            subject=payload.subject,
            sender=payload.sender,
            body=payload.body
        )
    
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
