import json
import math
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional

from jev_mcp.email_triage.auth import verify_auth_token
from jev_mcp.email_triage.mcp_tool import triage_email_content
from jev_mcp.email_triage.rate_limit import SlidingWindowLimiter, limit_from_env
from jev_mcp.email_triage.slack import build_escalation_message, build_hitl_message
from jev_mcp.provider import EngineUnavailableError

# Bounds what the server buffers before validation. Generous on purpose: 1M characters is at most
# 1 MB of ASCII but several MB when a client escapes non-ASCII text as \uXXXX.
MAX_REQUEST_BYTES = 8 * 1024 * 1024


class BodySizeLimitMiddleware:
    """Answers 413 once a request body passes max_bytes, whether declared or streamed."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send) -> None:
        body = b'{"detail":"Request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if (
            declared is not None
            and declared.isdigit()
            and int(declared) > self.max_bytes
        ):
            await self._reject(send)
            return

        received = 0
        rejected = False

        async def counting_receive():
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # Answer here and report a disconnect: FastAPI turns an exception raised
                    # from receive into a 400, and would then try to answer a second time.
                    if not rejected:
                        rejected = True
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not rejected:
                await send(message)

        await self.app(scope, counting_receive, guarded_send)


app = FastAPI()
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

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


# Per client address. Behind a reverse proxy every caller arrives from the proxy's address and
# shares one allowance.
limiter = SlidingWindowLimiter(limit=limit_from_env())


async def enforce_rate_limit(request: Request) -> None:
    # async so a refusal is answered from the event loop, not queued behind triage calls in the
    # worker thread pool.
    client_address = request.client.host if request.client else "unknown"
    retry_after = limiter.check(client_address)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )


# Route-level dependencies run before the handler's own, so the limit is checked before auth.
@app.post("/api/v1/triage/email", dependencies=[Depends(enforce_rate_limit)])
def triage_email(payload: EmailPayload, auth: dict = Depends(get_auth)):
    try:
        result_str = triage_email_content(
            context_id=payload.context_id,
            subject=payload.subject,
            sender=payload.sender,
            body=payload.body,
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
