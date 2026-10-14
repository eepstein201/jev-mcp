import io
import json
import logging
from typing import Any, Dict, List, Literal, Union
from abc import ABC, abstractmethod
import urllib.error
import urllib.request

from pydantic import BaseModel


class EngineUnavailableError(RuntimeError):
    """No local engine produced an answer, so there is no score to act on."""


# Polymorphic question definitions (drive MCP JSON-schema generation and
# provider serialization).


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    prompt: str
    key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the SystemOne question payload shape."""
        return {"type": "noul", "instructions": self.prompt}

    def to_llm_options(self) -> List[Dict[str, str]]:
        return [
            {"id": "true", "description": "True"},
            {"id": "false", "description": "False"},
        ]


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    prompt: str
    options: List[str]
    key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the SystemOne question payload shape."""
        return {
            "type": "choice",
            "instructions": self.prompt,
            "criteria": {opt: "" for opt in self.options},
        }

    def to_llm_options(self) -> List[Dict[str, str]]:
        return [
            {"id": str(idx), "description": str(opt)}
            for idx, opt in enumerate(self.options)
        ]


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    prompt: str
    labels: List[str]
    key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the SystemOne question payload shape."""
        return {"type": "score", "instructions": self.prompt, "criteria": self.labels}

    def to_llm_options(self) -> List[Dict[str, str]]:
        return [
            {"id": str(idx), "description": str(opt)}
            for idx, opt in enumerate(self.labels)
        ]


QuestionType = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]

# Dependency inversion: every engine backend (Kev SystemOne, MLX logprob
# daemon, hybrid router) subclasses this contract so the MCP server never
# depends on a concrete engine.


class JevProvider(ABC):
    @abstractmethod
    def evaluate_batch(
        self, state: Any, questions: List[QuestionType]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate one state against a batch of questions.

        Returns a dict keyed by question key, each holding at least a
        "probabilities" mapping plus the typed answer field (noul/choice/score).
        """
        pass

    def check_token_limit(self, state: Any) -> int:
        """Rough token estimate (~4 chars/token) used to enforce context limits."""
        state_str = json.dumps(state) if not isinstance(state, str) else state
        return len(state_str) // 4

    def evaluate_dataset(self, dataset: List[dict], question: Any) -> List[dict]:
        """Evaluate one question against every row's state (used by calibration).

        A row may carry its own "question" string, which replaces the prompt for that row only.
        """
        return [
            {
                "expected": row.get("expected"),
                "probabilities": self.evaluate_batch(
                    row.get("state", {}), [_question_for_row(row, question)]
                )
                .get(question.key, {})
                .get("probabilities", {}),
            }
            for row in dataset
        ]


def _question_for_row(row: dict, question: Any) -> Any:
    row_prompt = row.get("question")
    if isinstance(row_prompt, str) and row_prompt:
        return question.model_copy(update={"prompt": row_prompt})
    return question


# How much of an error response body is kept in the raised error.
ERROR_BODY_MAX_CHARS = 500


def _http_error_with_body(error: urllib.error.HTTPError) -> urllib.error.HTTPError:
    """The same HTTPError (same type, same code) with the response body in its reason.

    A daemon explains a 400 in the body (`exceed_context_size_error`); the bare
    "Bad Request" would hide it.
    """
    try:
        raw = error.read()
    except Exception:
        raw = b""
    body = raw.decode("utf-8", errors="replace")[:ERROR_BODY_MAX_CHARS]
    reason = f"{error.reason}: {body}" if body else str(error.reason)
    return urllib.error.HTTPError(
        error.filename, error.code, reason, error.headers, io.BytesIO(raw)
    )


def post_json(url: str, payload: dict, timeout: int = 45) -> Any:
    """POST JSON to a local daemon and return the parsed response.

    Refuses non-local hosts so a tainted env var cannot redirect engine
    traffic (and question content) off-box. An HTTP error keeps its status code
    and the start of its response body.
    """
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or ""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(f"post_json only calls local daemons, got host {host!r}")

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise _http_error_with_body(error) from error


def escalation_failures(results: Dict[str, Any]) -> List[str]:
    """The keys of questions the smart engine was asked but did not answer."""
    return [
        key
        for key, result in results.items()
        if isinstance(result, dict) and result.get("escalation_failed")
    ]


def mark_degraded(
    payload: Dict[str, Any], failed: List[str], keep_status: bool = False
) -> Dict[str, Any]:
    """A copy of `payload` that says it is degraded; `payload` itself when nothing failed.

    DEGRADED replaces a SUCCESS status (the default). A payload whose status carries
    its own meaning (BLOCKED, EVALUATED, ...) passes keep_status=True and keeps it,
    flagged by `degraded`/`escalation_failed` instead.
    """
    if not failed:
        return payload
    degraded: Dict[str, Any] = {
        **payload,
        "degraded": True,
        "escalation_failed": list(failed),
    }
    if not keep_status:
        degraded["status"] = "DEGRADED"
    return degraded
