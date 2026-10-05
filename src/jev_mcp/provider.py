import json
import logging
from typing import Any, Dict, List, Literal, Union
from abc import ABC, abstractmethod
import urllib.request

from pydantic import BaseModel

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
        return [{"id": "true", "description": "True"}, {"id": "false", "description": "False"}]

class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    prompt: str
    options: List[str]
    key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the SystemOne question payload shape."""
        return {"type": "choice", "instructions": self.prompt, "criteria": {opt: "" for opt in self.options}}

    def to_llm_options(self) -> List[Dict[str, str]]:
        return [{"id": str(idx), "description": str(opt)} for idx, opt in enumerate(self.options)]

class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    prompt: str
    labels: List[str]
    key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the SystemOne question payload shape."""
        return {"type": "score", "instructions": self.prompt, "criteria": self.labels}

    def to_llm_options(self) -> List[Dict[str, str]]:
        return [{"id": str(idx), "description": str(opt)} for idx, opt in enumerate(self.labels)]

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

def post_json(url: str, payload: dict, timeout: int = 45) -> Any:
    """POST JSON to a local daemon and return the parsed response.

    Refuses non-local hosts so a tainted env var cannot redirect engine
    traffic (and question content) off-box.
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
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())
