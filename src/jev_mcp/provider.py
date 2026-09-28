import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, Literal, Union

from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Polymorphic Question Definitions (Used by FastMCP for JSON Schema generation)
# -----------------------------------------------------------------------------


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    prompt: str
    key: str


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    prompt: str
    options: List[str]
    key: str


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    prompt: str
    labels: List[str]
    key: str


QuestionType = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]

# -----------------------------------------------------------------------------
# Dependency Inversion: The Provider Protocol
# -----------------------------------------------------------------------------


class JevProvider(Protocol):
    """
    Protocol defining the required interface for a Jev engine backend.
    This allows us to seamlessly swap between a local `open-jev` toy model,
    a production PyTorch model, or a remote API without changing the MCP server.
    """

    def evaluate_batch(
        self, state: Any, questions: List[QuestionType]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a single state against a batch of typed questions.

        Args:
            state: A JSON-serializable object (dict, list, str, etc.)
            questions: A list of polymorphic Question objects.

        Returns:
            A list of dictionary answers containing probabilities and confidence scores.
        """
        ...

    def check_token_limit(self, state: Any) -> int:
        """
        Measure the state payload size to enforce physical context limits.

        Returns:
            The number of tokens the state occupies.
        """
        ...
