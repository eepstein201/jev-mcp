import json
import logging
from typing import Any, List, Dict

from jev_mcp.provider import (
    EngineUnavailableError,
    JevProvider,
    QuestionType,
    NoulQuestion,
    ScoreQuestion,
)
from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.kev_provider import KevProvider
import os

logger = logging.getLogger(__name__)

SERVED_BY_FAST = "fast"
SERVED_BY_SMART = "smart"


class RoutingProvider(JevProvider):
    """
    Implements a Hybrid Complexity Router.
    Routes queries between the Fast 0.5B model and the Smart 7B model
    based on logit confidence scores and context complexity.
    """

    def __init__(self):
        fast_port = os.getenv("JEV_FAST_PORT", "8080")
        smart_port = os.getenv("JEV_SMART_PORT", "8081")

        # Determine which provider to use for the fast tier
        fast_engine = os.getenv(
            "JEV_FAST_ENGINE", "kev"
        )  # Default to kev for backwards compatibility with our new script

        if fast_engine == "kev":
            self.fast_provider = KevProvider(
                base_url=f"http://127.0.0.1:{fast_port}/v1/systemone"
            )
        else:
            self.fast_provider = DaemonProvider(
                base_url=f"http://127.0.0.1:{fast_port}/v1/chat/completions"
            )

        self.smart_provider = DaemonProvider(
            base_url=f"http://127.0.0.1:{smart_port}/v1/chat/completions"
        )
        # Large states are routed to the smart engine, so its limit applies too.
        self.max_tokens = min(
            self.fast_provider.max_tokens, self.smart_provider.max_tokens
        )
        self.current_model_id = "hybrid_router"

    def check_token_limit(self, state: Any) -> int:
        return self.fast_provider.check_token_limit(state)

    def evaluate_batch(
        self, state: Any, questions: List[QuestionType]
    ) -> Dict[str, Dict[str, Any]]:
        # 1. Rule-Based Complexity: Check state size
        tokens = self.check_token_limit(state)
        if tokens > 4000:
            logger.info(
                f"State size {tokens} exceeds fast-routing limit. Bypassing 0.5B and routing directly to 7B."
            )
            self.current_model_id = self.smart_provider.current_model_id
            smart_only = self.smart_provider.evaluate_batch(state, questions)
            return self._require_answers(
                questions,
                {
                    key: {**res, "served_by": SERVED_BY_SMART}
                    for key, res in smart_only.items()
                },
            )

        logger.info(f"Evaluating {len(questions)} questions on Fast 0.5B model...")
        fast_results = self.fast_provider.evaluate_batch(state, questions)

        uncertain_questions = []
        for q in questions:
            res = fast_results.get(q.key, {})
            probs = res.get("probabilities", {})
            if not probs:
                uncertain_questions.append(q)
                continue

            # Log-Sum-Exp Absolute Confidence Gate Triggered
            if all(v == 0.0 for v in probs.values()):
                logger.info(
                    f"Fast model triggered Absolute Confidence Gate for question '{q.key}'. Will escalate."
                )
                uncertain_questions.append(q)
                continue

            max_prob = max(probs.values())
            if max_prob < 0.85:
                logger.info(
                    f"Fast model confidence {max_prob * 100:.1f}% < 85% for question '{q.key}'. Will escalate."
                )
                uncertain_questions.append(q)

        if not uncertain_questions:
            self.current_model_id = self.fast_provider.current_model_id
            return {
                key: {**res, "served_by": SERVED_BY_FAST}
                for key, res in fast_results.items()
            }

        logger.info(
            f"Escalating {len(uncertain_questions)} uncertain questions to Smart 7B model..."
        )
        smart_results = self.smart_provider.evaluate_batch(state, uncertain_questions)

        # Merge results. A question the smart engine did not answer keeps its fast answer,
        # marked so callers (and tool statuses) can tell it was never second-checked.
        merged: Dict[str, Dict[str, Any]] = {
            key: {**res, "served_by": SERVED_BY_FAST}
            for key, res in fast_results.items()
        }
        for q in uncertain_questions:
            smart_answer = smart_results.get(q.key)
            if smart_answer and smart_answer.get("probabilities"):
                merged[q.key] = {
                    **smart_answer,
                    "escalated": True,
                    "served_by": SERVED_BY_SMART,
                }
            elif q.key in merged:
                logger.warning(
                    f"Smart engine did not answer escalated question '{q.key}'; "
                    f"returning the fast answer marked escalation_failed."
                )
                merged[q.key] = {**merged[q.key], "escalation_failed": True}

        self.current_model_id = "hybrid_routed"
        return self._require_answers(questions, merged)

    @staticmethod
    def _require_answers(
        questions: List[QuestionType], results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Raise when a question got no answer, so callers cannot mistake silence for a score of zero."""
        unanswered = [
            q.key for q in questions if not results.get(q.key, {}).get("probabilities")
        ]
        if unanswered:
            raise EngineUnavailableError(
                f"No engine answered: {', '.join(unanswered)}. Start the daemons with `make start`."
            )
        return results
