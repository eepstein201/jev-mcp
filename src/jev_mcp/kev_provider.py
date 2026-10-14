from functools import singledispatchmethod
import json
from jev_mcp.provider import post_json
import logging
from jev_mcp.security import sanitize_payload
from typing import Any, List, Dict

from jev_mcp.provider import (
    JevProvider,
    QuestionType,
    NoulQuestion,
    ScoreQuestion,
    ChoiceQuestion,
)

logger = logging.getLogger(__name__)


class KevProvider(JevProvider):
    """
    Evaluates requests natively against a local Kev daemon (TypeSafe SystemOne API) running on macOS.
    Bypasses text generation entirely in favor of a specialized pointer head.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1/systemone") -> None:
        self.base_url = base_url
        self.max_tokens = 65536
        self.current_model_id = "kev-0.8b"

    def check_token_limit(self, state: Any) -> int:
        state_str = json.dumps(state) if not isinstance(state, str) else state
        state_str = sanitize_payload(state_str)
        # Rough estimation
        return len(state_str) // 4

    @singledispatchmethod
    def _parse_answer(self, q: Any, ans: dict) -> dict:
        logger.warning(f"Unsupported question type in KevProvider: {type(q)}")
        return {}

    @_parse_answer.register
    def _(self, q: NoulQuestion, ans: dict) -> dict:
        # A missing or non-numeric score is no answer: a default of 0.0 would read as a
        # confident "false".
        noul = ans.get("noul")
        if isinstance(noul, bool) or not isinstance(noul, (int, float)):
            logger.warning(f"Kev answer for {q.key!r} has no numeric 'noul'; ignoring.")
            return {}
        return {
            "noul": noul,
            "probabilities": {"true": noul, "false": 1.0 - noul},
        }

    @_parse_answer.register
    def _(self, q: ChoiceQuestion, ans: dict) -> dict:
        choice = ans.get("choice")
        if not isinstance(choice, str) or not choice:
            logger.warning(f"Kev answer for {q.key!r} has no 'choice'; ignoring.")
            return {}
        return {
            "choice": choice,
            "confidence": ans.get("confidence", 1.0),
            "probabilities": ans.get("probabilities", {}),
        }

    @_parse_answer.register
    def _(self, q: ScoreQuestion, ans: dict) -> dict:
        raw_probs = ans.get("probabilities")
        if not isinstance(raw_probs, dict) or not raw_probs:
            logger.warning(
                f"Kev answer for {q.key!r} has no 'probabilities'; ignoring."
            )
            return {}
        mapped_probs = {}
        best_label = q.labels[0] if q.labels else "0"
        best_prob = -1
        for idx_str, p in raw_probs.items():
            # Only a plain index into the labels is translated; any other key is kept as sent.
            key = str(idx_str)
            label = (
                q.labels[int(key)]
                if key.isdigit() and int(key) < len(q.labels)
                else key
            )
            mapped_probs[label] = p
            if p > best_prob:
                best_prob = p
                best_label = label

        return {"score": best_label, "probabilities": mapped_probs}

    def evaluate_batch(
        self, state: Any, questions: List[QuestionType]
    ) -> Dict[str, Dict[str, Any]]:
        state_str = json.dumps(state) if not isinstance(state, str) else state
        state_str = sanitize_payload(state_str)

        q_payload = {}
        for q in questions:
            if not hasattr(q, "to_dict"):
                continue
            payload_q = q.to_dict()
            payload_q["instructions"] = sanitize_payload(payload_q["instructions"])
            q_payload[q.key] = payload_q

        payload = {"state": state_str, "questions": q_payload}

        try:
            res = post_json(self.base_url, payload, timeout=30)

        except Exception as e:
            logger.error(
                f"Kev API call failed: {e}\nPLEASE ENSURE THE KEV DAEMON IS RUNNING! Run: ./jev_mac_manager.sh hybrid"
            )
            return {}

        final_results = {}

        for q in questions:
            if q.key not in res.get("answers", {}):
                continue

            ans = res["answers"][q.key]
            if not isinstance(ans, dict):
                logger.warning(f"Kev answer for {q.key!r} is not an object; ignoring.")
                continue

            parsed = self._parse_answer(q, ans)
            if parsed:
                final_results[q.key] = parsed

        return final_results
