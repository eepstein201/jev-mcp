import json
from jev_mcp.provider import post_json
import logging
from jev_mcp.security import sanitize_payload
from typing import Any, List, Dict

from jev_mcp.provider import JevProvider, QuestionType, NoulQuestion, ScoreQuestion, ChoiceQuestion

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
        
    def evaluate_batch(self, state: Any, questions: List[QuestionType]) -> Dict[str, Dict[str, Any]]:
        state_str = json.dumps(state) if not isinstance(state, str) else state
        state_str = sanitize_payload(state_str)
        
        q_payload = {}
        for q in questions:
            if isinstance(q, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
                payload_q = q.to_dict()
                payload_q["instructions"] = sanitize_payload(payload_q["instructions"])
                q_payload[q.key] = payload_q
                
        payload = {
            "state": state_str,
            "questions": q_payload
        }
        
        try:
            res = post_json(self.base_url, payload, timeout=30)
                
        except Exception as e:
            logger.error(f"Kev API call failed: {e}\nPLEASE ENSURE THE KEV DAEMON IS RUNNING! Run: ./jev_mac_manager.sh hybrid")
            return {}
            
        final_results = {}
        
        for q in questions:
            if q.key not in res.get("answers", {}):
                continue
                
            ans = res["answers"][q.key]
            
            if isinstance(q, NoulQuestion):
                final_results[q.key] = {
                    "noul": ans.get("noul", 0.0),
                    "probabilities": {"true": ans.get("noul", 0.0), "false": 1.0 - ans.get("noul", 0.0)}
                }
            elif isinstance(q, ChoiceQuestion):
                final_results[q.key] = {
                    "choice": ans.get("choice", ""),
                    "probabilities": ans.get("probabilities", {})
                }
            elif isinstance(q, ScoreQuestion):
                raw_probs = ans.get("probabilities", {})
                mapped_probs = {}
                best_label = q.labels[0]
                best_prob = -1
                for idx_str, p in raw_probs.items():
                    idx = int(idx_str)
                    label = q.labels[idx] if idx < len(q.labels) else str(idx)
                    mapped_probs[label] = p
                    if p > best_prob:
                        best_prob = p
                        best_label = label
                        
                final_results[q.key] = {
                    "score": best_label,
                    "probabilities": mapped_probs
                }
                
        return final_results
