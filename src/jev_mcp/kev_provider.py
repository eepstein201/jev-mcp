import json
import urllib.request
import logging
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
        # Rough estimation
        return len(state_str) // 4
        
    def evaluate_batch(self, state: Any, questions: List[QuestionType]) -> Dict[str, Dict[str, Any]]:
        state_str = json.dumps(state) if not isinstance(state, str) else state
        
        q_payload = {}
        for q in questions:
            if isinstance(q, NoulQuestion):
                q_payload[q.key] = {
                    "type": "noul",
                    "instructions": q.prompt
                }
            elif isinstance(q, ChoiceQuestion):
                q_payload[q.key] = {
                    "type": "choice",
                    "instructions": q.prompt,
                    "criteria": {opt: "" for opt in q.options}
                }
            elif isinstance(q, ScoreQuestion):
                q_payload[q.key] = {
                    "type": "score",
                    "instructions": q.prompt,
                    "criteria": q.labels
                }
                
        payload = {
            "state": state_str,
            "questions": q_payload
        }
        
        try:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode())
                
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
                    "probabilities": {"1": ans.get("noul", 0.0), "0": 1.0 - ans.get("noul", 0.0)}
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
