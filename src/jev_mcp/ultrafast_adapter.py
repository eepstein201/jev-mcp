from typing import Dict, Any, List
from jev_mcp.routing_provider import RoutingProvider
from jev_mcp.provider import ChoiceQuestion, QuestionType

def local_choose(state: Dict[str, Any], goal: str, history: List[Any], operations: Dict[str, str], targets: Dict[str, Any], controls: Dict[str, Any]) -> Dict[str, Any]:
    """
    Acts as a drop-in local replacement for the TypeSafe API call made by Jev-Ultrafast.
    Maps operations/targets to ChoiceQuestions, evaluates them via Jev-MCP's local engines,
    and returns a TypeSafe-compatible response payload.
    """
    provider = RoutingProvider()
    
    questions: List[QuestionType] = []

    # 1. Add the operation choice question
    # operations maps keys like "CLICK" to descriptions
    q_operation = ChoiceQuestion(
        key="operation",
        prompt=f"Goal: {goal}\nChoose the best operation.",
        options=list(operations.keys())
    )
    questions.append(q_operation)
    
    # 2. Add target choice questions for each operation
    for op, candidates in targets.items():
        q_target = ChoiceQuestion(
            key=f"{op.lower()}_target",
            prompt=f"Goal: {goal}\nOperation: {op}\nChoose the target index.",
            options=list(candidates.keys())
        )
        questions.append(q_target)
        
    # Simplify state for context limits
    slim_state = {
        "page": {k: state["page"][k] for k in ("url", "title", "text") if k in state.get("page", {})},
        "recent_actions": [
            {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
        ],
    }
    
    # Run evaluation
    results = provider.evaluate_batch(slim_state, questions)
    
    # Map back to the TypeSafe "answers" dict format
    answers = {}
    for key, ans in results.items():
        answers[key] = {
            "choice": ans.get("choice", ""),
            "confidence": ans.get("confidence", 1.0),
            "probabilities": ans.get("probabilities", {})
        }
        
    return {
        "model": "jev-mcp-local",
        "usage": {"total_tokens": 0},
        "answers": answers
    }
