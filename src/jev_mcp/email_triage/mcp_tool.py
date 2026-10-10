import json
from pathlib import Path

def get_config_path() -> Path:
    config_dir = Path.home() / ".jev-mcp"
    config_dir.mkdir(exist_ok=True, parents=True)
    return config_dir / "triage_configs.json"

def configure_triage_labels(context_id: str, labels: list[str]) -> dict:
    """Configures dynamic labels for a specific mailbox or context ID."""
    config_path = get_config_path()
    
    if config_path.exists():
        data = json.loads(config_path.read_text())
    else:
        data = {}
        
    data[context_id] = labels
    config_path.write_text(json.dumps(data, indent=2))
    
    return {"status": "success", "message": f"Saved {len(labels)} labels for {context_id}"}

from jev_mcp.email_triage.core import build_triage_questions, decide_routing

def evaluate_email_with_mlx(subject: str, sender: str, body: str, labels: list[str]) -> dict:
    from jev_mcp.routing_provider import RoutingProvider
    from jev_mcp.provider import NoulQuestion, ChoiceQuestion
    
    questions = build_triage_questions(labels)
    eval_questions = []
    
    for q in questions:
        if q.type == "noul":
            eval_questions.append(NoulQuestion(key=q.question, prompt=q.instruction))
        elif q.type == "choice":
            eval_questions.append(ChoiceQuestion(key=q.question, prompt=q.instruction, options=q.choices))
            
    if len(body) > 2000:
        from jev_mcp.server import jev_compact_context
        import json
        
        try:
            goal = f"Identify if the email requires action, is important, or fits any of these categories: {', '.join(labels)}."
            compaction_res = jev_compact_context(
                state={"text": body},
                goal=goal,
                confidence_threshold=0.6
            )
            parsed_res = json.loads(compaction_res)
            if parsed_res.get("compressed_context"):
                body = parsed_res["compressed_context"]
        except Exception:
            pass

    state = {
        "email_subject": subject,
        "email_sender": sender,
        "email_body": body
    }
    
    provider = RoutingProvider()
    results = provider.evaluate_batch(state, eval_questions)
    
    requires_action_score = 0.0
    is_important_score = 0.0
    bucket = "Review"
    bucket_confidence = 0.0
    
    for key, res in results.items():
        probs = res.get("probabilities", {})
        if key == "requires_action":
            requires_action_score = probs.get("true", 0.0)
        elif key == "is_important":
            is_important_score = probs.get("true", 0.0)
        elif key == "bucket":
            if probs:
                best_choice = max(probs.items(), key=lambda x: x[1])
                bucket = best_choice[0]
                bucket_confidence = best_choice[1]
            
    decision = decide_routing(
        requires_action_score=requires_action_score,
        is_important_score=is_important_score,
        bucket=bucket,
        bucket_confidence=bucket_confidence
    )
    
    return {
        "decision": decision["decision"],
        "suggested_routing": decision["suggested_routing"],
        "requires_action_score": requires_action_score,
        "is_important_score": is_important_score,
        "bucket": bucket,
        "bucket_confidence": bucket_confidence
    }

def triage_email_content(context_id: str, subject: str, sender: str, body: str) -> str:
    """
    Evaluates an email using MLX and returns the recommended triage action.
    If 'status' == 'needs_config', you must ask the user what labels they want to track and call configure_triage_labels.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return json.dumps({
            "status": "needs_config", 
            "message": f"No labels configured for context '{context_id}'."
        })
        
    data = json.loads(config_path.read_text())
    if context_id not in data:
        return json.dumps({
            "status": "needs_config", 
            "message": f"No labels configured for context '{context_id}'."
        })
        
    labels = data[context_id]
    
    # Normally we call our local MLX evaluator here using `core.py` and `build_triage_questions`
    result = evaluate_email_with_mlx(subject, sender, body, labels)
    result["status"] = "success"
    
    return json.dumps(result)
