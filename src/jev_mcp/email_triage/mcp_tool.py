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

def evaluate_email_with_mlx(subject: str, sender: str, body: str, labels: list[str]) -> dict:
    # Stub for the actual MLX evaluation call that will be implemented later
    pass

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
