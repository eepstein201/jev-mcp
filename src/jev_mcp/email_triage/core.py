from pydantic import BaseModel, Field

class NoulQuestion(BaseModel):
    question: str
    instruction: str
    type: str = "noul"

class ChoiceQuestion(BaseModel):
    question: str
    instruction: str
    choices: list[str]
    type: str = "choice"

class JevMailConfig(BaseModel):
    action_threshold: float = Field(default=0.8)
    important_threshold: float = Field(default=0.8)

class CategoryConfig(BaseModel):
    categories: list[str] = Field(
        default=["Follow Up", "Pending", "Receipts", "Newsletter", "Notifications", "Review"]
    )

def build_triage_questions(categories: list[str] = None):
    if categories is None:
        categories = CategoryConfig().categories
        
    return [
        NoulQuestion(
            question="requires_action",
            instruction="Does this email require an action from me? Examples: a question I must answer, a meeting I must schedule, a task I must complete."
        ),
        NoulQuestion(
            question="is_important",
            instruction="Is this email important? Examples: messages from my boss, critical alerts, high-priority client emails."
        ),
        ChoiceQuestion(
            question="bucket",
            instruction="Which category best fits this email?",
            choices=categories
        )
    ]

def decide_routing(
    requires_action_score: float, 
    is_important_score: float, 
    bucket: str, 
    bucket_confidence: float, 
    config: JevMailConfig = None
) -> dict:
    if config is None:
        config = JevMailConfig()
        
    # High action probability means it's an Action item
    if requires_action_score > config.action_threshold:
        return {
            "decision": "Action",
            "suggested_routing": "escalate_to_slack"
        }
    
    # Ambiguous action scores (between 0.3 and the threshold) should be reviewed manually
    # or if the bucket confidence is low
    if (requires_action_score > 0.3 and requires_action_score <= config.action_threshold) or bucket_confidence < 0.6:
        return {
            "decision": "Review",
            "suggested_routing": "hitl_slack_block"
        }
        
    # Otherwise, confidently route to the assigned bucket
    return {
        "decision": bucket,
        "suggested_routing": "archive_and_label"
    }
