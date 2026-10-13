import logging
import math
import os

from pydantic import BaseModel, Field

from jev_mcp.provider import ChoiceQuestion, NoulQuestion, QuestionType

logger = logging.getLogger(__name__)


class JevMailConfig(BaseModel):
    action_threshold: float = Field(default=0.8)
    important_threshold: float = Field(default=0.8)


ACTION_THRESHOLD_ENV = "JEV_TRIAGE_ACTION_THRESHOLD"
IMPORTANT_THRESHOLD_ENV = "JEV_TRIAGE_IMPORTANT_THRESHOLD"


def _threshold_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    # NaN fails both comparisons, so it is rejected here too.
    if not 0.0 <= value <= 1.0:
        logger.warning(
            "%s=%r is not a number between 0 and 1; using the default %s",
            name,
            raw,
            default,
        )
        return default
    return value


def load_mail_config() -> JevMailConfig:
    """The routing thresholds, each overridable with an environment variable."""
    defaults = JevMailConfig()
    return JevMailConfig(
        action_threshold=_threshold_from_env(
            ACTION_THRESHOLD_ENV, defaults.action_threshold
        ),
        important_threshold=_threshold_from_env(
            IMPORTANT_THRESHOLD_ENV, defaults.important_threshold
        ),
    )


class CategoryConfig(BaseModel):
    categories: list[str] = Field(
        default=[
            "Follow Up",
            "Pending",
            "Receipts",
            "Newsletter",
            "Notifications",
            "Review",
        ]
    )


def build_triage_questions(categories: list[str] | None = None) -> list[QuestionType]:
    if categories is None:
        categories = CategoryConfig().categories

    return [
        NoulQuestion(
            key="requires_action",
            prompt="Does this email require an action from me? Examples: a question I must answer, a meeting I must schedule, a task I must complete. Not an action: a one-time code or sign-in approval, an automated notification (forum, social media, delivery or account status), a promotion, or an unsolicited message pressuring me to click a link.",
        ),
        NoulQuestion(
            key="is_important",
            prompt="Is this email important? Examples: messages from my boss, critical alerts, high-priority client emails.",
        ),
        ChoiceQuestion(
            key="bucket",
            prompt="Which category best fits this email?",
            options=categories,
        ),
    ]


def _decision(decision: str, suggested_routing: str, reason: str) -> dict:
    return {
        "decision": decision,
        "suggested_routing": suggested_routing,
        "reason": reason,
    }


def decide_routing(
    requires_action_score: float,
    is_important_score: float,
    bucket: str,
    bucket_confidence: float,
    config: JevMailConfig | None = None,
) -> dict:
    if config is None:
        config = JevMailConfig()

    # High action probability means it's an Action item
    if requires_action_score > config.action_threshold:
        return _decision("Action", "escalate_to_slack", "requires_action")

    # Anything below needs a person: an ambiguous action score (between 0.3 and the threshold),
    # an important email that needs no action (filing it away silently is the costly mistake,
    # an extra review card is not), or a low bucket confidence. The first that holds is the reason.
    if 0.3 < requires_action_score <= config.action_threshold:
        return _decision("Review", "hitl_slack_block", "ambiguous_action")
    if is_important_score > config.important_threshold:
        return _decision("Review", "hitl_slack_block", "important_no_action")
    if bucket_confidence < 0.6:
        return _decision("Review", "hitl_slack_block", "low_bucket_confidence")

    # Otherwise, confidently route to the assigned bucket
    return _decision(bucket, "archive_and_label", "confident_bucket")
