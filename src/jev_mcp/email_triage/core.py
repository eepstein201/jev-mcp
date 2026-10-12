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
            prompt="Does this email require an action from me? Examples: a question I must answer, a meeting I must schedule, a task I must complete.",
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
        return {"decision": "Action", "suggested_routing": "escalate_to_slack"}

    # Ambiguous action scores (between 0.3 and the threshold) should be reviewed manually,
    # as should a low bucket confidence, and an important email that needs no action: filing
    # it away silently is the costly mistake, an extra review card is not.
    if (
        (
            requires_action_score > 0.3
            and requires_action_score <= config.action_threshold
        )
        or bucket_confidence < 0.6
        or is_important_score > config.important_threshold
    ):
        return {"decision": "Review", "suggested_routing": "hitl_slack_block"}

    # Otherwise, confidently route to the assigned bucket
    return {"decision": bucket, "suggested_routing": "archive_and_label"}
