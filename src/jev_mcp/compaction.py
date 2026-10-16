"""Chunking and per-chunk scoring for jev_compact_context.

Kept apart from the tool so the question wording and threshold can vary and so the
evaluation harness can reuse the exact chunking the tool applies.
"""

import json
from typing import Any, List

from jev_mcp.provider import ChoiceQuestion, EngineUnavailableError, NoulQuestion

CHUNK_MAX_CHARS = 2000

GENERIC_QUESTION = "Is the information in this text snippet directly relevant to achieving the goal: '{goal}'?"

DEFAULT_CONFIDENCE_THRESHOLD = 0.3

GOAL_TYPES = ("open_items", "find_fact", "continue_work")
OTHER_GOAL = "other"

# Wordings are chosen on the tune split of tests/evals/compaction_goals.jsonl. They describe the
# content a goal needs, never how a state key is named.
TYPED_QUESTIONS = {
    "open_items": "Does this text describe something that is still unresolved, pending, blocked or not yet done (an open question, a remaining task or an unfinished item) relevant to: '{goal}'?",
    "find_fact": "Could the answer to '{goal}' be found in this snippet?",
    "continue_work": "Would someone resuming the task '{goal}' need this text: its current state, decisions made, constraints or next steps?",
}

# Per-type keep thresholds, chosen on the tune split as the highest compression at recall >= 0.95.
TYPED_THRESHOLDS = {
    "open_items": 0.35,
    "find_fact": 0.15,
    "continue_work": 0.4,
}

GOAL_TYPE_QUESTION = (
    "What kind of request is this goal? open_items: it asks what is still open, unresolved, "
    "pending or remaining. find_fact: it asks for one specific fact, value or decision. "
    "continue_work: it asks to resume, continue or pick up a task. other: none of these. "
    "Goal: '{goal}'"
)


def chunk_state(state: Any) -> List[str]:
    """Split `state` into verbatim chunks; blank state gives no chunks."""
    # Only a string under "text" is treated as prose; anything else is scored as JSON, key by key.
    has_text = isinstance(state, dict) and isinstance(state.get("text"), str)
    state_str = state["text"] if has_text else json.dumps(state)
    if not state_str.strip():
        return []

    if isinstance(state, dict) and state and not has_text:
        # One chunk per top-level key, so unrelated fields are scored separately.
        raw_chunks = [
            f"{k}: {v if isinstance(v, str) else json.dumps(v)}"
            for k, v in state.items()
        ]
    else:
        raw_chunks = [c.strip() for c in state_str.split("\n\n") if c.strip()]

    chunks: List[str] = []
    for c in raw_chunks:
        if len(c) > CHUNK_MAX_CHARS:
            # Fixed-width slices keep the text verbatim; re-wrapping would collapse its whitespace.
            chunks.extend(
                c[i : i + CHUNK_MAX_CHARS] for i in range(0, len(c), CHUNK_MAX_CHARS)
            )
        else:
            chunks.append(c)
    return chunks


def score_chunk(provider: Any, index: int, chunk: str, prompt: str) -> float:
    """The fast engine's 'noul' probability for `prompt` about `chunk`.

    Raises EngineUnavailableError when the engine gives no answer.
    """
    q = NoulQuestion(key=f"chunk_{index}", prompt=prompt)
    # We force the fast provider directly to save time and prevent escalation.
    target = getattr(provider, "fast_provider", provider)
    res = target.evaluate_batch({"text": chunk}, [q])
    if q.key not in res:
        # The fast engine reports an outage as an empty result, not an exception.
        raise EngineUnavailableError(f"No engine answered {q.key}")
    return res[q.key].get("noul", 0.0)


def classify_goal(provider: Any, goal: str) -> str:
    """One of GOAL_TYPES, or OTHER_GOAL when the engine fails or is unsure."""
    q = ChoiceQuestion(
        key="goal_type",
        prompt=GOAL_TYPE_QUESTION.format(goal=goal),
        options=[*GOAL_TYPES, OTHER_GOAL],
    )
    try:
        target = getattr(provider, "fast_provider", provider)
        choice = target.evaluate_batch({"text": goal}, [q]).get(q.key, {}).get("choice")
    except Exception:
        return OTHER_GOAL
    return choice if choice in GOAL_TYPES else OTHER_GOAL


def plan_scoring(
    provider: Any, goal: str, threshold: float, goal_aware: bool
) -> tuple[str, float, str | None]:
    """The question, keep threshold and goal type (None when goal-blind) to score with.

    Goal-aware scoring swaps in the type's question and default threshold; a caller threshold
    other than the default always wins, and an unclassified goal keeps the generic question.
    """
    if not goal_aware:
        return GENERIC_QUESTION.format(goal=goal), threshold, None
    goal_type = classify_goal(provider, goal)
    if goal_type == OTHER_GOAL:
        return GENERIC_QUESTION.format(goal=goal), threshold, goal_type
    if threshold == DEFAULT_CONFIDENCE_THRESHOLD:
        threshold = TYPED_THRESHOLDS[goal_type]
    return TYPED_QUESTIONS[goal_type].format(goal=goal), threshold, goal_type
