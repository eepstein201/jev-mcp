import re
from typing import Dict, Any, List
from jev_mcp.routing_provider import RoutingProvider
from jev_mcp.provider import ChoiceQuestion, QuestionType

# "The page shows MISSION_COMPLETE only when…" → MISSION_COMPLETE. A single
# all-caps token only, so ordinary prose after "shows" never matches.
_MARKER_RE = re.compile(r"\bshows?\s+[\"'“']?([A-Z][A-Z0-9_\-]{3,40})")

# Target groups larger than this get a second, focused pass over the top-k
# candidates: one-word discrimination works over five options, not fifty.
NARROW_MIN_CANDIDATES = 12
SHORTLIST_K = 5


def success_marker(goal: str) -> str | None:
    """The success token the goal says the page will show, or None if none is stated."""
    match = _MARKER_RE.search(goal or "")
    return match.group(1) if match else None


def _tried_labels(history: List[Any], lookback: int = 5) -> Dict[str, int]:
    """Element label → how many steps ago it was last acted on."""
    notes: Dict[str, int] = {}
    for age, entry in enumerate(reversed(history[-lookback:]), start=1):
        label = (entry.get("action") or "").strip() if isinstance(entry, dict) else ""
        if label and label not in notes:
            notes[label] = age
    return notes


def _is_alternating(history: List[Any]) -> bool:
    labels = [h.get("action") for h in history[-4:] if isinstance(h, dict)]
    return (
        len(history) >= 4
        and labels[0] is not None
        and labels[0] == labels[2]
        and labels[1] == labels[3]
        and labels[0] != labels[1]
    )

def local_choose(state: Dict[str, Any], goal: str, history: List[Any], operations: Dict[str, str], targets: Dict[str, Any], controls: Dict[str, Any]) -> Dict[str, Any]:
    """
    Acts as a drop-in local replacement for the TypeSafe API call made by Jev-Ultrafast.
    Maps operations/targets to ChoiceQuestions, evaluates them via Jev-MCP's local engines,
    and returns a TypeSafe-compatible response payload.
    """
    provider = RoutingProvider()
    
    questions: List[QuestionType] = []

    # Steering notes keep the decision layer from looping: the goal's stated
    # completion criterion, a no-progress flag, and an alternating-pick warning.
    notes: List[str] = []
    marker = success_marker(goal)
    if marker:
        notes.append(f"The goal is complete only when the page shows {marker}.")
    if history and isinstance(history[-1], dict) and history[-1].get("page_changed") is False:
        notes.append("Your last action did not change the page.")
    if _is_alternating(history):
        notes.append(
            "You are alternating between the same two actions without completing the goal; "
            "choose a different target, or DONE if the goal is visibly satisfied.")
    if history and isinstance(history[-1], dict) and history[-1].get("kind") == "fill":
        notes.append(
            "A field was just filled; waiting rarely completes a form — click the goal's "
            "next element (e.g. Submit) if the goal names one.")
    recent_labels = [h.get("action") for h in history[-5:] if isinstance(h, dict)]
    if recent_labels:
        common = max(set(recent_labels), key=recent_labels.count)
        if recent_labels.count(common) >= 4:
            notes.append(
                f"You have chosen '{common}' {recent_labels.count(common)} of the last "
                f"{len(recent_labels)} times in a row without completing the goal; choose a different operation.")
    if any(len(candidates) > NARROW_MIN_CANDIDATES for candidates in targets.values()):
        notes.append(
            "If no listed element clearly matches the goal, scroll to reveal more elements "
            "instead of clicking a poor match.")
    note_suffix = (" " + " ".join(notes)) if notes else ""

    # 1. Add the operation choice question
    # operations maps keys like "CLICK" to descriptions
    q_operation = ChoiceQuestion(
        key="operation",
        prompt=f"Goal: {goal}\nChoose the best operation.{note_suffix}",
        options=list(operations.keys())
    )
    questions.append(q_operation)
    
    # 2. Add target choice questions for each operation
    tried = _tried_labels(history)
    for op, candidates in targets.items():
        # The options are bare indices, so the prompt has to say which element each one is.
        # Recently tried elements are flagged so the engine does not pick them again.
        lines = []
        for index, candidate in candidates.items():
            label_text = candidate.get('label', candidate.get('element', ''))
            line = f"[{index}] {label_text}"
            if label_text in tried:
                line += f" (tried {tried[label_text]} step(s) ago — it did not complete the goal)"
            lines.append(line)
        listing = "\n".join(lines)
        q_target = ChoiceQuestion(
            key=f"{op.lower()}_target",
            prompt=f"Goal: {goal}\nOperation: {op}\nChoose the target index.{note_suffix}\nElements:\n{listing}",
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

    # Dense listings get a second, focused pass: re-ask over only the top-k
    # candidates from the first pass, where one-word discrimination is easy.
    for op, candidates in targets.items():
        if op not in ("CLICK", "SELECT") or len(candidates) <= NARROW_MIN_CANDIDATES:
            continue
        answer = results.get(f"{op.lower()}_target")
        if not answer or len(answer.get("probabilities", {})) < 2:
            continue
        top = sorted(candidates, key=lambda i: answer["probabilities"].get(i, 0.0), reverse=True)[:SHORTLIST_K]
        if len(top) < 2:
            continue
        short = {i: candidates[i] for i in top}
        listing = "\n".join(f"[{i}] {short[i].get('label', short[i].get('element', ''))}" for i in top)
        q_shortlist = ChoiceQuestion(
            key=f"{op.lower()}_target",
            prompt=(f"Goal: {goal}\nOperation: {op}\nChoose the target index.{note_suffix}\n"
                    f"Shortlist — the most likely candidates:\n{listing}"),
            options=top,
        )
        refined = provider.evaluate_batch(slim_state, [q_shortlist]).get(f"{op.lower()}_target")
        if refined:
            # validate_choice requires probabilities over exactly the offered ids,
            # so expand the shortlist answer back over every candidate (zeros elsewhere).
            expanded = {index: 0.0 for index in candidates}
            for index, probability in refined.get("probabilities", {}).items():
                expanded[index] = probability
            results[f"{op.lower()}_target"] = {**refined, "probabilities": expanded}
    
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
