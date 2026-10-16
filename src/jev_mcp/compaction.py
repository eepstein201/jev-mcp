"""Chunking and per-chunk scoring for jev_compact_context.

Kept apart from the tool so the question wording and threshold can vary and so the
evaluation harness can reuse the exact chunking the tool applies.
"""

import json
from typing import Any, List

from jev_mcp.provider import EngineUnavailableError, NoulQuestion

CHUNK_MAX_CHARS = 2000

GENERIC_QUESTION = "Is the information in this text snippet directly relevant to achieving the goal: '{goal}'?"


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
