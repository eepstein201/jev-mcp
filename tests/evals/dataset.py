"""Write labelled triage email datasets to a private path outside the repository.

The rows are real mail, so a dataset file must never live inside the repo where
it could be committed. `write_private_dataset` refuses repository paths and
writes the JSONL readable by its owner only.
"""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_private_dataset(rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write `rows` as one JSON object per line at `path` with mode 0o600.

    Returns the resolved path. Raises ValueError when `path` resolves inside
    the repository, so real email content cannot be committed by accident.
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"dataset path is inside the repository: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row)) + "\n" for row in rows)
    # Create the file owner-only from the first moment it exists.
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    # os.open's mode applies at creation only; tighten any pre-existing file.
    resolved.chmod(0o600)
    return resolved
