"""Build a labelled triage eval set from the Hugging Face email-classifier dataset.

    python -m tests.evals.hf_email_import [--per-class N] [--seed S] [--out PATH]

The source is `jason23322/high-accuracy-email-classifier`: synthetic bulk mail in six
categories. Every category is automated mail no person needs to answer, so each row's
expected routing is `archive_and_label` and `escalate_to_slack` is the failure to look for.
The set can measure false escalation; it cannot measure recall of real action items.

The dataset is gated (`huggingface-cli login`, with access granted on the dataset page) and
`datasets` is not a project dependency. Rows come from the test split, minus any text that
also appears in train. The output is written outside the repository by `write_private_dataset`.
"""

import argparse
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.evals.dataset import write_private_dataset

DATASET_ID = "jason23322/high-accuracy-email-classifier"
CATEGORIES = ("forum", "promotions", "social_media", "spam", "updates", "verify_code")
EXPECTED_ROUTING = "archive_and_label"
PLACEHOLDER_SENDER = "sender@dataset.example"
DEFAULT_PER_CLASS = 20
DEFAULT_OUT = Path.home() / ".jev" / "datasets" / "hf-email-classifier.jsonl"


def _category(row: Mapping[str, Any]) -> str:
    category = row["category"]
    if category not in CATEGORIES:
        raise ValueError(
            f"unknown category {category!r}; expected one of {', '.join(CATEGORIES)}"
        )
    return str(category)


def to_triage_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "subject": row["subject"],
        "sender": PLACEHOLDER_SENDER,
        "body": row["body"],
        "expected_routing": EXPECTED_ROUTING,
        "label": _category(row),
    }


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    per_class: int,
    seed: int,
    exclude_texts: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    """Take `per_class` rows from every category, skipping excluded and repeated texts."""
    seen = set(exclude_texts)
    pools: dict[str, list[Mapping[str, Any]]] = {
        category: [] for category in CATEGORIES
    }
    for row in rows:
        if row["text"] in seen:
            continue
        seen.add(row["text"])
        pools[_category(row)].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for category, pool in pools.items():
        if len(pool) < per_class:
            raise ValueError(
                f"category {category!r} has {len(pool)} usable rows, "
                f"fewer than --per-class {per_class}"
            )
        selected.extend(to_triage_row(row) for row in rng.sample(pool, per_class))
    return selected


def load_split(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "the `datasets` package is required: pip install datasets, then "
            "`huggingface-cli login` with access granted to " + DATASET_ID
        ) from error
    return list(load_dataset(DATASET_ID, split=split))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    train_texts = frozenset(row["text"] for row in load_split("train"))
    rows = select_rows(
        load_split("test"), args.per_class, args.seed, exclude_texts=train_texts
    )
    path = write_private_dataset(rows, args.out)
    print(f"{len(rows)} rows ({args.per_class} per category) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
