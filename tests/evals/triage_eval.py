"""Triage every email in a labelled dataset and report how often the routing matches.

    python -m tests.evals.triage_eval --dataset PATH [--json]

Each dataset line is a JSON object with `subject`, `sender`, `body` and `expected_routing`; any
other key is ignored, except that `label` is carried into the `--json` rows. Both daemons must be
running. It measures; it does not gate.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jev_mcp.email_triage.core import CategoryConfig
from jev_mcp.email_triage.mcp_tool import evaluate_email_with_mlx
from tests.evals.rates import PassRate

REQUIRED_KEYS = ("subject", "sender", "body", "expected_routing")
ROUTINGS = ("escalate_to_slack", "hitl_slack_block", "archive_and_label")
SUBJECT_WIDTH = 50


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for number, row in enumerate(rows, start=1):
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"{path} row {number} is missing {', '.join(missing)}")
        if row["expected_routing"] not in ROUTINGS:
            raise ValueError(
                f"{path} row {number} has expected_routing {row['expected_routing']!r}; "
                f"expected one of {', '.join(ROUTINGS)}"
            )
    return rows


def score_row(row: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    result = evaluate_email_with_mlx(row["subject"], row["sender"], row["body"], labels)
    label = {"label": row["label"]} if "label" in row else {}
    return {
        "subject": row["subject"],
        **label,
        "expected": row["expected_routing"],
        "got": result["suggested_routing"],
        "requires_action": result["requires_action_score"],
        "is_important": result["is_important_score"],
    }


def agreement(scored: Sequence[dict[str, Any]]) -> PassRate:
    return PassRate(
        runs=len(scored),
        passes=sum(row["expected"] == row["got"] for row in scored),
    )


def routing_by_label(scored: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count the routing each label received; rows without a label are left out."""
    counts: dict[str, Counter[str]] = {}
    for row in scored:
        if "label" in row:
            counts.setdefault(row["label"], Counter())[row["got"]] += 1
    return {label: dict(routings) for label, routings in counts.items()}


def format_report(scored: Sequence[dict[str, Any]]) -> str:
    rate = agreement(scored)
    lines = [f"agreement  {rate.passes}/{rate.runs}  {rate.rate:.0%}"]
    breakdown = routing_by_label(scored)
    if breakdown:
        lines.append("by label")
        lines.extend(
            f"  {label}: "
            + " ".join(
                f"{routing}={count}" for routing, count in sorted(routings.items())
            )
            for label, routings in sorted(breakdown.items())
        )
    disagreements = [row for row in scored if row["expected"] != row["got"]]
    if disagreements:
        lines.append("subject | expected | got | requires_action | is_important")
    lines.extend(
        f"{row['subject'][:SUBJECT_WIDTH]} | {row['expected']} | {row['got']} | "
        f"{row['requires_action']:.3f} | {row['is_important']:.3f}"
        for row in disagreements
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--json", action="store_true", help="print one JSON object with every row"
    )
    args = parser.parse_args(argv)

    labels = CategoryConfig().categories
    scored = [score_row(row, labels) for row in load_dataset(args.dataset)]
    if args.json:
        rate = agreement(scored)
        summary = {"runs": rate.runs, "passes": rate.passes, "rate": rate.rate}
        print(json.dumps({"agreement": summary, "rows": scored}))
    else:
        print(format_report(scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
