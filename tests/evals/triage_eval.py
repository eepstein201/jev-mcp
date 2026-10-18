"""Triage every email in a labelled dataset and report how often the routing matches.

    python -m tests.evals.triage_eval --dataset PATH [--json]
        [--binary-label-key KEY --positive VALUE]

Each dataset line is a JSON object with `subject`, `sender`, `body` and `expected_routing`; any
other key is ignored, except that `label` is carried into the `--json` rows. Both daemons must be
running. It measures; it does not gate.

With `--binary-label-key KEY --positive VALUE`, a row counts as a positive when `row[KEY] ==
VALUE`, and the report adds ranking metrics for `requires_action` (ROC-AUC with a bootstrap
interval, average precision, the best-F1 threshold) plus precision/recall/F1 of the routing
itself, where `escalate_to_slack` is the positive prediction.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jev_mcp.email_triage.core import CategoryConfig
from jev_mcp.email_triage.mcp_tool import evaluate_email_with_mlx
from tests.evals.metrics import (
    average_precision,
    best_f1,
    bootstrap_auc_ci,
    precision_recall_f1_from_predictions,
    roc_auc,
    threshold_sweep,
)
from tests.evals.rates import PassRate

REQUIRED_KEYS = ("subject", "sender", "body", "expected_routing")
ROUTINGS = ("escalate_to_slack", "hitl_slack_block", "archive_and_label")
SUBJECT_WIDTH = 50
POSITIVE_ROUTING = "escalate_to_slack"


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


def binary_labels(
    rows: Sequence[dict[str, Any]], key: str, positive: str
) -> list[bool]:
    """True where `row[key] == positive`; a row without the key is an error."""
    for number, row in enumerate(rows, start=1):
        if key not in row:
            raise ValueError(f"row {number} has no {key!r} key for --binary-label-key")
    return [row[key] == positive for row in rows]


def binary_summary(
    labels: Sequence[bool], scores: Sequence[float], routed: Sequence[bool]
) -> dict[str, Any]:
    """Ranking metrics for the scores, plus precision/recall/F1 of the routing decision."""
    sweep = threshold_sweep(labels, scores)
    interval = bootstrap_auc_ci(labels, scores)
    return {
        "n": len(labels),
        "positives": sum(labels),
        "auc": roc_auc(labels, scores),
        "auc_ci": list(interval) if interval else None,
        "average_precision": average_precision(labels, scores),
        "best_f1": best_f1(sweep),
        "sweep": sweep,
        "routing": precision_recall_f1_from_predictions(labels, routed),
    }


def format_binary(summary: dict[str, Any]) -> str:
    if summary["auc"] is None:
        auc_line = "auc  undefined (needs both classes)"
    else:
        low, high = summary["auc_ci"]
        auc_line = f"auc  {summary['auc']:.3f}  95% CI [{low:.3f}, {high:.3f}]"
    ap = summary["average_precision"]
    best = summary["best_f1"]
    routing = summary["routing"]
    return "\n".join(
        [
            f"binary  n={summary['n']}  positives={summary['positives']}",
            auc_line,
            "average_precision  " + ("undefined" if ap is None else f"{ap:.3f}"),
            f"best_f1  threshold={best['threshold']:.2f}  "
            f"precision={best['precision']:.3f}  recall={best['recall']:.3f}  "
            f"f1={best['f1']:.3f}",
            f"routing  precision={routing['precision']:.3f}  "
            f"recall={routing['recall']:.3f}  f1={routing['f1']:.3f}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--json", action="store_true", help="print one JSON object with every row"
    )
    parser.add_argument(
        "--binary-label-key", help="dataset key holding the binary label (e.g. label)"
    )
    parser.add_argument(
        "--positive", help="value of that key that counts as the positive class"
    )
    args = parser.parse_args(argv)
    if (args.binary_label_key is None) != (args.positive is None):
        parser.error("--binary-label-key and --positive must be given together")

    labels = CategoryConfig().categories
    rows = load_dataset(args.dataset)
    scored = [score_row(row, labels) for row in rows]
    binary = None
    if args.binary_label_key is not None:
        binary = binary_summary(
            binary_labels(rows, args.binary_label_key, args.positive),
            [row["requires_action"] for row in scored],
            [row["got"] == POSITIVE_ROUTING for row in scored],
        )
    if args.json:
        rate = agreement(scored)
        summary = {"runs": rate.runs, "passes": rate.passes, "rate": rate.rate}
        payload = {"agreement": summary, "rows": scored}
        print(json.dumps({**payload, "binary": binary} if binary else payload))
    else:
        print(format_report(scored))
        if binary:
            print(format_binary(binary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
