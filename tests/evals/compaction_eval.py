"""Measure recall / precision / compression of jev_compact_context across confidence thresholds.

    python -m tests.evals.compaction_eval [--dataset PATH] [--json]

Each dataset line has `id`, `goal_type`, `split` (tune | holdout), `goal`, `state` ({key: text}),
`relevant` (keys) and `note`. The fast engine scores every chunk once with the tool's generic
question; thresholds are then swept offline. The recommended threshold is chosen on the tune split
only, all goal types pooled: the highest compression with recall >= 0.95 (ties go to the higher
threshold), else the highest recall. Holdout is reported at that frozen value and never used to
choose it. It measures; it does not gate. The fast daemon must be running.
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_mcp.compaction import GENERIC_QUESTION, chunk_state

DEFAULT_DATASET = Path(__file__).with_name("compaction_goals.jsonl")
SPLITS = ("tune", "holdout")
GOAL_TYPES = ("open_items", "find_fact", "continue_work")
REQUIRED_KEYS = ("id", "goal_type", "split", "goal", "state", "relevant")
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 15))
MIN_RECALL = 0.95
OLD_DEFAULT = 0.3

Score = float
ScoreFn = Callable[[str, str], Score]


@dataclass(frozen=True)
class Row:
    id: str
    goal_type: str
    split: str
    goal: str
    keys: tuple[str, ...]
    chunks: tuple[str, ...]
    relevant: frozenset[str]


@dataclass(frozen=True)
class Counts:
    relevant: int = 0
    relevant_kept: int = 0
    kept: int = 0
    total: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.relevant + other.relevant,
            self.relevant_kept + other.relevant_kept,
            self.kept + other.kept,
            self.total + other.total,
        )

    @property
    def recall(self) -> float:
        return self.relevant_kept / self.relevant if self.relevant else 1.0

    @property
    def precision(self) -> float:
        return self.relevant_kept / self.kept if self.kept else 1.0

    @property
    def compression(self) -> float:
        return 1.0 - self.kept / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "recall": round(self.recall, 3),
            "precision": round(self.precision, 3),
            "compression": round(self.compression, 3),
            "chunks": self.total,
        }


def load_dataset(path: Path) -> list[Row]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [key for key in REQUIRED_KEYS if key not in raw]
        if missing:
            raise ValueError(f"{path} row {number} is missing {', '.join(missing)}")
        if raw["goal_type"] not in GOAL_TYPES:
            raise ValueError(f"{path} row {number} has goal_type {raw['goal_type']!r}")
        if raw["split"] not in SPLITS:
            raise ValueError(f"{path} row {number} has split {raw['split']!r}")
        unknown = set(raw["relevant"]) - set(raw["state"])
        if unknown:
            raise ValueError(
                f"{path} row {number} marks unknown keys {sorted(unknown)}"
            )
        keys = tuple(raw["state"])
        chunks = tuple(chunk_state(raw["state"]))
        if len(chunks) != len(keys):
            raise ValueError(f"{path} row {number} has a value longer than one chunk")
        rows.append(
            Row(
                raw["id"],
                raw["goal_type"],
                raw["split"],
                raw["goal"],
                keys,
                chunks,
                frozenset(raw["relevant"]),
            )
        )
    return rows


def counts_for(row: Row, scores: Sequence[Score], threshold: float) -> Counts:
    kept = [score >= threshold for score in scores]
    relevant = [key in row.relevant for key in row.keys]
    return Counts(
        relevant=sum(relevant),
        relevant_kept=sum(k and r for k, r in zip(kept, relevant)),
        kept=sum(kept),
        total=len(kept),
    )


def pooled(
    rows: Sequence[Row], scores: Mapping[str, Sequence[Score]], threshold: float
) -> Counts:
    total = Counts()
    for row in rows:
        total += counts_for(row, scores[row.id], threshold)
    return total


def choose_threshold(
    rows: Sequence[Row], scores: Mapping[str, Sequence[Score]]
) -> float:
    """Most compression with recall >= MIN_RECALL, else most recall; ties go to the higher value."""
    swept = [(t, pooled(rows, scores, t)) for t in THRESHOLDS]
    passing = [(t, c) for t, c in swept if c.recall >= MIN_RECALL]
    if passing:
        return max(passing, key=lambda tc: (tc[1].compression, tc[0]))[0]
    return max(swept, key=lambda tc: (tc[1].recall, tc[1].compression, tc[0]))[0]


def score_all(rows: Sequence[Row], score_fn: ScoreFn) -> dict[str, list[Score]]:
    """Each chunk scored once with the generic question, keyed by row id."""
    return {
        row.id: [
            score_fn(chunk, GENERIC_QUESTION.format(goal=row.goal))
            for chunk in row.chunks
        ]
        for row in rows
    }


def _groups(rows: Sequence[Row]) -> dict[str, list[Row]]:
    groups = {
        goal_type: [r for r in rows if r.goal_type == goal_type]
        for goal_type in GOAL_TYPES
    }
    return {**groups, "all": list(rows)}


def metrics_at(
    rows: Sequence[Row], scores: Mapping[str, Sequence[Score]], threshold: float
) -> dict[str, Any]:
    """Per group (goal type and 'all'): tune, holdout and all-rows metrics at one threshold."""
    report: dict[str, Any] = {}
    for name, group in _groups(rows).items():
        entry = {
            split: pooled(
                [r for r in group if r.split == split], scores, threshold
            ).as_dict()
            for split in SPLITS
        }
        entry["all_rows"] = pooled(group, scores, threshold).as_dict()
        report[name] = entry
    return report


def evaluate(rows: Sequence[Row], score_fn: ScoreFn) -> dict[str, Any]:
    scores = score_all(rows, score_fn)
    tune = [r for r in rows if r.split == "tune"]
    chosen = choose_threshold(tune, scores)
    return {
        "chosen_threshold": chosen,
        "old_default": OLD_DEFAULT,
        "at_chosen": metrics_at(rows, scores, chosen),
        "at_old_default": metrics_at(rows, scores, OLD_DEFAULT),
        "sweep": {f"{t:.2f}": metrics_at(rows, scores, t) for t in THRESHOLDS},
    }


def _line(label: str, entry: Mapping[str, Any]) -> str:
    parts = [f"  {label:<14}"]
    for split in (*SPLITS, "all_rows"):
        m = entry[split]
        parts.append(
            f"{split} R {m['recall']:.2f} P {m['precision']:.2f}"
            f" C {m['compression']:.2f} (n={m['chunks']})"
        )
    return "  ".join(parts)


def format_report(report: Mapping[str, Any]) -> str:
    lines = ["pooled sweep (R recall, P precision, C compression)"]
    for threshold, body in report["sweep"].items():
        lines.append(_line(f"thr {threshold}", body["all"]))
    for key, title in (
        ("at_old_default", f"at old default {report['old_default']:.2f}"),
        ("at_chosen", f"at chosen {report['chosen_threshold']:.2f} (tune only)"),
    ):
        lines.append(title)
        lines.extend(_line(name, entry) for name, entry in report[key].items())
    return "\n".join(lines)


def live_score_fn() -> ScoreFn:
    """Score through the tool's own provider path."""
    from jev_mcp import app
    from jev_mcp.compaction import score_chunk

    def score_fn(chunk: str, prompt: str) -> Score:
        # An outage aborts the run: dropping unscored chunks would corrupt every number.
        return score_chunk(app.provider, 0, chunk, prompt)

    return score_fn


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate(load_dataset(args.dataset), live_score_fn())
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
