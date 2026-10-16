"""Compare relevance questions for jev_compact_context on a labelled goal set.

    python -m tests.evals.compaction_eval [--dataset PATH] [--json]

Each dataset line has `id`, `goal_type` (open_items | find_fact | continue_work), `split`
(tune | holdout), `goal`, `state` ({key: text}), `relevant` (keys) and `note`. The fast engine
scores every chunk once per distinct question; thresholds are then swept offline. Each goal
type's threshold is chosen on the tune split only (highest compression with recall >= 0.95, else
the highest recall) and the holdout numbers are reported at that frozen threshold. It measures;
it does not gate. The fast daemon must be running.
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_mcp.compaction import (
    GENERIC_QUESTION,
    GOAL_TYPES,
    OTHER_GOAL,
    TYPED_QUESTIONS,
    TYPED_THRESHOLDS,
    chunk_state,
    classify_goal,
)

DEFAULT_DATASET = Path(__file__).with_name("compaction_goals.jsonl")
SPLITS = ("tune", "holdout")
REQUIRED_KEYS = ("id", "goal_type", "split", "goal", "state", "relevant")
THRESHOLDS = tuple(round(0.10 + 0.05 * step, 2) for step in range(13))
MIN_RECALL = 0.95
DEFAULT_THRESHOLD = 0.3

# Two candidate wordings per goal type; the one shipped in jev_mcp.compaction was picked on tune.
CANDIDATES: Mapping[str, Mapping[str, str]] = {
    "open_items": {
        "a": "Does this text describe something that is still unresolved, pending, blocked or not yet done (an open question, a remaining task or an unfinished item) relevant to: '{goal}'?",
        "b": "Is this snippet about work or questions that remain open and still need action, rather than something already finished or decided? Goal: '{goal}'",
    },
    "find_fact": {
        "a": "Does this text state a specific fact, value, name, number or decision that directly answers: '{goal}'?",
        "b": "Could the answer to '{goal}' be found in this snippet?",
    },
    "continue_work": {
        "a": "Would someone resuming the task '{goal}' need this text: its current state, decisions made, constraints or next steps?",
        "b": "Is this snippet about the same task as '{goal}' (its progress, files, decisions or next steps) rather than an unrelated topic?",
    },
}

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
    rows: Sequence[Row],
    scores: Mapping[str, Sequence[Score]],
    threshold_for: Callable[[Row], float],
) -> Counts:
    total = Counts()
    for row in rows:
        total += counts_for(row, scores[row.id], threshold_for(row))
    return total


def choose_threshold(
    rows: Sequence[Row], scores: Mapping[str, Sequence[Score]]
) -> float:
    """The tune-split threshold: most compression with recall >= MIN_RECALL, else most recall."""
    swept = [(t, pooled(rows, scores, lambda _row, t=t: t)) for t in THRESHOLDS]
    passing = [(t, c) for t, c in swept if c.recall >= MIN_RECALL]
    if passing:
        # Compression only grows with the threshold; on a tie keep the lower, safer one.
        return max(passing, key=lambda tc: (tc[1].compression, -tc[0]))[0]
    return max(swept, key=lambda tc: (tc[1].recall, tc[1].compression, -tc[0]))[0]


def score_all(
    rows: Sequence[Row],
    score_fn: ScoreFn,
    classify_fn: Callable[[str], str],
    shipped: Mapping[str, str],
) -> tuple[dict[str, dict[str, list[Score]]], dict[str, str]]:
    """Per-variant chunk scores by row id, and the goal type the classifier chose per row.

    Each (chunk, question) pair is sent to the engine once, however many variants share it.
    """
    cache: dict[tuple[str, str], Score] = {}

    def scored(row: Row, template: str) -> list[Score]:
        prompt = template.format(goal=row.goal)
        out = []
        for chunk in row.chunks:
            if (chunk, prompt) not in cache:
                cache[(chunk, prompt)] = score_fn(chunk, prompt)
            out.append(cache[(chunk, prompt)])
        return out

    predicted = {row.id: classify_fn(row.goal) for row in rows}
    variants: dict[str, dict[str, list[Score]]] = {
        name: {}
        for name in (
            "baseline",
            "oracle_a",
            "oracle_b",
            "typed_oracle",
            "typed_classified",
        )
    }
    for row in rows:
        variants["baseline"][row.id] = scored(row, GENERIC_QUESTION)
        variants["oracle_a"][row.id] = scored(row, CANDIDATES[row.goal_type]["a"])
        variants["oracle_b"][row.id] = scored(row, CANDIDATES[row.goal_type]["b"])
        variants["typed_oracle"][row.id] = scored(row, shipped[row.goal_type])
        guess = predicted[row.id]
        template = shipped[guess] if guess in shipped else GENERIC_QUESTION
        variants["typed_classified"][row.id] = scored(row, template)
    return variants, predicted


def summarise(
    rows: Sequence[Row], scores: Mapping[str, Sequence[Score]]
) -> dict[str, Any]:
    """Per goal type: the tune-chosen threshold and tune/holdout metrics at that threshold."""
    report: dict[str, Any] = {}
    chosen: dict[str, float] = {}
    for goal_type in GOAL_TYPES:
        tune = [r for r in rows if r.goal_type == goal_type and r.split == "tune"]
        chosen[goal_type] = choose_threshold(tune, scores)
    for goal_type in (*GOAL_TYPES, "all"):
        group = [r for r in rows if goal_type in (r.goal_type, "all")]
        entry: dict[str, Any] = {
            "threshold": chosen.get(goal_type),
        }
        for split in SPLITS:
            part = [r for r in group if r.split == split]
            entry[split] = pooled(part, scores, lambda r: chosen[r.goal_type]).as_dict()
        report[goal_type] = entry
    return report


def fixed_report(
    rows: Sequence[Row],
    scores: Mapping[str, Sequence[Score]],
    threshold_for: Callable[[Row], float],
) -> dict[str, Any]:
    """Tune/holdout metrics at thresholds fixed in advance rather than chosen here."""
    report: dict[str, Any] = {}
    for goal_type in (*GOAL_TYPES, "all"):
        group = [r for r in rows if goal_type in (r.goal_type, "all")]
        report[goal_type] = {
            split: pooled(
                [r for r in group if r.split == split], scores, threshold_for
            ).as_dict()
            for split in SPLITS
        }
    return report


def evaluate(
    rows: Sequence[Row],
    score_fn: ScoreFn,
    classify_fn: Callable[[str], str],
    shipped: Mapping[str, str] = TYPED_QUESTIONS,
    shipped_thresholds: Mapping[str, float] = TYPED_THRESHOLDS,
) -> dict[str, Any]:
    variants, predicted = score_all(rows, score_fn, classify_fn, shipped)
    report = {name: summarise(rows, scores) for name, scores in variants.items()}
    report["baseline@default"] = fixed_report(
        rows, variants["baseline"], lambda _row: DEFAULT_THRESHOLD
    )
    # What goal_aware=True would do: the shipped threshold for the guessed type, else the default.
    report["typed_classified@shipped"] = fixed_report(
        rows,
        variants["typed_classified"],
        lambda row: shipped_thresholds.get(predicted[row.id], DEFAULT_THRESHOLD),
    )
    hits = sum(predicted[r.id] == r.goal_type for r in rows)
    report["classifier"] = {
        "correct": hits,
        "rows": len(rows),
        "other": sum(guess == OTHER_GOAL for guess in predicted.values()),
        "predicted": predicted,
    }
    return report


def format_report(report: Mapping[str, Any]) -> str:
    lines = []
    for name, body in report.items():
        if name == "classifier":
            lines.append(
                f"classifier  {body['correct']}/{body['rows']} correct, {body['other']} other"
            )
            continue
        lines.append(name)
        for goal_type, entry in body.items():
            threshold = entry.get("threshold")
            head = f"  {goal_type:<14}" + (
                f" thr {threshold:.2f}" if threshold else " " * 9
            )
            for split in SPLITS:
                m = entry[split]
                head += (
                    f"  {split} recall {m['recall']:.2f} prec {m['precision']:.2f}"
                    f" comp {m['compression']:.2f} (n={m['chunks']})"
                )
            lines.append(head)
    return "\n".join(lines)


def live_functions() -> tuple[ScoreFn, Callable[[str], str]]:
    """Score and classify through the tool's own provider path."""
    from jev_mcp import app
    from jev_mcp.compaction import score_chunk

    def score_fn(chunk: str, prompt: str) -> Score:
        # An outage aborts the run: dropping unscored chunks would corrupt every number.
        return score_chunk(app.provider, 0, chunk, prompt)

    return score_fn, lambda goal: classify_goal(app.provider, goal)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    score_fn, classify_fn = live_functions()
    report = evaluate(load_dataset(args.dataset), score_fn, classify_fn)
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
