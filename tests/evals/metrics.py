"""Dependency-free binary-classification metrics for the eval harnesses.

Labels are booleans (True = positive); scores are floats where higher means "more positive".
A prediction is positive when `score >= threshold`. Precision, recall and F1 follow the
zero-division-is-zero convention, so an empty or one-sided input scores 0.0 instead of raising.
ROC-AUC and its interval return None when only one class is present, because they are undefined
there. Average precision needs at least one positive.
"""

import random
from collections.abc import Sequence

SWEEP_STEPS = 19  # 0.05 .. 0.95 in steps of 0.05
SWEEP_STEP = 0.05
DEFAULT_BOOTSTRAP_ROUNDS = 1000
DEFAULT_SEED = 0
CI_LEVEL = 0.95

Scores = dict[str, float]


def _check_lengths(labels: Sequence[bool], other: Sequence[object]) -> None:
    if len(labels) != len(other):
        raise ValueError(f"length mismatch: {len(labels)} labels, {len(other)} values")


def precision_recall_f1_from_predictions(
    labels: Sequence[bool], predicted: Sequence[bool]
) -> Scores:
    """Precision, recall and F1 of boolean predictions against boolean labels."""
    _check_lengths(labels, predicted)
    true_pos = sum(a and p for a, p in zip(labels, predicted))
    predicted_pos = sum(predicted)
    actual_pos = sum(labels)
    precision = true_pos / predicted_pos if predicted_pos else 0.0
    recall = true_pos / actual_pos if actual_pos else 0.0
    total = precision + recall
    f1 = 2 * precision * recall / total if total else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def precision_recall_f1(
    labels: Sequence[bool], scores: Sequence[float], threshold: float
) -> Scores:
    """Precision, recall and F1 when `score >= threshold` counts as a positive prediction."""
    _check_lengths(labels, scores)
    return precision_recall_f1_from_predictions(
        labels, [score >= threshold for score in scores]
    )


def _average_ranks(scores: Sequence[float]) -> list[float]:
    """1-based ranks, with tied scores sharing the mean of the ranks they span."""
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        mean_rank = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = mean_rank
        start = end + 1
    return ranks


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.

    Ties share an average rank, so each tied positive/negative pair counts one half. None when
    either class is missing.
    """
    _check_lengths(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    u_statistic = positive_rank_sum - positives * (positives + 1) / 2
    return u_statistic / (positives * negatives)


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Area under the precision-recall step curve: sum of (R_n - R_n-1) * P_n.

    Each distinct score is one operating point, so tied scores enter together. None without
    positives.
    """
    _check_lengths(labels, scores)
    actual_pos = sum(labels)
    if actual_pos == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_pos = 0
    seen = 0
    previous_recall = 0.0
    total = 0.0
    index = 0
    while index < len(ranked):
        cut = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == cut:
            true_pos += ranked[index][1]
            seen += 1
            index += 1
        recall = true_pos / actual_pos
        total += (recall - previous_recall) * (true_pos / seen)
        previous_recall = recall
    return total


def threshold_sweep(
    labels: Sequence[bool], scores: Sequence[float]
) -> list[dict[str, float]]:
    """Precision, recall and F1 at thresholds 0.05, 0.10, ..., 0.95."""
    return [
        {"threshold": threshold, **precision_recall_f1(labels, scores, threshold)}
        for threshold in (
            round(SWEEP_STEP * step, 2) for step in range(1, SWEEP_STEPS + 1)
        )
    ]


def best_f1(sweep: Sequence[dict[str, float]]) -> dict[str, float] | None:
    """The sweep row with the highest F1; the lowest threshold wins a tie."""
    if not sweep:
        return None
    return max(sweep, key=lambda row: (row["f1"], -row["threshold"]))


def bootstrap_auc_ci(
    labels: Sequence[bool],
    scores: Sequence[float],
    rounds: int = DEFAULT_BOOTSTRAP_ROUNDS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float] | None:
    """Percentile 95% interval for ROC-AUC from resampling rows with replacement.

    A fixed seed makes it reproducible. Resamples that draw a single class are skipped; None
    when the data has one class or no resample is usable.
    """
    _check_lengths(labels, scores)
    if roc_auc(labels, scores) is None:
        return None
    rng = random.Random(seed)
    size = len(labels)
    estimates: list[float] = []
    for _ in range(rounds):
        picks = [rng.randrange(size) for _ in range(size)]
        estimate = roc_auc([labels[i] for i in picks], [scores[i] for i in picks])
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:  # pragma: no cover - needs every resample to be one class
        return None
    estimates.sort()
    tail = (1 - CI_LEVEL) / 2
    low = estimates[int(tail * (len(estimates) - 1))]
    high = estimates[int((1 - tail) * (len(estimates) - 1))]
    return (low, high)
