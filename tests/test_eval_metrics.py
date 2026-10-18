"""Binary-classification metrics, checked against values worked out by hand."""

import pytest

from tests.evals import metrics
from tests.evals.metrics import (
    average_precision,
    best_f1,
    bootstrap_auc_ci,
    precision_recall_f1,
    roc_auc,
    threshold_sweep,
)

# labels [0, 0, 1, 1], scores [.1, .4, .35, .8]: 3 of the 4 pos/neg pairs rank right.
CLASSIC_LABELS = [False, False, True, True]
CLASSIC_SCORES = [0.1, 0.4, 0.35, 0.8]


def test_precision_recall_f1_at_a_threshold():
    # >= 0.4 predicts the .4 (neg) and .8 (pos): tp=1 fp=1 fn=1.
    result = precision_recall_f1(CLASSIC_LABELS, CLASSIC_SCORES, 0.4)

    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_the_threshold_is_inclusive():
    result = precision_recall_f1([True], [0.5], 0.5)

    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_no_predicted_positives_scores_zero_not_a_crash():
    result = precision_recall_f1(CLASSIC_LABELS, CLASSIC_SCORES, 0.99)

    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_no_actual_positives_scores_zero_recall():
    result = precision_recall_f1([False, False], [0.9, 0.1], 0.5)

    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_empty_input_scores_zero():
    assert precision_recall_f1([], [], 0.5) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_predictions_can_be_scored_directly():
    result = metrics.precision_recall_f1_from_predictions(
        [True, True, False, False], [True, False, True, False]
    )

    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_prediction_and_label_lengths_must_match():
    with pytest.raises(ValueError, match="length"):
        metrics.precision_recall_f1_from_predictions([True], [True, False])
    with pytest.raises(ValueError, match="length"):
        roc_auc([True], [0.1, 0.2])


def test_auc_matches_the_hand_counted_pairs():
    assert roc_auc(CLASSIC_LABELS, CLASSIC_SCORES) == pytest.approx(0.75)


def test_auc_of_a_perfect_ranking_is_one_and_a_reversed_one_is_zero():
    labels = [False, False, True, True]

    assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_of_an_interleaved_ranking_is_one_half():
    # pos {.1, .4} vs neg {.2, .3}: 2 of 4 pairs rank right.
    labels = [True, False, False, True]

    assert roc_auc(labels, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.5)


def test_tied_scores_count_half_a_pair():
    # pos {.5, .9} vs neg {.2, .5}: 1 + .5 + 1 + 1 = 3.5 of 4.
    labels = [False, True, False, True]

    assert roc_auc(labels, [0.2, 0.5, 0.5, 0.9]) == pytest.approx(0.875)
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5


def test_a_constant_score_has_an_auc_of_one_half():
    assert roc_auc([True, False, True, False], [0.3] * 4) == 0.5


@pytest.mark.parametrize(
    "labels, scores",
    [
        ([True, True], [0.1, 0.9]),
        ([False, False], [0.1, 0.9]),
        ([True], [0.5]),
        ([False], [0.5]),
        ([], []),
    ],
)
def test_auc_is_undefined_without_both_classes(labels, scores):
    assert roc_auc(labels, scores) is None


def test_average_precision_matches_the_hand_computation():
    # Ranked .8(P) .4(N) .35(P) .1(N): AP = .5 * 1 + .5 * 2/3.
    assert average_precision(CLASSIC_LABELS, CLASSIC_SCORES) == pytest.approx(5 / 6)


def test_average_precision_groups_tied_scores():
    # .9(P) then {.5(P), .5(N)} then .2(N): .5 * 1 + .5 * 2/3.
    labels = [False, True, False, True]

    assert average_precision(labels, [0.2, 0.5, 0.5, 0.9]) == pytest.approx(5 / 6)


def test_average_precision_of_a_perfect_ranking_is_one():
    assert average_precision([False, True, True], [0.1, 0.8, 0.9]) == 1.0


def test_average_precision_is_defined_when_everything_is_positive():
    assert average_precision([True, True], [0.2, 0.9]) == 1.0


def test_average_precision_is_undefined_without_positives():
    assert average_precision([False, False], [0.2, 0.9]) is None
    assert average_precision([], []) is None


def test_the_sweep_covers_0_05_to_0_95_in_steps_of_0_05():
    sweep = threshold_sweep(CLASSIC_LABELS, CLASSIC_SCORES)

    thresholds = [row["threshold"] for row in sweep]
    assert thresholds == [round(0.05 * step, 2) for step in range(1, 20)]
    at_half = next(row for row in sweep if row["threshold"] == 0.5)
    assert at_half == {"threshold": 0.5, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}


def test_best_f1_takes_the_highest_f1_and_the_lowest_threshold_on_a_tie():
    sweep = [
        {"threshold": 0.1, "precision": 0.5, "recall": 1.0, "f1": 0.6},
        {"threshold": 0.2, "precision": 0.7, "recall": 0.9, "f1": 0.8},
        {"threshold": 0.3, "precision": 0.8, "recall": 0.8, "f1": 0.8},
    ]

    assert best_f1(sweep)["threshold"] == 0.2


def test_best_f1_of_an_empty_sweep_is_none():
    assert best_f1([]) is None


def test_the_bootstrap_interval_is_reproducible_for_a_fixed_seed():
    first = bootstrap_auc_ci(CLASSIC_LABELS * 5, CLASSIC_SCORES * 5, seed=7)
    second = bootstrap_auc_ci(CLASSIC_LABELS * 5, CLASSIC_SCORES * 5, seed=7)

    assert first == second
    assert first is not None
    low, high = first
    assert 0.0 <= low <= 0.75 <= high <= 1.0


def test_the_bootstrap_interval_of_a_perfect_ranking_is_one_to_one():
    labels = [False, False, True, True]

    assert bootstrap_auc_ci(labels, [0.1, 0.2, 0.8, 0.9]) == (1.0, 1.0)


def test_a_small_sample_gives_a_wide_interval():
    labels = [True, False, True, False, True, False, True, False]
    scores = [0.9, 0.2, 0.4, 0.5, 0.8, 0.1, 0.3, 0.6]

    low, high = bootstrap_auc_ci(labels, scores)  # type: ignore[misc]

    assert high - low > 0.2


@pytest.mark.parametrize(
    "labels, scores",
    [([True, True], [0.1, 0.9]), ([False], [0.5]), ([], [])],
)
def test_the_bootstrap_interval_is_undefined_without_both_classes(labels, scores):
    assert bootstrap_auc_ci(labels, scores) is None
