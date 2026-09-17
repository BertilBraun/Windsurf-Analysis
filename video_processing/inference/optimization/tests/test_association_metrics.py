from __future__ import annotations

import pytest

from video_processing.inference.optimization.association_metrics import association_metrics, rejection_metrics
from video_processing.inference.optimization.optimization_util import AssignmentKey


def _key(frame_index: int) -> AssignmentKey:
    return AssignmentKey(frame_index, frame_index, 0, frame_index + 1, 1)


def test_association_metrics_are_label_permutation_invariant() -> None:
    gold = {_key(0): 1, _key(1): 1, _key(2): 2, _key(3): 2}
    prediction = {_key(0): 9, _key(1): 9, _key(2): 4, _key(3): 4}

    result = association_metrics(gold, prediction)

    assert result.pairwise_precision == 1.0
    assert result.pairwise_recall == 1.0
    assert result.exact_partition


def test_catastrophic_merge_and_fragmentation_are_counted() -> None:
    gold = {_key(0): 1, _key(1): 1, _key(2): 2, _key(3): 2}
    prediction = {_key(0): 7, _key(1): 8, _key(2): 7, _key(3): 7}

    result = association_metrics(gold, prediction)

    assert result.contaminated_track_count == 1
    assert result.contaminated_observation_count == 3
    assert result.fragmented_identity_count == 1
    assert result.fragmentation_excess == 1
    assert not result.exact_partition


@pytest.mark.parametrize(
    ('prediction_labels', 'expected_true_positive', 'expected_false_positive', 'expected_false_negative'),
    [
        ([4, 4, 8, 8], 2, 0, 0),
        ([4, 4, 4, 4], 2, 4, 0),
        ([4, 5, 8, 9], 0, 0, 2),
        ([4, 5, 4, 8], 0, 1, 2),
    ],
)
def test_pair_counts_match_brute_force_examples(
    prediction_labels: list[int],
    expected_true_positive: int,
    expected_false_positive: int,
    expected_false_negative: int,
) -> None:
    keys = [_key(frame_index) for frame_index in range(4)]
    gold = dict(zip(keys, [1, 1, 2, 2]))
    prediction = dict(zip(keys, prediction_labels))

    result = association_metrics(gold, prediction)

    assert result.pair_true_positive == expected_true_positive
    assert result.pair_false_positive == expected_false_positive
    assert result.pair_false_negative == expected_false_negative


def test_missing_observations_are_unique_singletons() -> None:
    gold = {_key(0): 1, _key(1): 1, _key(2): 1}

    result = association_metrics(gold, {})

    assert result.covered_observation_count == 0
    assert result.rejected_observation_count == 3
    assert result.pairwise_recall == 0.0
    assert result.fragmentation_excess == 2


def test_numeric_prediction_zero_is_not_implicitly_rejected() -> None:
    gold = {_key(0): 0, _key(1): 1}
    prediction = {_key(0): 12, _key(1): 0}

    result = rejection_metrics(gold, prediction)

    assert result.rejected_count == 0
    assert result.false_negative == 1


def test_extra_prediction_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match='outside the evaluation universe'):
        association_metrics({_key(0): 1}, {_key(0): 1, _key(1): 1})
