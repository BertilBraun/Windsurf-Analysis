from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from video_processing.inference.optimization.optimization_util import AssignmentKey


@dataclass(frozen=True)
class AssociationMetrics:
    observation_count: int
    covered_observation_count: int
    rejected_observation_count: int
    pair_true_positive: int
    pair_false_positive: int
    pair_false_negative: int
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    emitted_track_count: int
    scored_partition_count: int
    contaminated_track_count: int
    contaminated_observation_count: int
    gold_identity_count: int
    fragmented_identity_count: int
    fragmentation_excess: int
    exact_partition: bool


@dataclass(frozen=True)
class RejectionMetrics:
    observation_count: int
    gold_discard_count: int
    rejected_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def _choose_two(value: int) -> int:
    return value * (value - 1) // 2


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def association_metrics(
    gold_assignment: dict[AssignmentKey, int],
    predicted_assignment: dict[AssignmentKey, int],
    *,
    discard_track_id: int = 0,
) -> AssociationMetrics:
    extra_keys = set(predicted_assignment) - set(gold_assignment)
    if extra_keys:
        raise ValueError(f'Predictions contain {len(extra_keys)} observations outside the evaluation universe.')

    kept_keys = sorted(key for key, gold_id in gold_assignment.items() if gold_id != discard_track_id)
    covered_keys = [key for key in kept_keys if key in predicted_assignment]
    rejected_count = len(kept_keys) - len(covered_keys)

    next_singleton_id = max(predicted_assignment.values(), default=0) + 1
    aligned_prediction: dict[AssignmentKey, int] = {}
    for key in kept_keys:
        if key in predicted_assignment:
            aligned_prediction[key] = predicted_assignment[key]
        else:
            aligned_prediction[key] = next_singleton_id
            next_singleton_id += 1

    gold_sizes = Counter(gold_assignment[key] for key in kept_keys)
    predicted_sizes = Counter(aligned_prediction.values())
    emitted_prediction_ids = {predicted_assignment[key] for key in covered_keys}
    contingency = Counter((gold_assignment[key], aligned_prediction[key]) for key in kept_keys)

    true_positive = sum(_choose_two(count) for count in contingency.values())
    predicted_positive = sum(_choose_two(count) for count in predicted_sizes.values())
    gold_positive = sum(_choose_two(count) for count in gold_sizes.values())
    false_positive = predicted_positive - true_positive
    false_negative = gold_positive - true_positive
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    gold_ids_by_prediction: dict[int, set[int]] = defaultdict(set)
    observations_by_prediction: Counter[int] = Counter()
    prediction_ids_by_gold: dict[int, set[int]] = defaultdict(set)
    for key in kept_keys:
        prediction_id = aligned_prediction[key]
        gold_id = gold_assignment[key]
        gold_ids_by_prediction[prediction_id].add(gold_id)
        observations_by_prediction[prediction_id] += 1
        prediction_ids_by_gold[gold_id].add(prediction_id)

    contaminated_ids = {
        prediction_id for prediction_id, gold_ids in gold_ids_by_prediction.items() if len(gold_ids) > 1
    }
    fragmented_gold_ids = {
        gold_id for gold_id, prediction_ids in prediction_ids_by_gold.items() if len(prediction_ids) > 1
    }
    fragmentation_excess = sum(len(prediction_ids) - 1 for prediction_ids in prediction_ids_by_gold.values())
    exact_partition = (
        rejected_count == 0
        and not contaminated_ids
        and not fragmented_gold_ids
        and len(predicted_sizes) == len(gold_sizes)
    )

    return AssociationMetrics(
        observation_count=len(kept_keys),
        covered_observation_count=len(covered_keys),
        rejected_observation_count=rejected_count,
        pair_true_positive=true_positive,
        pair_false_positive=false_positive,
        pair_false_negative=false_negative,
        pairwise_precision=precision,
        pairwise_recall=recall,
        pairwise_f1=f1,
        emitted_track_count=len(emitted_prediction_ids),
        scored_partition_count=len(predicted_sizes),
        contaminated_track_count=len(contaminated_ids),
        contaminated_observation_count=sum(
            observations_by_prediction[prediction_id] for prediction_id in contaminated_ids
        ),
        gold_identity_count=len(gold_sizes),
        fragmented_identity_count=len(fragmented_gold_ids),
        fragmentation_excess=fragmentation_excess,
        exact_partition=exact_partition,
    )


def rejection_metrics(
    gold_assignment: dict[AssignmentKey, int],
    predicted_assignment: dict[AssignmentKey, int],
    *,
    discard_track_id: int = 0,
) -> RejectionMetrics:
    extra_keys = set(predicted_assignment) - set(gold_assignment)
    if extra_keys:
        raise ValueError(f'Predictions contain {len(extra_keys)} observations outside the evaluation universe.')

    gold_discard = {key for key, gold_id in gold_assignment.items() if gold_id == discard_track_id}
    rejected = set(gold_assignment) - set(predicted_assignment)
    true_positive = len(gold_discard & rejected)
    false_positive = len(rejected - gold_discard)
    false_negative = len(gold_discard - rejected)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RejectionMetrics(
        observation_count=len(gold_assignment),
        gold_discard_count=len(gold_discard),
        rejected_count=len(rejected),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )
