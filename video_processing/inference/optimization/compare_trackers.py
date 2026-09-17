from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib.metadata import version
from pathlib import Path

import boxmot
from pydantic import BaseModel, ConfigDict


this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.optimization.association_benchmark_data import load_golden_sequences
from video_processing.inference.optimization.association_benchmark_trackers import (
    BenchmarkTracker,
    TrackerPrediction,
    run_tracker,
)
from video_processing.inference.optimization.association_metrics import association_metrics, rejection_metrics
from video_processing.inference.optimization.optimization_util import AssignmentKey


EXPECTED_BOXMOT_DISTRIBUTION_VERSION = '13.0.17'


class VideoResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    sequence: str
    tracker: BenchmarkTracker
    runtime_seconds: float
    observation_count: int
    kept_observation_count: int
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
    gold_discard_count: int
    rejection_true_positive: int
    rejection_false_positive: int
    rejection_false_negative: int
    rejection_precision: float
    rejection_recall: float
    rejection_f1: float


class AggregateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tracker: BenchmarkTracker
    video_count: int
    runtime_seconds: float
    observation_count: int
    kept_observation_count: int
    coverage: float
    pairwise_precision_micro: float
    pairwise_recall_micro: float
    pairwise_f1_micro: float
    pairwise_precision_macro: float
    pairwise_recall_macro: float
    pairwise_f1_macro: float
    emitted_track_count: int
    contaminated_track_count: int
    contaminated_track_rate: float
    contaminated_observation_count: int
    fragmented_identity_count: int
    fragmentation_excess: int
    exact_video_count: int
    exact_video_rate: float
    gold_discard_count: int
    rejection_precision_micro: float
    rejection_recall_micro: float
    rejection_f1_micro: float


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    protocol: str
    dataset_scope: str
    boxmot_declared_version: str
    boxmot_reported_version: str
    development_set_warning: str
    observation_provenance_warning: str
    tracker_configurations: dict[str, str]
    per_video: list[VideoResult]
    aggregate: list[AggregateResult]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _video_result(
    sequence_name: str,
    prediction: TrackerPrediction,
    gold_assignment: dict[AssignmentKey, int],
) -> VideoResult:
    association = association_metrics(gold_assignment, prediction.assignment)
    rejection = rejection_metrics(gold_assignment, prediction.assignment)
    return VideoResult(
        sequence=sequence_name,
        tracker=prediction.tracker,
        runtime_seconds=prediction.runtime_seconds,
        observation_count=rejection.observation_count,
        kept_observation_count=association.observation_count,
        covered_observation_count=association.covered_observation_count,
        rejected_observation_count=association.rejected_observation_count,
        pair_true_positive=association.pair_true_positive,
        pair_false_positive=association.pair_false_positive,
        pair_false_negative=association.pair_false_negative,
        pairwise_precision=association.pairwise_precision,
        pairwise_recall=association.pairwise_recall,
        pairwise_f1=association.pairwise_f1,
        emitted_track_count=association.emitted_track_count,
        scored_partition_count=association.scored_partition_count,
        contaminated_track_count=association.contaminated_track_count,
        contaminated_observation_count=association.contaminated_observation_count,
        gold_identity_count=association.gold_identity_count,
        fragmented_identity_count=association.fragmented_identity_count,
        fragmentation_excess=association.fragmentation_excess,
        exact_partition=association.exact_partition,
        gold_discard_count=rejection.gold_discard_count,
        rejection_true_positive=rejection.true_positive,
        rejection_false_positive=rejection.false_positive,
        rejection_false_negative=rejection.false_negative,
        rejection_precision=rejection.precision,
        rejection_recall=rejection.recall,
        rejection_f1=rejection.f1,
    )


def _aggregate(tracker: BenchmarkTracker, results: list[VideoResult]) -> AggregateResult:
    tracker_results = [result for result in results if result.tracker == tracker]
    if not tracker_results:
        raise ValueError(f'No results for {tracker}.')
    pair_true_positive = sum(result.pair_true_positive for result in tracker_results)
    pair_false_positive = sum(result.pair_false_positive for result in tracker_results)
    pair_false_negative = sum(result.pair_false_negative for result in tracker_results)
    pair_precision = _safe_ratio(pair_true_positive, pair_true_positive + pair_false_positive)
    pair_recall = _safe_ratio(pair_true_positive, pair_true_positive + pair_false_negative)
    pair_f1 = (
        2.0 * pair_precision * pair_recall / (pair_precision + pair_recall) if pair_precision + pair_recall else 0.0
    )

    rejection_true_positive = sum(result.rejection_true_positive for result in tracker_results)
    rejection_false_positive = sum(result.rejection_false_positive for result in tracker_results)
    rejection_false_negative = sum(result.rejection_false_negative for result in tracker_results)
    rejection_precision = _safe_ratio(rejection_true_positive, rejection_true_positive + rejection_false_positive)
    rejection_recall = _safe_ratio(rejection_true_positive, rejection_true_positive + rejection_false_negative)
    rejection_f1 = (
        2.0 * rejection_precision * rejection_recall / (rejection_precision + rejection_recall)
        if rejection_precision + rejection_recall
        else 0.0
    )
    kept_observations = sum(result.kept_observation_count for result in tracker_results)
    covered_observations = sum(result.covered_observation_count for result in tracker_results)
    exact_video_count = sum(result.exact_partition for result in tracker_results)
    video_count = len(tracker_results)
    emitted_track_count = sum(result.emitted_track_count for result in tracker_results)
    contaminated_track_count = sum(result.contaminated_track_count for result in tracker_results)
    return AggregateResult(
        tracker=tracker,
        video_count=video_count,
        runtime_seconds=sum(result.runtime_seconds for result in tracker_results),
        observation_count=sum(result.observation_count for result in tracker_results),
        kept_observation_count=kept_observations,
        coverage=_safe_ratio(covered_observations, kept_observations),
        pairwise_precision_micro=pair_precision,
        pairwise_recall_micro=pair_recall,
        pairwise_f1_micro=pair_f1,
        pairwise_precision_macro=sum(result.pairwise_precision for result in tracker_results) / video_count,
        pairwise_recall_macro=sum(result.pairwise_recall for result in tracker_results) / video_count,
        pairwise_f1_macro=sum(result.pairwise_f1 for result in tracker_results) / video_count,
        emitted_track_count=emitted_track_count,
        contaminated_track_count=contaminated_track_count,
        contaminated_track_rate=_safe_ratio(contaminated_track_count, emitted_track_count),
        contaminated_observation_count=sum(result.contaminated_observation_count for result in tracker_results),
        fragmented_identity_count=sum(result.fragmented_identity_count for result in tracker_results),
        fragmentation_excess=sum(result.fragmentation_excess for result in tracker_results),
        exact_video_count=exact_video_count,
        exact_video_rate=exact_video_count / video_count,
        gold_discard_count=sum(result.gold_discard_count for result in tracker_results),
        rejection_precision_micro=rejection_precision,
        rejection_recall_micro=rejection_recall,
        rejection_f1_micro=rejection_f1,
    )


def _write_csv(output_path: Path, results: list[VideoResult]) -> None:
    rows = [result.model_dump(mode='json') for result in results]
    with output_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint(checkpoint_path: Path) -> list[VideoResult]:
    if not checkpoint_path.is_file():
        return []
    payload = json.loads(checkpoint_path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'Invalid checkpoint in {checkpoint_path}.')
    return [VideoResult.model_validate(item) for item in payload]


def _write_checkpoint(checkpoint_path: Path, results: list[VideoResult]) -> None:
    checkpoint_path.write_text(
        json.dumps([result.model_dump(mode='json') for result in results], indent=2) + '\n',
        encoding='utf-8',
    )


def _write_markdown(output_path: Path, report: BenchmarkReport) -> None:
    lines = [
        '# Fixed-observation association benchmark',
        '',
        report.protocol,
        '',
        f'**Dataset:** {report.dataset_scope}',
        '',
        f'**Development-set warning:** {report.development_set_warning}',
        '',
        f'**Observation provenance:** {report.observation_provenance_warning}',
        '',
        '## Aggregate results',
        '',
        '| Tracker | Coverage | Pair precision (micro) | Pair recall (micro) | Pair F1 (micro / macro) | Contaminated tracks | Fragmentation excess | Exact videos | Runtime |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for result in report.aggregate:
        lines.append(
            f'| `{result.tracker.value}` | {result.coverage:.3f} | {result.pairwise_precision_micro:.3f} | '
            f'{result.pairwise_recall_micro:.3f} | {result.pairwise_f1_micro:.3f} / '
            f'{result.pairwise_f1_macro:.3f} | {result.contaminated_track_count}/{result.emitted_track_count} '
            f'({result.contaminated_track_rate:.1%}) | '
            f'{result.fragmentation_excess} | {result.exact_video_count}/{result.video_count} | '
            f'{result.runtime_seconds:.1f} s |'
        )
    lines.extend(
        [
            '',
            '## Metric definitions',
            '',
            '- Pairwise precision/recall/F1 compare whether every pair of kept observations is assigned to the same identity. Missing outputs become unique singleton predictions.',
            '- Micro pair metrics pool pair counts and therefore weight long identities quadratically. Macro metrics average the per-video scores.',
            '- A contaminated predicted track contains observations from more than one gold identity. This is the catastrophic false-merge error for the application.',
            '- Fragmentation excess is the sum, over gold identities, of the number of predicted pieces beyond one.',
            '- An exact video has full kept-observation coverage and a partition identical to gold, up to permutation of track labels.',
            '- Rejection metrics treat an omitted observation as rejected and gold identity 0 as the manually marked discard class. Tracker numeric label 0 has no special meaning.',
            '',
            '## Findings',
            '',
            '- The production pipeline has the strongest overall association F1, much less fragmentation, and the most exact videos.',
            '- It does not satisfy the application’s nominal zero-false-merge requirement on this reconstruction: 20 of 86 emitted tracks are identity-contaminated, spanning seven videos.',
            '- OC-SORT is conservative: it has the best pairwise precision but fragments the 88 gold identities into hundreds of pieces.',
            '- BoT-SORT reduces fragmentation relative to OC-SORT but still trails the production pipeline on recall, F1, and exact-video rate.',
            '- These results document the intended offline-association trade-off—far fewer splits at the cost of more false merges—but do not support a claim of uniformly superior or near-perfect tracking.',
            '',
            '## Reproduction',
            '',
            'From the repository root, install `requirements.txt` and run:',
            '',
            '```powershell',
            'python -m video_processing.inference.optimization.compare_trackers --no-resume',
            '```',
            '',
            f'The pinned BoxMOT distribution is {report.boxmot_declared_version}; its package-level version string reports {report.boxmot_reported_version}. A checkpoint is written under `tmp/` after every completed video-method pair.',
            '',
            '## Tracker configurations',
            '',
        ]
    )
    lines.extend(f'- **{name}:** {configuration}' for name, configuration in report.tracker_configurations.items())
    lines.extend(
        [
            '',
            '## Limitations',
            '',
            '- This is not a detector benchmark and does not support MOTA or HOTA claims. Every method receives the same saved observations.',
            '- The manually curated files are the development set. The production tracker configuration is frozen for this run, but the provenance of its numerical tuning is uncertain.',
            '- The saved non-interpolated boxes were emitted after gold-track reconstruction and RTS smoothing. The benchmark therefore measures association conditional on preselected, gold-derived smoothed observations.',
            '- The historical preprocessor fragments are not preserved. The production pipeline starts from singleton saved observations and reruns preprocessing before ILP association.',
            '- BoT-SORT is run without ReID because its standard appearance model is pedestrian-specific. It retains its own ECC camera-motion compensation.',
            '- Runtime is wall-clock method execution on this machine. OC-SORT needs no video decode; BoT-SORT includes video decode and ECC GMC; production includes masked stabilization and association but excludes initial crop embedding extraction. Runtime values are therefore operational diagnostics, not a speed ranking.',
            '',
            'See `benchmark_results.json` and `per_video.csv` for exact machine-readable values.',
        ]
    )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _tracker_configurations() -> dict[str, str]:
    return {
        BenchmarkTracker.OC_SORT.value: 'BoxMOT OC-SORT defaults frozen explicitly: min_conf=0.1, det_thresh=0.2, max_age=30, min_hits=3, IoU threshold=0.3, delta_t=3, inertia=0.2, BYTE disabled, Q_xy=0.01, Q_s=0.0001.',
        BenchmarkTracker.BOT_SORT.value: 'BoxMOT BoT-SORT with ECC GMC and ReID disabled; thresholds 0.5/0.1/0.6, buffer=30, match=0.8, fuse-score disabled.',
        BenchmarkTracker.PRODUCTION.value: 'Current TrackPreProcessor followed by current ILPTracker defaults; masked VidStab transforms with 20 px observation masks; no refitting.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate fixed-observation identity association against golden tracks.'
    )
    parser.add_argument('--golden-dir', type=Path, default=Path('tmp/golden2'))
    parser.add_argument('--output-dir', type=Path, default=Path('documentation/tracker_evaluation'))
    parser.add_argument(
        '--trackers',
        type=str,
        default=','.join(tracker.value for tracker in BenchmarkTracker),
        help='Comma-separated tracker identifiers.',
    )
    parser.add_argument('--sequences', type=str, default='', help='Optional comma-separated sequence names.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    arguments = parser.parse_args()

    selected_trackers = [BenchmarkTracker(value.strip()) for value in arguments.trackers.split(',') if value.strip()]
    selected_sequences = {value.strip() for value in arguments.sequences.split(',') if value.strip()}
    sequences = load_golden_sequences(arguments.golden_dir)
    if selected_sequences:
        sequences = [sequence for sequence in sequences if sequence.name in selected_sequences]
    if not sequences:
        raise ValueError('No sequences selected.')

    installed_boxmot_version = version('boxmot')
    if installed_boxmot_version != EXPECTED_BOXMOT_DISTRIBUTION_VERSION:
        raise RuntimeError(f'Expected BoxMOT {EXPECTED_BOXMOT_DISTRIBUTION_VERSION}, found {installed_boxmot_version}.')

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path('tmp/tracker_evaluation_checkpoint.json')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results = _load_checkpoint(checkpoint_path) if arguments.resume else []
    selected_sequence_names = {sequence.name for sequence in sequences}
    results = [
        result
        for result in results
        if result.sequence in selected_sequence_names and result.tracker in selected_trackers
    ]
    completed = {(result.sequence, result.tracker) for result in results}
    for sequence in sequences:
        for tracker in selected_trackers:
            if (sequence.name, tracker) in completed:
                print(f'Reusing checkpoint for {tracker.value} on {sequence.name}.', flush=True)
                continue
            print(f'Running {tracker.value} on {sequence.name}...', flush=True)
            prediction = run_tracker(tracker, sequence)
            result = _video_result(sequence.name, prediction, sequence.gold_assignment)
            results.append(result)
            _write_checkpoint(checkpoint_path, results)
            print(
                f'  F1={result.pairwise_f1:.4f} contaminated={result.contaminated_track_count} '
                f'fragmentation={result.fragmentation_excess} exact={result.exact_partition}',
                flush=True,
            )

    aggregates = [_aggregate(tracker, results) for tracker in selected_trackers]
    report = BenchmarkReport(
        protocol='Identity association on the fixed universe of non-interpolated saved observations. Gold identity 0 is excluded from association scoring and retained for rejection scoring.',
        dataset_scope=f'{len(sequences)} manually curated development videos; no held-out split and no parameter refitting.',
        boxmot_declared_version=installed_boxmot_version,
        boxmot_reported_version=str(boxmot.__version__),
        development_set_warning='In-sample/development-set comparison. Algorithm design may have been informed by these videos, and the provenance of ILP parameter tuning is uncertain.',
        observation_provenance_warning='Saved source observations were preselected and their boxes were RTS-smoothed after manual identity reconstruction; results are conditional association scores, not end-to-end MOT scores.',
        tracker_configurations=_tracker_configurations(),
        per_video=results,
        aggregate=aggregates,
    )
    (arguments.output_dir / 'benchmark_results.json').write_text(
        json.dumps(report.model_dump(mode='json'), indent=2) + '\n', encoding='utf-8'
    )
    _write_csv(arguments.output_dir / 'per_video.csv', results)
    _write_markdown(arguments.output_dir / 'README.md', report)


if __name__ == '__main__':
    main()
