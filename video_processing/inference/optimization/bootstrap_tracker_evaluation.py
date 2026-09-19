from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .association_benchmark_types import BenchmarkTracker


DEFAULT_INPUT_PATH = Path('documentation/tracker_evaluation/per_video.csv')
DEFAULT_OUTPUT_PATH = Path('documentation/tracker_evaluation/bootstrap_intervals.json')
DEFAULT_SAMPLE_COUNT = 10_000
DEFAULT_SEED = 20_260_918


@dataclass(frozen=True)
class PairCounts:
    true_positive: int
    false_positive: int
    false_negative: int


class ConfidenceInterval(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tracker: BenchmarkTracker
    lower: float
    upper: float


class BootstrapReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    resampling_unit: str
    statistic: str
    sample_count: int
    rng: str
    seed: int
    interval: str
    confidence_intervals: tuple[ConfidenceInterval, ...]


def _column_indices(header: list[str]) -> tuple[int, int, int, int]:
    required = ('tracker', 'pair_true_positive', 'pair_false_positive', 'pair_false_negative')
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f'Missing required CSV columns: {missing}')
    return (
        header.index(required[0]),
        header.index(required[1]),
        header.index(required[2]),
        header.index(required[3]),
    )


def load_counts(input_path: Path) -> dict[BenchmarkTracker, list[PairCounts]]:
    counts_by_tracker: dict[BenchmarkTracker, list[PairCounts]] = {tracker: [] for tracker in BenchmarkTracker}
    with input_path.open(newline='', encoding='utf-8') as stream:
        rows = csv.reader(stream)
        header = next(rows)
        tracker_index, true_positive_index, false_positive_index, false_negative_index = _column_indices(header)
        for row in rows:
            tracker = BenchmarkTracker(row[tracker_index])
            counts_by_tracker[tracker].append(
                PairCounts(
                    true_positive=int(row[true_positive_index]),
                    false_positive=int(row[false_positive_index]),
                    false_negative=int(row[false_negative_index]),
                )
            )
    if not counts_by_tracker or any(not counts for counts in counts_by_tracker.values()):
        raise ValueError(f'No complete tracker results found in {input_path}.')
    return counts_by_tracker


def bootstrap_micro_f1(
    counts: list[PairCounts],
    *,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if sample_count <= 0:
        raise ValueError('sample_count must be positive.')
    values: NDArray[np.int64] = np.array(
        [(count.true_positive, count.false_positive, count.false_negative) for count in counts],
        dtype=np.int64,
    )
    sampled_indices = rng.integers(0, len(counts), size=(sample_count, len(counts)))
    pooled = values[sampled_indices].sum(axis=1)
    denominator = 2 * pooled[:, 0] + pooled[:, 1] + pooled[:, 2]
    f1_values = np.divide(
        2 * pooled[:, 0],
        denominator,
        out=np.zeros(sample_count, dtype=np.float64),
        where=denominator > 0,
    )
    lower, upper = np.percentile(f1_values, (2.5, 97.5), method='linear')
    return float(lower), float(upper)


def build_report(input_path: Path, *, sample_count: int, seed: int) -> BootstrapReport:
    counts_by_tracker = load_counts(input_path)
    rng = np.random.Generator(np.random.PCG64(seed))
    intervals: list[ConfidenceInterval] = []
    for tracker, counts in counts_by_tracker.items():
        lower, upper = bootstrap_micro_f1(counts, sample_count=sample_count, rng=rng)
        intervals.append(
            ConfidenceInterval(
                tracker=tracker,
                lower=lower,
                upper=upper,
            )
        )
    return BootstrapReport(
        resampling_unit='complete video',
        statistic='micro pair F1 after pooling TP, FP, and FN over sampled videos',
        sample_count=sample_count,
        rng='NumPy PCG64',
        seed=seed,
        interval='2.5th and 97.5th percentiles',
        confidence_intervals=tuple(intervals),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bootstrap video-level micro pair-F1 confidence intervals.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = build_report(arguments.input, sample_count=arguments.samples, seed=arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')
    print(report.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
