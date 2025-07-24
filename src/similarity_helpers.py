import numpy as np
from typing import Callable

from common_types import Detection, Track


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def histogram_similarity(det1: Detection, det2: Detection) -> float:
    """
    Compute similarity between two detections based on their color histograms.

    Args:
        det1: First detection
        det2: Second detection

    Returns:
        Cosine similarity between the color histograms (0-1 range)
    """
    return cosine_similarity(det1.color_histogram, det2.color_histogram)


def _calc_pairwise(a: Track, b: Track, metric: Callable[[Detection, Detection], float]) -> float:
    """Calculate pairwise metric between all detections in two tracks."""
    total = 0
    count = 0
    for da in a.sorted_detections:
        for db in b.sorted_detections:
            total += metric(da, db)
            count += 1
    return total / count if count > 0 else 0.0


def mean_embedding(t: Track) -> np.ndarray:
    """Calculate the mean embedding of a track."""
    return np.mean([d.embedding for d in t.sorted_detections], axis=0)


def pairwise_cosine_similarity(a: Track, b: Track) -> float:
    """Calculate pairwise cosine similarity between all detections in two tracks."""
    return _calc_pairwise(a, b, lambda a, b: cosine_similarity(a.embedding, b.embedding))


def mean_embedding_cosine_similarity(a: Track, b: Track) -> float:
    """Calculate cosine similarity between mean embeddings of two tracks."""
    return cosine_similarity(
        mean_embedding(a),
        mean_embedding(b),
    )


def pairwise_histogram_similarity(a: Track, b: Track) -> float:
    """Calculate pairwise histogram similarity between all detections in two tracks."""
    return _calc_pairwise(a, b, lambda a, b: histogram_similarity(a, b))


def _mean_embedding_histogram(t: Track) -> np.ndarray:
    """Calculate the mean embedding of a track."""
    return np.mean([d.color_histogram for d in t.sorted_detections], axis=0)


def mean_embedding_histogram_similarity(a: Track, b: Track) -> float:
    """Calculate histogram similarity between mean embeddings of two tracks."""
    return cosine_similarity(
        _mean_embedding_histogram(a),
        _mean_embedding_histogram(b),
    )


def pairwise_squared_cosine_similarity(a: Track, b: Track) -> float:
    """Calculate pairwise squared cosine similarity between all detections in two tracks."""
    return _calc_pairwise(a, b, lambda a, b: cosine_similarity(a.embedding, b.embedding) ** 2)


def prop_embeddings_sim(a: Track, b: Track, min_sim: float = 0.5) -> float:
    """Count how many embeddings in two tracks have cosine similarity above a threshold."""
    count = 0
    cnt = 0
    for da in a.sorted_detections:
        for db in b.sorted_detections:
            cnt += 1
            if cosine_similarity(da.embedding, db.embedding) >= min_sim:
                count += 1
    return count / cnt if cnt > 0 else 0.0


def print_track_similarity_statistics(tracks: list[Track], greedy_min_cosine_similarity: float):
    # for the tracks, I want to experiment:
    # I want to compute the average embedding vector of each track
    # I want to compute the average cosine similarity between the embedding vectors of each detection of a track with the average embedding vector of the track
    # I want to compute the average cosine similarity between the embedding vectors of each detection of other tracks with the average embedding vector of the track
    # I want to compute the pairwise cosine similarity between the average embedding vectors of each of the tracks

    average_embeddings = {}
    for track in tracks:
        average_embedding = np.mean([d.embedding for d in track.sorted_detections], axis=0)
        average_embeddings[track.track_id] = average_embedding

        average_cosine_similarity = 0
        for detection in track.sorted_detections:
            average_cosine_similarity += cosine_similarity(detection.embedding, average_embedding)
        average_cosine_similarity /= len(track.sorted_detections)
        print(
            f'Track {track.track_id} average cosine similarity with its own detections: {average_cosine_similarity} ({len(track.sorted_detections)} detections)'
        )

        for other_track in tracks:
            if other_track.track_id == track.track_id:
                continue
            if (
                other_track.start_frame() < track.end_frame()  # It does not start after the track
                or other_track.start_frame() > track.end_frame() + 30  # It starts too far in the future
            ):
                continue
            average_cosine_similarity = 0
            for detection in other_track.sorted_detections:
                average_cosine_similarity += cosine_similarity(detection.embedding, average_embedding)
            average_cosine_similarity /= len(other_track.sorted_detections)
            print(
                f'Track {track.track_id} average cosine similarity with track {other_track.track_id}: {average_cosine_similarity} ({len(other_track.sorted_detections)} detections)'
            )

    for i in range(len(tracks)):
        for j in range(len(tracks)):
            track_i = tracks[i]
            track_j = tracks[j]
            if (
                track_i.start_frame() < track_j.end_frame()  # It does not start after the track
                or track_i.start_frame() > track_j.end_frame() + 30  # It starts too far in the future
                or i == j
            ):
                continue
            average_cosine_similarity = cosine_similarity(
                average_embeddings[track_i.track_id], average_embeddings[track_j.track_id]
            )
            print(
                f'Pairwise cosine similarity between track {track_j.track_id} and track {track_i.track_id}: {average_cosine_similarity} ({len(track_i.sorted_detections)} vs {len(track_j.sorted_detections)} detections)'
            )

            average_pairwise_cosine_similarity = 0
            number_of_detections_with_good_similarity = 0
            for detection_i in track_i.sorted_detections:
                for detection_j in track_j.sorted_detections:
                    sim = cosine_similarity(detection_i.embedding, detection_j.embedding)
                    if sim >= greedy_min_cosine_similarity:
                        number_of_detections_with_good_similarity += 1
                    average_pairwise_cosine_similarity += sim
            average_pairwise_cosine_similarity /= len(track_i.sorted_detections) * len(track_j.sorted_detections)
            print(
                f'Average pairwise cosine similarity between track {track_j.track_id} and track {track_i.track_id}: {average_pairwise_cosine_similarity}'
            )
            print(
                f'Number of detections with good similarity: {number_of_detections_with_good_similarity}/{len(track_i.sorted_detections) * len(track_j.sorted_detections)} ({number_of_detections_with_good_similarity / (len(track_i.sorted_detections) * len(track_j.sorted_detections)) * 100:.2f}%)'
            )
