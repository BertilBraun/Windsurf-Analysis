from __future__ import annotations
from typing import Protocol

import numpy as np

from .algebra import chi2_distance, clip_probability, cosine_similarity, l2_normalize, l1_normalize, hellinger_distance


class Embedding(Protocol):
    def distance(self, other: Embedding) -> float: ...
    def interpolate(self, other: Embedding, alpha: float) -> Embedding: ...
    @staticmethod
    def mean(embeddings: list[Embedding]) -> Embedding: ...
    def probability(self, other: Embedding, gamma: float = 1) -> float: ...


class VectorEmbedding:
    def __init__(self, embedding: np.ndarray):
        self.__embedding = l2_normalize(embedding)

    @property
    def embedding(self) -> np.ndarray:
        return self.__embedding

    def distance(self, other: Embedding) -> float:
        if not isinstance(other, VectorEmbedding):
            raise ValueError(f'Expected VectorEmbedding, got {type(other)}')
        return 1 - cosine_similarity(self.embedding, other.embedding)

    def probability(self, other: Embedding, gamma: float = 1) -> float:
        if not isinstance(other, VectorEmbedding):
            raise ValueError(f'Expected VectorEmbedding, got {type(other)}')
        return 1 - self.distance(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorEmbedding):
            return False
        return np.allclose(self.embedding, other.embedding)

    def __hash__(self) -> int:
        return hash(tuple(self.embedding))

    def interpolate(self, other: Embedding, alpha: float) -> VectorEmbedding:
        if not isinstance(other, VectorEmbedding):
            raise ValueError(f'Expected VectorEmbedding, got {type(other)}')
        return VectorEmbedding(self.embedding * (1 - alpha) + other.embedding * alpha)

    @staticmethod
    def mean(embeddings: list[Embedding]) -> VectorEmbedding:
        if not all(isinstance(e, VectorEmbedding) for e in embeddings):
            raise ValueError(f'Expected list of VectorEmbedding, got {type(embeddings)}')
        return VectorEmbedding(np.mean([e.embedding for e in embeddings], axis=0))  # type: ignore


class HistogramEmbedding:
    def __init__(self, histogram: np.ndarray):
        self.__histogram = l1_normalize(histogram)

    @property
    def histogram(self) -> np.ndarray:
        return self.__histogram

    def distance(self, other: Embedding) -> float:
        if not isinstance(other, HistogramEmbedding):
            raise ValueError(f'Expected HistogramEmbedding, got {type(other)}')
        """Calculate the chi2 distance between two embeddings."""
        return chi2_distance(self.histogram, other.histogram)

    def probability(self, other: Embedding, gamma: float = 4) -> float:
        if not isinstance(other, HistogramEmbedding):
            raise ValueError(f'Expected HistogramEmbedding, got {type(other)}')
        """Calculate the probability, that the two tracks are the same based on the appearance similarity."""
        distance = self.distance(other)
        # Distance values 0-0.05 are very good matches
        # Distance values 0.05-0.15 are still good matches
        # Distance values >0.15 (up to ~0.9 for the worst matching pairs in the sample dataset) are bad matches
        return clip_probability(1 - distance * gamma)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HistogramEmbedding):
            return False
        return np.allclose(self.histogram, other.histogram)

    def __hash__(self) -> int:
        return hash(tuple(self.histogram))

    def interpolate(self, other: Embedding, alpha: float) -> HistogramEmbedding:
        if not isinstance(other, HistogramEmbedding):
            raise ValueError(f'Expected HistogramEmbedding, got {type(other)}')
        return HistogramEmbedding(self.histogram * (1 - alpha) + other.histogram * alpha)

    @staticmethod
    def mean(embeddings: list[Embedding]) -> HistogramEmbedding:
        if not all(isinstance(e, HistogramEmbedding) for e in embeddings):
            raise ValueError(f'Expected list of HistogramEmbedding, got {type(embeddings)}')
        return HistogramEmbedding(np.mean([e.histogram for e in embeddings], axis=0))  # type: ignore


class HellingerEmbedding:
    """
    Stores L2-normalized vector AFTER per-block L1->sqrt mapping.
    Distance = 1 - cosine (∈[0,1] for non-negative vectors).
    """

    def __init__(self, vec: np.ndarray):
        self.__embedding = l2_normalize(vec.astype(np.float32))

    @property
    def embedding(self) -> np.ndarray:
        return self.__embedding

    def distance(self, other: Embedding) -> float:
        if not isinstance(other, HellingerEmbedding):
            raise ValueError(f'Expected HellingerEmbedding, got {type(other)}')
        return hellinger_distance(self.embedding, other.embedding)

    def probability(self, other: Embedding, gamma: float = 10) -> float:
        if not isinstance(other, HellingerEmbedding):
            raise ValueError(f'Expected HellingerEmbedding, got {type(other)}')
        distance = self.distance(other)
        # linear clip; tune to your data (e.g., use Platt if you have labels)
        # d≈0.0→~1, d≈0.1→~0
        return clip_probability(1.0 - distance * gamma)

    def interpolate(self, other: Embedding, alpha: float) -> HellingerEmbedding:
        if not isinstance(other, HellingerEmbedding):
            raise ValueError(f'Expected HellingerEmbedding, got {type(other)}')
        return HellingerEmbedding((1 - alpha) * self.embedding + alpha * other.embedding)

    @staticmethod
    def mean(embeddings: list[Embedding]) -> HellingerEmbedding:
        if not all(isinstance(e, HellingerEmbedding) for e in embeddings):
            raise ValueError(f'Expected list of HellingerEmbedding, got {type(embeddings)}')
        V = np.stack([e.embedding for e in embeddings], axis=0)  # type: ignore
        return HellingerEmbedding(V.mean(axis=0))
