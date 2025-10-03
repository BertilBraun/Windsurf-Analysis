from __future__ import annotations
from typing import Protocol

import numpy as np

from .algebra import l2_normalize, l1_normalize


class Embedding(Protocol):
    def distance(self, other: Embedding) -> float: ...
    def interpolate(self, other: Embedding, alpha: float) -> Embedding: ...
    @staticmethod
    def mean(embeddings: list[Embedding]) -> Embedding: ...


class VectorEmbedding:
    def __init__(self, embedding: np.ndarray):
        self.__embedding = l2_normalize(embedding)

    @property
    def embedding(self) -> np.ndarray:
        return self.__embedding

    def distance(self, other: Embedding) -> float:
        if not isinstance(other, VectorEmbedding):
            raise ValueError(f'Expected VectorEmbedding, got {type(other)}')
        similarity = np.dot(self.embedding, other.embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        )
        return 1 - similarity

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

    def distance(self, other: Embedding, eps: float = 1e-8) -> float:
        if not isinstance(other, HistogramEmbedding):
            raise ValueError(f'Expected HistogramEmbedding, got {type(other)}')
        """Calculate the chi2 distance between two embeddings."""
        num = (self.histogram - other.histogram) ** 2
        den = self.histogram + other.histogram + eps
        return 0.5 * float((num / den).sum())

    def probability(
        self, other: HistogramEmbedding, a: float = 7.427828328625088, b: float = 4.088360175681194
    ) -> float:
        """Calculate the probability for a distance to say, that the two tracks are the same. `a` and `b` are parameters of the platt scaling. The returned probability is in the range [0, 1] (sigmoid(a * -d + b))"""
        z = a * (-self.distance(other)) + b
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 1e-6, 1 - 1e-6))

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
