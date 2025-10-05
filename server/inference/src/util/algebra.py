import numpy as np
from scipy.stats import chi2


def l2_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.maximum(1e-8, np.linalg.norm(a)))


def l1_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.maximum(1e-8, np.linalg.norm(a, ord=1)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def chi2_dist(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    num = (a - b) ** 2
    den = a + b + eps
    return 0.5 * float((num / den).sum())


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def clip_probability(p: float) -> float:
    return max(1e-6, min(1 - 1e-6, p))


def platt_prob_from_dist(d: float, a: float, b: float) -> float:
    return clip_probability(sigmoid(a * (-d) + b))


def probability_from_dist(d: float, df: int = 2) -> float:
    """Calculate the probability for a distance to say, that the two tracks are the same. `df` is the degrees of freedom."""
    return 1 - float(chi2.cdf(d, df=df))
