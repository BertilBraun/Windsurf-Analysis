import numpy as np
from scipy.stats import chi2

from server.inference.src.settings import EPS


def floor(x: float) -> int:
    return int(np.floor(x))


def ceil(x: float) -> int:
    return int(np.ceil(x))


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def l2_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.maximum(1e-8, np.linalg.norm(a)))


def l1_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.maximum(1e-8, np.linalg.norm(a, ord=1)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def hellinger_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    After per-block L1->sqrt and final L2, cosine ~ Bhattacharyya.
    We use 1 - cosine as a bounded [0,1] distance.
    """
    return float(1.0 - float(np.dot(u, v)))


def chi2_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
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


def clamp_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))


def NLL_from_prob(p: float) -> float:
    """Negative log-likelihood ratio cost: -logit(p)."""
    p = clamp_prob(p)
    return float(-np.log(p / (1.0 - p)))
