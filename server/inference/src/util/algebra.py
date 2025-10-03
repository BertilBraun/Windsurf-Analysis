import numpy as np


def l2_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.maximum(1e-8, np.linalg.norm(a)))


def l1_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.maximum(1e-8, np.linalg.norm(a, ord=1)))
