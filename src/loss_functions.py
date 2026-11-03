"""Loss and metric helpers."""
import numpy as np

EPS = 1e-12


def one_hot(y: np.ndarray, n_classes: int = 10, dtype=np.float32) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=dtype)
    out[np.arange(len(y)), y] = 1.0
    return out


def cross_entropy(P: np.ndarray, Y: np.ndarray) -> float:
    """Mean categorical cross-entropy:  L = -(1/m) sum_n sum_i y_i * log(p_i).

    Probabilities are clipped to [EPS, 1] so log(0) can never occur.
    """
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, EPS, 1.0)), axis=1)))


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(P, axis=1) == y))
