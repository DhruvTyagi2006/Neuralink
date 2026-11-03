"""Activation functions and their derivatives (pure NumPy)."""
import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU(z) = max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """dReLU/dz = 1 if z > 0 else 0.  Convention: the derivative at z = 0 is 0."""
    return (z > 0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax: P(i) = exp(z_i) / sum_j exp(z_j).

    Subtracting the row maximum first does not change the result
    (softmax is shift-invariant) but prevents exp() from overflowing.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)
