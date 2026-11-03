import numpy as np
from src.activations import relu, relu_derivative, softmax


def test_relu_positive_negative_zero():
    z = np.array([-2.0, 0.0, 3.5])
    assert np.allclose(relu(z), [0.0, 0.0, 3.5])
    assert np.allclose(relu_derivative(z), [0.0, 0.0, 1.0])   # convention: derivative(0) = 0


def test_softmax_sums_to_one_and_is_finite():
    z = np.random.default_rng(0).normal(size=(5, 10)) * 10
    p = softmax(z)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_softmax_stable_for_huge_logits():
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1.0)
    assert p.argmax() == 1


def test_softmax_shift_invariance():
    z = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(z), softmax(z + 50))
