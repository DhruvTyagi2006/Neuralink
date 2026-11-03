import numpy as np
import pytest
from src.neural_network import NeuralNetwork


def test_architecture_and_parameter_count():
    net = NeuralNetwork()
    assert net.layer_sizes == (784, 128, 64, 10)
    assert net.params["W1"].shape == (784, 128) and net.params["W2"].shape == (128, 64) and net.params["W3"].shape == (64, 10)
    assert net.count_params() == 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10 == 109386
    assert all(np.all(net.params[f"b{l}"] == 0) for l in (1, 2, 3))     # biases start at zero


def test_he_initialisation_scale():
    net = NeuralNetwork(seed=1)
    assert abs(net.params["W1"].std() - np.sqrt(2 / 784)) < 0.002


def test_forward_shapes_and_probabilities():
    net = NeuralNetwork()
    X = np.random.default_rng(0).random((16, 784)).astype(np.float32)
    P, cache = net.forward(X)
    assert P.shape == (16, 10)
    assert cache["A1"].shape == (16, 128) and cache["A2"].shape == (16, 64)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5) and np.all(np.isfinite(P))


def test_wrong_input_size_is_rejected():
    with pytest.raises(ValueError):
        NeuralNetwork().forward(np.zeros((2, 100)))


def test_batch_prediction_dimensions():
    net = NeuralNetwork()
    X = np.random.default_rng(0).random((25, 784)).astype(np.float32)
    assert net.predict_proba(X, batch_size=10).shape == (25, 10)
    assert net.predict(X).shape == (25,)


def test_images_become_784_features():
    from src.data_loader import preprocess_images
    out = preprocess_images(np.full((3, 28, 28), 255, dtype=np.uint8))
    assert out.shape == (3, 784) and out.max() == 1.0 and out.min() == 1.0
