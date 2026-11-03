import numpy as np
from src.loss_functions import cross_entropy, one_hot
from src.neural_network import NeuralNetwork
from src.optimizers import SGD, Adam


def _tiny():
    rng = np.random.default_rng(0)
    net = NeuralNetwork((6, 5, 4, 3), seed=1, dtype=np.float64)
    X = rng.normal(size=(8, 6))
    Y = one_hot(rng.integers(0, 3, 8), 3, np.float64)
    return net, X, Y


def test_gradient_shapes_and_finite():
    net, X, Y = _tiny()
    P, cache = net.forward(X)
    grads = net.backward(cache, Y)
    for k, p in net.params.items():
        assert grads[k].shape == p.shape and np.all(np.isfinite(grads[k]))


def test_numerical_gradient_check():
    """Analytic gradients must match central finite differences."""
    net, X, Y = _tiny()
    grads = net.backward(net.forward(X)[1], Y)
    eps = 1e-6
    for k, p in net.params.items():
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            lp = cross_entropy(net.forward(X)[0], Y)
            p[i] = old - eps
            lm = cross_entropy(net.forward(X)[0], Y)
            p[i] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[k][i]) < 1e-6 + 1e-4 * abs(num), (k, i)


def test_l2_gradient_adds_weight_term():
    net, X, Y = _tiny()
    g0 = net.backward(net.forward(X)[1], Y)
    net.l2 = 0.1
    g1 = net.backward(net.forward(X)[1], Y)
    assert np.allclose(g1["W1"], g0["W1"] + 0.1 * net.params["W1"])
    assert np.allclose(g1["b1"], g0["b1"])


def test_loss_decreases_on_small_dataset():
    net, X, Y = _tiny()
    for opt in (SGD(0.1), SGD(0.05, momentum=0.9), Adam(0.01)):
        net, X, Y = _tiny()
        first = cross_entropy(net.forward(X)[0], Y)
        for _ in range(200):
            P, cache = net.forward(X)
            opt.step(net.params, net.backward(cache, Y))
        assert cross_entropy(net.forward(X)[0], Y) < 0.5 * first


def test_gradient_descent_update_rule():
    params = {"W": np.array([[1.0, 2.0]]), "b": np.array([0.5])}
    SGD(0.1).step(params, {"W": np.array([[1.0, -1.0]]), "b": np.array([2.0])})
    assert np.allclose(params["W"], [[0.9, 2.1]]) and np.allclose(params["b"], [0.3])


def test_cross_entropy_known_values_and_safe_log():
    Y = one_hot(np.array([0]), 3)
    assert np.isclose(cross_entropy(np.array([[0.9, 0.05, 0.05]]), Y), -np.log(0.9), atol=1e-6)
    assert np.isfinite(cross_entropy(np.array([[0.0, 0.5, 0.5]]), Y))   # log(0) is guarded
