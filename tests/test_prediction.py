import numpy as np
from PIL import Image, ImageDraw
from src.model_io import ModelNotFoundError, load_model, save_model
from src.neural_network import NeuralNetwork
from src.predict import DigitPredictor
import pytest


def test_prediction_is_valid_distribution():
    pred = DigitPredictor(NeuralNetwork(seed=3))
    r = pred.predict_vector(np.random.default_rng(0).random(784))
    assert isinstance(r["digit"], int) and 0 <= r["digit"] <= 9
    p = np.array(r["probabilities"])
    assert p.shape == (10,) and np.all(np.isfinite(p)) and np.isclose(p.sum(), 1.0, atol=1e-5)
    assert np.isclose(r["confidence"], p.max())


def test_wrong_vector_size_rejected():
    with pytest.raises(ValueError):
        DigitPredictor(NeuralNetwork()).predict_vector(np.zeros(10))


def test_save_and_reload_gives_identical_probabilities(tmp_path):
    net = NeuralNetwork(seed=5)
    x = np.random.default_rng(1).random((4, 784)).astype(np.float32)
    path = save_model(net, tmp_path / "m.npz", {"note": "test"})
    net2, meta = load_model(path)
    assert meta == {"note": "test"} and net2.layer_sizes == net.layer_sizes
    assert np.array_equal(net.predict_proba(x), net2.predict_proba(x))


def test_missing_model_raises_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        load_model(tmp_path / "nope.npz")


def test_corrupt_model_is_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, layer_sizes=np.array([784, 10]), W1=np.zeros((3, 3)), b1=np.zeros(10))
    with pytest.raises(ValueError):
        load_model(p)


def test_predict_image_end_to_end():
    img = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(img).line([(140, 40), (140, 240)], fill=255, width=20)
    r = DigitPredictor(NeuralNetwork(seed=3)).predict_image(img)
    assert r["processed"].shape == (28, 28) and 0 <= r["digit"] <= 9


def test_evaluation_metrics_match_sklearn():
    from sklearn.metrics import confusion_matrix as skcm, precision_recall_fscore_support
    from src.evaluate import classification_metrics, confusion_matrix
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 500)
    p = np.where(rng.random(500) < 0.8, y, rng.integers(0, 10, 500))
    cm = confusion_matrix(y, p)
    assert np.array_equal(cm, skcm(y, p, labels=range(10)))
    m = classification_metrics(cm)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(10), zero_division=0)
    assert np.allclose([m["per_class"][str(i)]["precision"] for i in range(10)], pr)
    assert np.allclose([m["per_class"][str(i)]["recall"] for i in range(10)], rc)
    assert np.allclose([m["per_class"][str(i)]["f1"] for i in range(10)], f1)
    assert np.isclose(m["macro"]["f1"], f1.mean())
