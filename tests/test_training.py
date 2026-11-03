import numpy as np
from src.train import TrainConfig, saved_model_matches, train


def _toy_data(n=600, seed=0):
    """Ten separable clusters in 784-d so a few epochs are enough to learn."""
    rng = np.random.default_rng(seed)
    centers = rng.random((10, 784)).astype(np.float32)
    def make(k):
        y = rng.integers(0, 10, k)
        return np.clip(centers[y] + rng.normal(0, 0.15, (k, 784)), 0, 1).astype(np.float32), y
    (xt, yt), (xv, yv), (xs, ys) = make(n), make(150), make(150)
    return {"x_train": xt, "y_train": yt, "x_val": xv, "y_val": yv, "x_test": xs, "y_test": ys}


def test_training_reduces_loss_and_learns(tmp_dirs):
    cfg = TrainConfig(hidden_sizes=(32,), epochs=6, batch_size=32, patience=6)
    model, hist, meta = train(cfg, _toy_data(), save=True)
    assert hist["train_loss"][-1] < 0.5 * hist["train_loss"][0]
    assert hist["val_acc"][-1] > 0.9
    assert meta["n_parameters"] == 784 * 32 + 32 + 32 * 10 + 10
    assert saved_model_matches(cfg) and not saved_model_matches(TrainConfig(epochs=3))


def test_training_is_reproducible_with_seed(tmp_dirs):
    cfg = TrainConfig(hidden_sizes=(16,), epochs=2, batch_size=64)
    a = train(cfg, _toy_data(), save=False, verbose=False)[1]
    b = train(cfg, _toy_data(), save=False, verbose=False)[1]
    assert a["train_loss"] == b["train_loss"]


def test_early_stopping_and_best_checkpoint(tmp_dirs):
    cfg = TrainConfig(hidden_sizes=(16,), epochs=30, learning_rate=0.5, patience=2)
    _, hist, meta = train(cfg, _toy_data(), save=False, verbose=False)
    assert meta["epochs_run"] <= 30 and meta["best_epoch"] >= 1
    assert min(hist["val_loss"]) == hist["val_loss"][meta["best_epoch"] - 1]
