"""TensorFlow/Keras MLP used ONLY for comparison with the from-scratch NumPy network.

Same architecture (784-128-64-10), same seed and the very same train/val/test partitions.
Run:  python -m src.keras_model --optimizer adam --epochs 20
(requires `pip install tensorflow`; the NumPy ANN does not depend on TensorFlow)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import config as C
from .data_loader import load_mnist


def build_keras_model(hidden=C.DEFAULT_HIDDEN, optimizer="adam", lr=0.001, l2=1e-4, seed=C.DEFAULT_SEED):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(l2 / 2) if l2 else None   # same penalty (l2/2)*sum(W^2) as the NumPy net
    layers = [tf.keras.Input(shape=(C.N_FEATURES,))]
    layers += [tf.keras.layers.Dense(h, activation="relu", kernel_initializer="he_normal",
                                     kernel_regularizer=reg) for h in hidden]
    layers += [tf.keras.layers.Dense(C.N_CLASSES, activation="softmax", kernel_regularizer=reg)]
    model = tf.keras.Sequential(layers)
    opt = (tf.keras.optimizers.Adam(lr) if optimizer == "adam"
           else tf.keras.optimizers.SGD(lr, momentum=0.9))
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run(optimizer="adam", epochs=20, batch_size=64, lr=None, seed=C.DEFAULT_SEED, save=True) -> dict:
    import tensorflow as tf
    lr = lr or (0.001 if optimizer == "adam" else 0.01)
    data = load_mnist(seed)
    model = build_keras_model(optimizer=optimizer, lr=lr, seed=seed)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(data["x_train"], data["y_train"], validation_data=(data["x_val"], data["y_val"]),
                     epochs=epochs, batch_size=batch_size, callbacks=[stop], verbose=2, shuffle=True)
    seconds = time.time() - t0
    # evaluate() includes the L2 penalty in "loss"; compute plain cross-entropy from probabilities instead
    from .loss_functions import cross_entropy, one_hot
    P = model.predict(data["x_test"], verbose=0)
    result = {
        "optimizer": optimizer, "learning_rate": lr, "batch_size": batch_size, "epochs_run": len(hist.history["loss"]),
        "test_accuracy": float((P.argmax(1) == data["y_test"]).mean()),
        "test_loss": cross_entropy(P, one_hot(data["y_test"])),
        "train_seconds": seconds, "trainable_parameters": int(model.count_params()),
        "tensorflow_version": tf.__version__,
    }
    if save:
        C.outputs_dir().mkdir(parents=True, exist_ok=True)
        existing = {}
        if C.keras_results_path().exists():
            existing = json.loads(C.keras_results_path().read_text())
        existing[optimizer] = result
        C.keras_results_path().write_text(json.dumps(existing, indent=2))
        C.models_dir().mkdir(parents=True, exist_ok=True)
        model.save(C.models_dir() / f"keras_mlp_{optimizer}.keras")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    run(a.optimizer, a.epochs, a.batch_size)
