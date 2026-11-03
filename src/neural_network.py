"""Fully connected feedforward ANN implemented from scratch with NumPy.

Default architecture:  784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)

Convention: data are ROWS.  For a batch X with shape (m, n_in) a layer computes
        Z = A_prev @ W + b          W: (n_in, n_out)   b: (n_out,)   Z: (m, n_out)
which is the batch version of the textbook  z = W^T x + b.
"""
from __future__ import annotations

import numpy as np

from .activations import relu, relu_derivative, softmax
from .loss_functions import cross_entropy


class NeuralNetwork:
    def __init__(self, layer_sizes=(784, 128, 64, 10), seed: int = 42, l2: float = 0.0, dtype=np.float32):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and an output layer")
        self.layer_sizes = tuple(int(s) for s in layer_sizes)
        self.n_layers = len(self.layer_sizes) - 1       # number of weight layers
        self.l2, self.dtype, self.seed = l2, dtype, seed
        self.params = self._initialize(seed)

    # ------------------------------------------------------------------ init
    def _initialize(self, seed: int) -> dict:
        """He initialisation for ReLU layers, Glorot for the softmax layer, zero biases.

        He:     W ~ N(0, 2 / fan_in)            keeps the variance of ReLU activations stable
        Glorot: W ~ N(0, 2 / (fan_in+fan_out))  small, balanced weights for the output layer
        """
        rng = np.random.default_rng(seed)
        params = {}
        for l in range(1, self.n_layers + 1):
            fan_in, fan_out = self.layer_sizes[l - 1], self.layer_sizes[l]
            if l < self.n_layers:
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))
            params[f"W{l}"] = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(self.dtype)
            params[f"b{l}"] = np.zeros(fan_out, dtype=self.dtype)
        return params

    # --------------------------------------------------------------- forward
    def forward(self, X: np.ndarray):
        """Forward propagation. Returns (P, cache); cache holds A0..AL and Z1..ZL for backprop."""
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected input of shape (m, {self.layer_sizes[0]}), got {X.shape}")
        cache = {"A0": X}
        A = X
        for l in range(1, self.n_layers + 1):
            Z = A @ self.params[f"W{l}"] + self.params[f"b{l}"]
            A = relu(Z) if l < self.n_layers else softmax(Z)
            cache[f"Z{l}"], cache[f"A{l}"] = Z, A
        return A, cache

    # -------------------------------------------------------------- backward
    def backward(self, cache: dict, Y: np.ndarray) -> dict:
        """Backpropagation (chain rule), returning gradients for every parameter.

        Output layer (softmax + cross-entropy simplifies beautifully):
            dZ_L = (P - Y) / m
        For each layer l = L ... 1:
            dW_l = A_{l-1}^T @ dZ_l   (+ l2 * W_l  for weight decay)
            db_l = sum_rows(dZ_l)
            dA_{l-1} = dZ_l @ W_l^T
            dZ_{l-1} = dA_{l-1} * ReLU'(Z_{l-1})
        """
        P = cache[f"A{self.n_layers}"]
        m = P.shape[0]
        dZ = (P - Y) / m
        grads = {}
        for l in range(self.n_layers, 0, -1):
            W = self.params[f"W{l}"]
            grads[f"W{l}"] = cache[f"A{l-1}"].T @ dZ + self.l2 * W
            grads[f"b{l}"] = dZ.sum(axis=0)
            if l > 1:
                dA_prev = dZ @ W.T
                dZ = dA_prev * relu_derivative(cache[f"Z{l-1}"])
        return grads

    # ------------------------------------------------------------- inference
    def predict_proba(self, X: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim == 1:
            X = X[None, :]
        out = [self.forward(X[i:i + batch_size])[0] for i in range(0, len(X), batch_size)]
        return np.concatenate(out, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        return cross_entropy(self.predict_proba(X), Y)

    # ----------------------------------------------------------- bookkeeping
    def layer_param_counts(self) -> list[dict]:
        rows = []
        for l in range(1, self.n_layers + 1):
            n_in, n_out = self.layer_sizes[l - 1], self.layer_sizes[l]
            rows.append({"layer": l, "shape": f"{n_in} -> {n_out}", "weights": n_in * n_out,
                         "biases": n_out, "total": n_in * n_out + n_out,
                         "activation": "ReLU" if l < self.n_layers else "Softmax"})
        return rows

    def count_params(self) -> int:
        return sum(r["total"] for r in self.layer_param_counts())

    def copy_params(self) -> dict:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: dict) -> None:
        for k in self.params:
            if params[k].shape != self.params[k].shape:
                raise ValueError(f"Shape mismatch for {k}: {params[k].shape} vs {self.params[k].shape}")
            self.params[k] = np.array(params[k], dtype=self.dtype)
