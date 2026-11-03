"""Inference: image -> preprocessing -> ANN -> softmax probabilities."""
from __future__ import annotations

import numpy as np

from .model_io import load_model
from .neural_network import NeuralNetwork
from .preprocessing import preprocess_image


class DigitPredictor:
    def __init__(self, model: NeuralNetwork, meta: dict | None = None):
        self.model, self.meta = model, meta or {}

    @classmethod
    def from_saved(cls, path=None) -> "DigitPredictor":
        model, meta = load_model(path)
        return cls(model, meta)

    def predict_vector(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.model.layer_sizes[0]:
            raise ValueError(f"Expected 784 features, got {x.shape[1]}")
        p = self.model.predict_proba(x)[0]
        digit = int(np.argmax(p))
        return {"digit": digit, "confidence": float(p[digit]), "probabilities": p.astype(float).tolist()}

    def predict_image(self, img) -> dict:
        vec, img28 = preprocess_image(img)
        result = self.predict_vector(vec)
        result["processed"] = img28
        return result
