"""Saving / loading the NumPy network as an .npz archive.

The archive contains the weight matrices, bias vectors, the layer sizes and a JSON
string with the training configuration.  It is loaded with allow_pickle=False, so a
tampered file cannot execute code (plain arrays only).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import numpy_model_path
from .neural_network import NeuralNetwork


class ModelNotFoundError(FileNotFoundError):
    pass


def save_model(model: NeuralNetwork, path=None, meta: dict | None = None) -> Path:
    path = Path(path) if path else numpy_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(model.params)
    arrays["layer_sizes"] = np.array(model.layer_sizes, dtype=np.int64)
    arrays["meta_json"] = np.array(json.dumps(meta or {}))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_model(path=None):
    """Return (model, meta). Raises ModelNotFoundError with setup instructions if missing."""
    path = Path(path) if path else numpy_model_path()
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model at {path}. Train one first:  python -m src.train")
    with np.load(path, allow_pickle=False) as data:
        sizes = tuple(int(s) for s in data["layer_sizes"])
        model = NeuralNetwork(sizes)
        params = {}
        for k in model.params:
            if k not in data:
                raise ValueError(f"Model file is missing array '{k}'")
            arr = data[k]
            if arr.shape != model.params[k].shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"Model file has an invalid array '{k}'")
            params[k] = arr
        model.set_params(params)
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    return model, meta
