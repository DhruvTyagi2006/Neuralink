"""MNIST loading, normalisation and the reproducible train/validation/test split.

MNIST = 70,000 greyscale 28x28 images of handwritten digits:
    60,000 official training images  -> split into 54,000 train + 6,000 validation
    10,000 official test images      -> touched only for the final evaluation
"""
from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .config import DEFAULT_SEED, DEFAULT_VAL_SIZE, N_FEATURES, data_dir

# Original MNIST files (MD5 values are those published for the official release).
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRRORS = [
    "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]


class DatasetDownloadError(RuntimeError):
    pass


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(name: str, folder: Path) -> Path:
    dest = folder / name
    if dest.exists() and _md5(dest) == FILES[name]:
        return dest
    folder.mkdir(parents=True, exist_ok=True)
    errors = []
    for base in MIRRORS:
        try:
            req = urllib.request.Request(base + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if _md5(dest) == FILES[name]:
                return dest
            errors.append(f"{base}: checksum mismatch")
        except Exception as exc:  # network errors differ a lot, report them all
            errors.append(f"{base}: {exc}")
    raise DatasetDownloadError(
        f"Could not download {name}.\n" + "\n".join(errors) +
        f"\n\nManual fix: download the four MNIST .gz files (or Keras' mnist.npz) and "
        f"place them in {folder}."
    )


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name}: bad image magic number {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name}: bad label magic number {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n)


def load_mnist_raw(folder: Path | None = None):
    """Return (x_train, y_train, x_test, y_test); images are uint8 (N, 28, 28)."""
    folder = Path(folder) if folder else data_dir()
    npz = folder / "mnist.npz"  # offline alternative (Keras format)
    if npz.exists():
        with np.load(npz, allow_pickle=False) as d:
            return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    paths = {n: _ensure_file(n, folder) for n in FILES}
    return (
        _read_idx_images(paths["train-images-idx3-ubyte.gz"]),
        _read_idx_labels(paths["train-labels-idx1-ubyte.gz"]),
        _read_idx_images(paths["t10k-images-idx3-ubyte.gz"]),
        _read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"]),
    )


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """(N, 28, 28) uint8  ->  (N, 784) float32 in [0, 1]:  x_norm = x / 255."""
    return (images.reshape(len(images), -1) / 255.0).astype(np.float32)


def load_mnist(seed: int = DEFAULT_SEED, val_size: int = DEFAULT_VAL_SIZE, folder=None) -> dict:
    """Load MNIST and build the reproducible split.

    The validation set is drawn from the 60,000 official training images with a
    seeded permutation, so every run (NumPy and Keras) sees identical partitions.
    """
    x_tr_img, y_tr, x_te_img, y_te = load_mnist_raw(folder)
    if x_tr_img.shape != (60000, 28, 28) or x_te_img.shape != (10000, 28, 28):
        raise ValueError(f"Unexpected MNIST shapes: {x_tr_img.shape}, {x_te_img.shape}")
    perm = np.random.default_rng(seed).permutation(len(x_tr_img))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    data = {
        "x_train": preprocess_images(x_tr_img[train_idx]), "y_train": y_tr[train_idx].astype(np.int64),
        "x_val": preprocess_images(x_tr_img[val_idx]), "y_val": y_tr[val_idx].astype(np.int64),
        "x_test": preprocess_images(x_te_img), "y_test": y_te.astype(np.int64),
        "test_images": x_te_img,  # raw uint8 images, used for galleries
    }
    assert data["x_train"].shape[1] == N_FEATURES
    return data


def split_summary(data: dict) -> dict:
    return {k: int(len(data[f"y_{k}"])) for k in ("train", "val", "test")}
