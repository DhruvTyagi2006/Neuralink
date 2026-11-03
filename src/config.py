"""Project-wide paths and constants.

Paths can be redirected with environment variables (used by the unit tests so
they never touch the real trained model):
    ANN_DATA_DIR, ANN_MODELS_DIR, ANN_OUTPUTS_DIR
"""
import os
from pathlib import Path

PROJECT_NAME = "Neuralink"
PROJECT_SUBTITLE = "Handwritten digit recognition with an artificial neural network built from scratch"

ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIDE = 28
N_FEATURES = IMAGE_SIDE * IMAGE_SIDE      # 784 input features
N_CLASSES = 10                            # digits 0-9
DEFAULT_HIDDEN = (128, 64)                # architecture 784 -> 128 -> 64 -> 10
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 6000                   # 54,000 train / 6,000 validation / 10,000 test


def data_dir() -> Path:
    return Path(os.environ.get("ANN_DATA_DIR", ROOT / "data"))


def models_dir() -> Path:
    return Path(os.environ.get("ANN_MODELS_DIR", ROOT / "models"))


def outputs_dir() -> Path:
    return Path(os.environ.get("ANN_OUTPUTS_DIR", ROOT / "outputs"))


def numpy_model_path() -> Path:
    return models_dir() / "ann_numpy.npz"


def keras_model_path() -> Path:
    return models_dir() / "keras_mlp.keras"


def history_path() -> Path:
    return outputs_dir() / "history.json"


def status_path() -> Path:
    return outputs_dir() / "training_status.json"


def evaluation_path() -> Path:
    return outputs_dir() / "evaluation.json"


def keras_results_path() -> Path:
    return outputs_dir() / "keras_results.json"


def experiments_path() -> Path:
    return outputs_dir() / "experiments.json"
