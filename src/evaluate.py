"""Model evaluation: loss, accuracy, confusion matrix, precision / recall / F1.

All numbers are computed from real predictions - nothing is hard-coded.
Run:  python -m src.evaluate
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from . import config as C
from .data_loader import load_mnist
from .loss_functions import cross_entropy, one_hot
from .model_io import load_model


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Rows = actual class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def classification_metrics(cm: np.ndarray) -> dict:
    """Per-class precision, recall, F1 plus macro / weighted averages, derived from the confusion matrix.

        precision_i = TP_i / (column sum)     recall_i = TP_i / (row sum)
        F1_i = 2 * precision * recall / (precision + recall)
    """
    tp = np.diag(cm).astype(float)
    precision = _safe_div(tp, cm.sum(axis=0).astype(float))
    recall = _safe_div(tp, cm.sum(axis=1).astype(float))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    support = cm.sum(axis=1)
    w = support / support.sum()
    return {
        "per_class": {str(i): {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])} for i in range(len(tp))},
        "macro": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())},
        "weighted": {"precision": float((precision * w).sum()), "recall": float((recall * w).sum()),
                     "f1": float((f1 * w).sum())},
    }


def evaluate_split(model, X, y, n_gallery: int = 60) -> dict:
    P = model.predict_proba(X)
    pred = np.argmax(P, axis=1)
    cm = confusion_matrix(y, pred)
    wrong = np.where(pred != y)[0]
    # show the most confidently wrong predictions first
    order = wrong[np.argsort(-P[wrong, pred[wrong]])][:n_gallery]
    off = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    return {
        "n": int(len(y)), "loss": cross_entropy(P, one_hot(y)), "accuracy": float((pred == y).mean()),
        "correct": int((pred == y).sum()), "incorrect": int((pred != y).sum()),
        "confusion": cm.tolist(), **classification_metrics(cm),
        "top_confusions": [{"true": i, "pred": j, "count": c} for c, i, j in off[:10]],
        "misclassified": [{"index": int(i), "true": int(y[i]), "pred": int(pred[i]),
                           "prob": float(P[i, pred[i]])} for i in order],
    }


def run_evaluation(model=None, data=None, meta=None, save_plots: bool = True) -> dict:
    if model is None:
        model, meta = load_model()
    meta = meta or {}
    cfg = meta.get("config", {})
    data = data or load_mnist(cfg.get("seed", C.DEFAULT_SEED), cfg.get("val_size", C.DEFAULT_VAL_SIZE))
    result = {
        "test": evaluate_split(model, data["x_test"], data["y_test"]),
        "validation": {k: v for k, v in evaluate_split(model, data["x_val"], data["y_val"]).items()
                       if k in ("n", "loss", "accuracy")},
        "train": {k: v for k, v in evaluate_split(model, data["x_train"], data["y_train"]).items()
                  if k in ("n", "loss", "accuracy")},
        "meta": meta, "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    C.outputs_dir().mkdir(parents=True, exist_ok=True)
    C.evaluation_path().write_text(json.dumps(result))
    if save_plots:
        from . import visualization as viz
        hist = C.history_path()
        if hist.exists():
            viz.save_training_curves(json.loads(hist.read_text())["history"], C.outputs_dir())
        viz.save_confusion_matrix(np.array(result["test"]["confusion"]), C.outputs_dir())
        viz.save_misclassified_gallery(result["test"]["misclassified"], data["test_images"], C.outputs_dir())
    t = result["test"]
    print(f"TEST  accuracy {t['accuracy']:.4f}  loss {t['loss']:.4f}  "
          f"macro-F1 {t['macro']['f1']:.4f}  ({t['correct']}/{t['n']} correct)")
    return result


if __name__ == "__main__":
    run_evaluation()
