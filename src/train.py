"""Mini-batch training loop for the NumPy ANN.

Run:  python -m src.train --epochs 20 --lr 0.01 --batch-size 64

Techniques used (all implemented by hand):
  * He initialisation, mini-batch gradient descent with shuffling each epoch
  * Momentum (classical) - optional Adam
  * Exponential learning-rate decay:  lr_t = lr_0 * decay^(epoch-1)
  * L2 weight decay (regularisation) to reduce overfitting
  * Validation after every epoch, early stopping and best-model checkpointing
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from . import config as C
from .data_loader import load_mnist, split_summary
from .loss_functions import accuracy, cross_entropy, one_hot
from .model_io import save_model
from .neural_network import NeuralNetwork
from .optimizers import get_optimizer


@dataclass
class TrainConfig:
    hidden_sizes: tuple = C.DEFAULT_HIDDEN
    learning_rate: float = 0.01
    batch_size: int = 64
    epochs: int = 20
    optimizer: str = "momentum"      # "sgd" | "momentum" | "adam"
    momentum: float = 0.9
    lr_decay: float = 0.95           # multiplicative decay per epoch (1.0 = constant)
    l2: float = 1e-4
    seed: int = C.DEFAULT_SEED
    patience: int = 6                # early-stopping patience on validation loss
    val_size: int = C.DEFAULT_VAL_SIZE

    @property
    def layer_sizes(self) -> tuple:
        return (C.N_FEATURES, *[int(h) for h in self.hidden_sizes], C.N_CLASSES)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = [int(h) for h in self.hidden_sizes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["hidden_sizes"] = tuple(d.get("hidden_sizes", C.DEFAULT_HIDDEN))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ status file
def write_status(status: dict) -> None:
    """Atomically write the status JSON that the Streamlit dashboard polls."""
    p = C.status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status))
    os.replace(tmp, p)


def read_status() -> dict | None:
    p = C.status_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def is_training_running(ignore_pid: int | None = None) -> bool:
    """True if a status file says 'running' and that process is still alive."""
    s = read_status()
    if not s or s.get("state") != "running":
        return False
    pid = int(s.get("pid", -1))
    return pid != ignore_pid and _pid_alive(pid)


# ------------------------------------------------------------------------ train
def evaluate_split(model: NeuralNetwork, X, y) -> tuple[float, float]:
    P = model.predict_proba(X)
    return cross_entropy(P, one_hot(y)), accuracy(P, y)


def train(cfg: TrainConfig, data: dict | None = None, progress_cb=None, save: bool = True,
          model_path=None, verbose: bool = True):
    """Train and return (model, history, meta). The best epoch (lowest validation loss) is kept."""
    data = data or load_mnist(cfg.seed, cfg.val_size)
    X, y = data["x_train"], data["y_train"]
    Y = one_hot(y)
    n = len(X)
    rng = np.random.default_rng(cfg.seed)                 # controls the shuffling
    model = NeuralNetwork(cfg.layer_sizes, seed=cfg.seed, l2=cfg.l2)
    opt = get_optimizer(cfg.optimizer, cfg.learning_rate, cfg.momentum)

    keys = ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_seconds")
    history = {k: [] for k in keys}
    best = {"val_loss": np.inf, "epoch": 0, "params": model.copy_params(), "val_acc": 0.0}
    wait, t_start = 0, time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        lr = cfg.learning_rate * (cfg.lr_decay ** (epoch - 1))
        opt.lr = lr
        perm = rng.permutation(n)                         # shuffle every epoch
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            P, cache = model.forward(X[idx])              # 1. forward propagation
            grads = model.backward(cache, Y[idx])         # 2. backpropagation
            opt.step(model.params, grads)                 # 3. gradient-descent update
        tr_loss, tr_acc = evaluate_split(model, X, y)     # metrics with the weights frozen
        va_loss, va_acc = evaluate_split(model, data["x_val"], data["y_val"])
        for k, v in zip(keys, (tr_loss, va_loss, tr_acc, va_acc, lr, time.time() - t0)):
            history[k].append(float(v))

        if va_loss < best["val_loss"] - 1e-4:
            best.update(val_loss=va_loss, epoch=epoch, params=model.copy_params(), val_acc=va_acc)
            wait = 0
        else:
            wait += 1
        if verbose:
            print(f"epoch {epoch:2d}/{cfg.epochs}  lr={lr:.5f}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
                  f"| val loss {va_loss:.4f} acc {va_acc:.4f}  ({history['epoch_seconds'][-1]:.1f}s)"
                  + ("  *best*" if best["epoch"] == epoch else ""), flush=True)
        if progress_cb:
            progress_cb({"state": "running", "epoch": epoch, "epochs": cfg.epochs,
                         "history": history, "best_epoch": best["epoch"]})
        if wait >= cfg.patience:
            if verbose:
                print(f"Early stopping: no validation-loss improvement for {cfg.patience} epochs.")
            break

    model.set_params(best["params"])                      # restore the best checkpoint
    meta = {
        "config": cfg.to_dict(), "layer_sizes": list(cfg.layer_sizes),
        "n_parameters": model.count_params(), "best_epoch": best["epoch"],
        "epochs_run": len(history["train_loss"]), "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"], "train_seconds": time.time() - t_start,
        "split": split_summary(data), "numpy_version": np.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if save:
        save_model(model, model_path, meta)
        C.history_path().parent.mkdir(parents=True, exist_ok=True)
        C.history_path().write_text(json.dumps({"history": history, "meta": meta}))
    return model, history, meta


def saved_model_matches(cfg: TrainConfig) -> bool:
    """True if a saved model exists that was trained with exactly this configuration."""
    p = C.history_path()
    if not (C.numpy_model_path().exists() and p.exists()):
        return False
    try:
        saved = json.loads(p.read_text())["meta"]["config"]
    except (KeyError, json.JSONDecodeError, OSError):
        return False
    return saved == cfg.to_dict()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the NumPy ANN on MNIST")
    ap.add_argument("--hidden", type=int, nargs="+", default=list(C.DEFAULT_HIDDEN))
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--optimizer", default="momentum", choices=["sgd", "momentum", "adam"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr-decay", type=float, default=0.95)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-evaluate", action="store_true", help="skip the final test-set evaluation")
    args = ap.parse_args(argv)

    cfg = TrainConfig(tuple(args.hidden), args.lr, args.batch_size, args.epochs, args.optimizer,
                      args.momentum, args.lr_decay, args.l2, args.seed, args.patience)
    if is_training_running(ignore_pid=os.getpid()):
        raise SystemExit("A training run is already in progress - wait for it to finish.")
    base = {"pid": os.getpid(), "config": cfg.to_dict(), "started": time.time()}
    write_status({**base, "state": "running", "epoch": 0, "epochs": cfg.epochs,
                  "history": {}, "message": "Loading MNIST..."})
    try:
        data = load_mnist(cfg.seed, cfg.val_size)
        print("Split:", split_summary(data), flush=True)
        model, history, meta = train(cfg, data, progress_cb=lambda s: write_status({**base, **s}))
        write_status({**base, "state": "finished", "epoch": meta["epochs_run"], "epochs": cfg.epochs,
                      "history": history, "best_epoch": meta["best_epoch"], "finished": time.time(),
                      "message": "Training finished."})
        print(f"Best epoch {meta['best_epoch']}  val acc {meta['best_val_acc']:.4f}  "
              f"saved to {C.numpy_model_path()}")
        if not args.no_evaluate:
            from .evaluate import run_evaluation
            run_evaluation(model, data, meta)
    except Exception as exc:
        write_status({**base, "state": "failed", "message": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
