"""Hyper-parameter experiments (selected on the VALIDATION set only; the test set is not used).

Run:  python -m src.experiments            (about 2-3 minutes on a laptop CPU)
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

from . import config as C
from .data_loader import load_mnist
from .train import TrainConfig, train

EPOCHS = 8


def run_experiments(epochs: int = EPOCHS) -> list[dict]:
    data = load_mnist()
    base = TrainConfig(epochs=epochs, patience=epochs)
    plan = [("learning rate", f"lr={lr}", replace(base, learning_rate=lr)) for lr in (0.001, 0.01, 0.1)]
    plan += [("hidden sizes", f"hidden={h}", replace(base, hidden_sizes=h)) for h in ((32,), (128, 64), (256, 128))]
    plan += [("optimizer", name, replace(base, optimizer=name, learning_rate=lr))
             for name, lr in (("sgd", 0.1), ("momentum", 0.01), ("adam", 0.001))]
    plan += [("batch size", f"batch={b}", replace(base, batch_size=b)) for b in (16, 64, 256)]
    plan += [("L2 weight decay", f"l2={l}", replace(base, l2=l)) for l in (0.0, 1e-4, 1e-3)]
    rows = []
    for group, label, cfg in plan:
        t0 = time.time()
        _, hist, meta = train(cfg, data, save=False, verbose=False)
        rows.append({"group": group, "setting": label, "val_accuracy": meta["best_val_acc"],
                     "val_loss": meta["best_val_loss"], "final_train_accuracy": hist["train_acc"][-1],
                     "params": meta["n_parameters"], "seconds": time.time() - t0, "config": cfg.to_dict()})
        print(f"{group:16s} {label:22s} val acc {meta['best_val_acc']:.4f}  train acc {hist['train_acc'][-1]:.4f}",
              flush=True)
    C.outputs_dir().mkdir(parents=True, exist_ok=True)
    C.experiments_path().write_text(json.dumps({"epochs": epochs, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run_experiments()
