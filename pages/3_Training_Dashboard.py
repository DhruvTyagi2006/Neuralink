"""Training Dashboard: configure, start and monitor training (runs in a background process)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st

from src import config as C
from src.train import TrainConfig, is_training_running, read_status, saved_model_matches, write_status
from src.ui import page_header, read_json, setup_page, tile
from src.theme import PALETTE as P
from src.visualization import curves_figure

setup_page("Training Dashboard", "📈")
page_header("Training Dashboard", "Set the hyper-parameters, start training and watch the curves update live.")

st.caption("Training runs in a separate process, so refreshing this page never restarts it, and an existing model "
           "trained with the same settings is reused unless you ask to retrain.")

with st.container(border=True):
    with st.form("train_form"):
        a, b, c = st.columns(3)
        epochs = a.number_input("Epochs", 1, 200, 25)
        lr = b.number_input("Learning rate", 0.0001, 1.0, 0.01, step=0.001, format="%.4f")
        batch = c.selectbox("Batch size", [16, 32, 64, 128, 256], index=2)
        d, e, f = st.columns(3)
        opt = d.selectbox("Optimizer", ["momentum", "sgd", "adam"], help="momentum = SGD with momentum")
        mom = e.slider("Momentum", 0.0, 0.99, 0.9, disabled=False)
        decay = f.slider("Learning-rate decay per epoch", 0.80, 1.0, 0.95, 0.01)
        g, h, i = st.columns(3)
        h1 = g.number_input("Hidden layer 1 neurons", 8, 1024, 128, step=8)
        h2 = h.number_input("Hidden layer 2 neurons", 0, 1024, 64, step=8, help="0 = use a single hidden layer")
        l2 = i.number_input("L2 weight decay", 0.0, 0.1, 0.0001, step=0.0001, format="%.4f")
        j, k = st.columns(2)
        seed = j.number_input("Random seed", 0, 99999, C.DEFAULT_SEED)
        patience = k.number_input("Early-stopping patience (epochs)", 1, 50, 6)
        retrain = st.checkbox("Retrain even if a model with identical settings already exists")
        go_btn = st.form_submit_button("Start training", type="primary")

if go_btn:
    cfg = TrainConfig(tuple(int(x) for x in (h1, h2) if x > 0), float(lr), int(batch), int(epochs), opt, float(mom),
                      float(decay), float(l2), int(seed), int(patience))
    if is_training_running():
        st.warning("A training run is already in progress. Wait for it to finish.", icon="⏳")
    elif saved_model_matches(cfg) and not retrain:
        st.success("A model trained with exactly these settings already exists - nothing to do. "
                   "Tick *retrain* to train again.", icon="✅")
    else:
        C.outputs_dir().mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "src.train", "--epochs", str(cfg.epochs), "--lr", str(cfg.learning_rate),
               "--batch-size", str(cfg.batch_size), "--optimizer", cfg.optimizer, "--momentum", str(cfg.momentum),
               "--lr-decay", str(cfg.lr_decay), "--l2", str(cfg.l2), "--seed", str(cfg.seed),
               "--patience", str(cfg.patience), "--hidden", *[str(x) for x in cfg.hidden_sizes]]
        log = open(C.outputs_dir() / "train_log.txt", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_status({"pid": proc.pid, "state": "running", "epoch": 0, "epochs": cfg.epochs, "history": {},
                      "config": cfg.to_dict(), "started": time.time(), "message": "Starting..."})
        st.session_state["_was_running"] = True
        st.toast("Training started")


def live_view():
    status, running = read_status(), is_training_running()
    if st.session_state.get("_was_running") and not running:
        st.session_state["_was_running"] = False
        st.rerun()
    st.session_state["_was_running"] = running
    saved = read_json(C.history_path())
    hist = (status or {}).get("history") or (saved or {}).get("history") or {}
    state = (status or {}).get("state", "idle") if running or (status or {}).get("state") != "running" else "stopped"
    labels = {"running": "Training in progress", "finished": "Finished", "failed": "Failed", "idle": "Idle",
              "stopped": "Stopped unexpectedly"}
    st.markdown(f"#### Status: <span class='pill'>{labels.get(state, state)}</span>", unsafe_allow_html=True)
    if state == "failed":
        st.error(status.get("message", "Training failed."))
    if running or (status and status.get("epochs")):
        ep, total = status.get("epoch", 0), max(1, status.get("epochs", 1))
        st.progress(min(1.0, ep / total), text=f"Epoch {ep} of {total}")
    if hist and hist.get("train_loss"):
        n = len(hist["train_loss"])
        m = st.columns(5)
        m[0].markdown(tile("Epoch", str(n)), unsafe_allow_html=True)
        m[1].markdown(tile("Training loss", f"{hist['train_loss'][-1]:.4f}"), unsafe_allow_html=True)
        m[2].markdown(tile("Validation loss", f"{hist['val_loss'][-1]:.4f}"), unsafe_allow_html=True)
        m[3].markdown(tile("Training accuracy", f"{hist['train_acc'][-1]:.2%}"), unsafe_allow_html=True)
        m[4].markdown(tile("Validation accuracy", f"{hist['val_acc'][-1]:.2%}", color=P["primary"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(curves_figure(hist, "loss"), use_container_width=True, config={"displayModeBar": False},
                        key=f"loss_{n}_{state}")
        c2.plotly_chart(curves_figure(hist, "acc"), use_container_width=True, config={"displayModeBar": False},
                        key=f"acc_{n}_{state}")
        gap = hist["train_acc"][-1] - hist["val_acc"][-1]
        st.caption(f"Gap between training and validation accuracy: {gap:.2%}. A growing gap while validation loss rises "
                   "is the classic sign of overfitting; early stopping keeps the best epoch.")
    else:
        st.info("No training history yet. Configure the settings above and press **Start training**.", icon="🧩")


st.fragment(run_every=2 if is_training_running() else None)(live_view)()

saved = read_json(C.history_path())
if saved:
    meta = saved["meta"]
    with st.expander("Configuration of the saved model", expanded=False):
        cfg_rows = {**meta["config"], "hidden_sizes": " × ".join(map(str, meta["config"]["hidden_sizes"])),
                    "best_epoch": meta["best_epoch"], "epochs_run": meta["epochs_run"],
                    "parameters": f"{meta['n_parameters']:,}", "train_seconds": f"{meta['train_seconds']:.1f}",
                    "split": meta["split"], "trained_at": meta["trained_at"]}
        st.dataframe(pd.DataFrame({"Setting": list(cfg_rows), "Value": [str(v) for v in cfg_rows.values()]}),
                     hide_index=True, use_container_width=True)
