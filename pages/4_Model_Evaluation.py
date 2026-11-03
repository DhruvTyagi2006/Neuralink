"""Model Evaluation: test metrics, confusion matrix, errors, NumPy vs Keras, experiments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.theme import PALETTE as P
from src.ui import get_test_images, model_missing_notice, page_header, read_json, setup_page, tile
from src.visualization import _base_layout, confusion_figure

setup_page("Model Evaluation", "🎯")
page_header("Model Evaluation", "Every number here is computed from the saved model's predictions on the 10,000 test images.")

if not C.numpy_model_path().exists():
    model_missing_notice()
    st.stop()
ev = read_json(C.evaluation_path())
if ev is None:
    st.info("The model has not been evaluated yet.", icon="🧩")
    if st.button("Evaluate on the test set", type="primary"):
        from src.evaluate import run_evaluation
        with st.spinner("Evaluating..."):
            run_evaluation()
        st.rerun()
    st.stop()

t = ev["test"]
tab_m, tab_c, tab_e, tab_k, tab_x = st.tabs(["Metrics", "Confusion matrix", "Mistakes", "NumPy vs Keras", "Experiments"])

with tab_m:
    c = st.columns(5)
    c[0].markdown(tile("Test accuracy", f"{t['accuracy']:.2%}", color=P["primary"]), unsafe_allow_html=True)
    c[1].markdown(tile("Test loss", f"{t['loss']:.4f}"), unsafe_allow_html=True)
    c[2].markdown(tile("Macro precision", f"{t['macro']['precision']:.4f}"), unsafe_allow_html=True)
    c[3].markdown(tile("Macro recall", f"{t['macro']['recall']:.4f}"), unsafe_allow_html=True)
    c[4].markdown(tile("Macro F1", f"{t['macro']['f1']:.4f}"), unsafe_allow_html=True)
    st.write("")
    s = st.columns(2)
    s[0].markdown(tile("Correct predictions", f"{t['correct']:,}", f"of {t['n']:,}", P["mint"]), unsafe_allow_html=True)
    s[1].markdown(tile("Incorrect predictions", f"{t['incorrect']:,}", "", P["peach"]), unsafe_allow_html=True)
    st.markdown("#### Training vs validation vs test")
    gap = pd.DataFrame({"Split": ["Training", "Validation", "Test"],
                        "Accuracy": [f"{ev[k]['accuracy']:.2%}" for k in ("train", "validation", "test")],
                        "Loss": [f"{ev[k]['loss']:.4f}" for k in ("train", "validation", "test")],
                        "Images": [f"{ev[k]['n']:,}" for k in ("train", "validation", "test")]})
    st.dataframe(gap, hide_index=True, use_container_width=True)
    st.caption("Generalization = performance on data the network never trained on. A training accuracy far above "
               "validation/test accuracy would indicate overfitting.")
    st.markdown("#### Per-digit precision, recall and F1")
    pc = pd.DataFrame(t["per_class"]).T
    pc.index.name = "Digit"
    st.dataframe(pc.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", "support": "{:.0f}"}),
                 use_container_width=True)
    st.caption("Precision: of the images predicted as this digit, how many were right. Recall: of the true images of "
               "this digit, how many were found. F1: their harmonic mean.")

with tab_c:
    st.plotly_chart(confusion_figure(np.array(t["confusion"])), use_container_width=True, config={"displayModeBar": False})
    st.markdown("**Most frequent confusions**")
    st.dataframe(pd.DataFrame(t["top_confusions"]).rename(columns={"true": "Actual", "pred": "Predicted", "count": "Times"}),
                 hide_index=True, use_container_width=True)

with tab_e:
    items = t["misclassified"]
    st.write(f"Showing the most confidently wrong predictions out of {t['incorrect']} errors. "
             "Many are digits that people also find ambiguous.")
    try:
        imgs = get_test_images()
        n_show = st.slider("How many to show", 6, min(60, len(items)), 18, step=6) if len(items) > 6 else len(items)
        cols = st.columns(6)
        for k, it in enumerate(items[:n_show]):
            with cols[k % 6]:
                st.image(imgs[it["index"]], width=95, clamp=True)
                st.caption(f"actual **{it['true']}** · predicted **{it['pred']}** ({it['prob']:.0%})")
    except Exception as exc:
        st.warning(f"Test images are unavailable: {exc}")

with tab_k:
    meta = ev.get("meta", {})
    krs = read_json(C.keras_results_path())
    rows = [{"Implementation": "NumPy ANN (from scratch)", "Optimizer": meta.get("config", {}).get("optimizer", "?") + " (+L2, lr decay)",
             "Test accuracy": f"{t['accuracy']:.2%}", "Test loss": f"{t['loss']:.4f}",
             "Epochs run": meta.get("epochs_run", "?"), "Training time (s)": f"{meta.get('train_seconds', float('nan')):.1f}",
             "Parameters": f"{meta.get('n_parameters', 0):,}"}]
    for name, r in (krs or {}).items():
        rows.append({"Implementation": "TensorFlow/Keras MLP", "Optimizer": name, "Test accuracy": f"{r['test_accuracy']:.2%}",
                     "Test loss": f"{r['test_loss']:.4f}", "Epochs run": r["epochs_run"],
                     "Training time (s)": f"{r['train_seconds']:.1f}", "Parameters": f"{r['trainable_parameters']:,}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if not krs:
        st.info("The Keras comparison has not been run. It is optional and needs TensorFlow:", icon="ℹ️")
        st.code("pip install tensorflow\npython -m src.keras_model --optimizer adam", language="bash")
    st.caption("Both are the same kind of model (a 784-128-64-10 fully connected ANN) trained on identical data splits. "
               "Differences come from implementation, optimizer and training time, not from different algorithms. "
               "Timings depend on your machine and include per-epoch evaluation.")

with tab_x:
    ex = read_json(C.experiments_path())
    if not ex:
        st.info("No experiments yet. Run `python -m src.experiments` to compare learning rates, layer sizes, "
                "optimizers, batch sizes and regularisation (about 2-3 minutes).", icon="ℹ️")
    else:
        st.write(f"Each setting was trained for {ex['epochs']} epochs and compared on the **validation** set "
                 "(the test set is never used for tuning).")
        df = pd.DataFrame(ex["rows"])
        for group, g in df.groupby("group", sort=False):
            fig = go.Figure(go.Bar(x=g["setting"], y=g["val_accuracy"], marker=dict(color=P["primary"]),
                                   text=[f"{v:.2%}" for v in g["val_accuracy"]], textposition="outside"))
            _base_layout(fig, f"Effect of {group}", 280)
            fig.update_yaxes(range=[max(0.9, g["val_accuracy"].min() - 0.02), min(1.0, g["val_accuracy"].max() + 0.012)],
                             tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(df[["group", "setting", "val_accuracy", "val_loss", "final_train_accuracy", "params", "seconds"]]
                     .style.format({"val_accuracy": "{:.4f}", "val_loss": "{:.4f}", "final_train_accuracy": "{:.4f}",
                                    "seconds": "{:.1f}"}), hide_index=True, use_container_width=True)
