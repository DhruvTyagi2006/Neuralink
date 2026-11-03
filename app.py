"""Neuralink - Home / Overview.  Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st

from src import config as C
from src.neural_network import NeuralNetwork
from src.theme import PALETTE as P
from src.ui import page_header, read_json, setup_page, tile
from src.visualization import architecture_figure

setup_page("Home")
st.markdown(
    f"<div class='hero'><h1>{C.PROJECT_NAME}</h1><p>{C.PROJECT_SUBTITLE}. Draw a digit, watch a neural network that "
    f"learned everything by gradient descent (no deep-learning framework) recognise it, and explore how it works.</p></div>",
    unsafe_allow_html=True)

ev, hist = read_json(C.evaluation_path()), read_json(C.history_path())
net = NeuralNetwork()

st.subheader("Key results of the trained model")
if ev:
    t, meta = ev["test"], ev.get("meta", {})
    c = st.columns(4)
    c[0].markdown(tile("Test accuracy", f"{t['accuracy']:.2%}", f"{t['correct']:,} of {t['n']:,} unseen images",
                       P["primary"]), unsafe_allow_html=True)
    c[1].markdown(tile("Test loss", f"{t['loss']:.4f}", "cross-entropy"), unsafe_allow_html=True)
    c[2].markdown(tile("Macro F1-score", f"{t['macro']['f1']:.4f}", "average over the 10 digits"),
                  unsafe_allow_html=True)
    c[3].markdown(tile("Trainable parameters", f"{meta.get('n_parameters', net.count_params()):,}",
                       f"best epoch {meta.get('best_epoch', '?')} of {meta.get('epochs_run', '?')}"),
                  unsafe_allow_html=True)
else:
    st.info("No evaluation results yet. Train the model, then evaluate it:", icon="🧩")
    st.code("python -m src.train --epochs 25\npython -m src.evaluate", language="bash")

st.write("")
left, right = st.columns([1.25, 1])
with left:
    with st.container(border=True):
        st.markdown("#### The network")
        st.plotly_chart(architecture_figure((784, 128, 64, 10)), use_container_width=True,
                        config={"displayModeBar": False})
with right:
    with st.container(border=True):
        st.markdown("#### What is handwritten digit recognition?")
        st.write("A computer sees an image as a grid of numbers. The task is to learn which pattern of numbers "
                 "means *3* and which means *8*, from labelled examples, and then to name digits it has never seen.")
        st.markdown("#### Dataset: MNIST")
        st.markdown("- 70,000 grey images, 28 × 28 pixels, digits 0-9\n"
                    "- **54,000** training · **6,000** validation · **10,000** official test images\n"
                    "- Pixels scaled to 0-1 and flattened into **784** input features")
        st.markdown(f"#### Model size\n`{net.count_params():,}` trainable parameters in 3 weight layers.")

st.write("")
st.subheader("Explore")
n = st.columns(5)
for col, (page, label, icon) in zip(n, [
        ("pages/1_Digit_Recognition.py", "Digit Recognition", "✏️"),
        ("pages/2_Architecture_Explorer.py", "Architecture", "🧩"),
        ("pages/3_Training_Dashboard.py", "Training", "📈"),
        ("pages/4_Model_Evaluation.py", "Evaluation", "🎯"),
        ("pages/5_Learn_ANN.py", "Learn ANN", "📚")]):
    with col:
        st.page_link(page, label=label, icon=icon, use_container_width=True)

st.caption("Predictions come from the NumPy network trained in this project. Accuracy on MNIST does not guarantee "
           "perfect recognition of every handwriting style.")
