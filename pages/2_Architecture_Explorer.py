"""ANN Architecture Explorer: structure, parameter counts, concepts, live forward pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DatasetDownloadError
from src.theme import PALETTE as P
from src.ui import get_predictor, get_test_images, page_header, setup_page, tile
from src.neural_network import NeuralNetwork
from src.visualization import activation_bar, architecture_figure, weights_montage, _base_layout

setup_page("Architecture Explorer", "🧩")
page_header("ANN Architecture Explorer", "How 784 pixel values become ten probabilities.")

predictor = get_predictor()
net = predictor.model if predictor else NeuralNetwork()
sizes = net.layer_sizes
st.caption("Showing the trained model's architecture." if predictor else
           "No trained model yet - showing the default architecture.")

with st.container(border=True):
    st.plotly_chart(architecture_figure(sizes), use_container_width=True, config={"displayModeBar": False})

rows = net.layer_param_counts()
table = pd.DataFrame([{"Layer": f"Layer {r['layer']}", "Maps": r["shape"], "Weight matrix W": f"{sizes[r['layer']-1]} × {sizes[r['layer']]}",
                       "Weights": f"{r['weights']:,}", "Biases": f"{r['biases']:,}", "Trainable parameters": f"{r['total']:,}",
                       "Activation": r["activation"]} for r in rows])
c1, c2 = st.columns([2.6, 1])
with c1:
    st.markdown("#### Dimensions and parameters per layer")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Parameters of a layer = (inputs × outputs) weights + outputs biases.")
with c2:
    st.markdown(tile("Total trainable parameters", f"{net.count_params():,}",
                     " + ".join(f"{r['total']:,}" for r in rows), P["primary"]), unsafe_allow_html=True)

st.markdown("#### The building blocks")
t1, t2, t3, t4 = st.tabs(["Neurons, weights, biases", "Activations", "Loss", "Learning"])
with t1:
    st.markdown("**Input features** - the 28 × 28 image is flattened into **784** numbers between 0 (black) and 1 (white).")
    st.markdown("**Weights** say how strongly each input influences a neuron; **biases** shift its firing threshold. "
                "Each layer computes")
    st.latex(r"Z = A_{prev}\,W + b")
    st.markdown("Together they are the only things the network learns.")
with t2:
    st.markdown("**ReLU** (hidden layers) keeps positive values and zeros out negatives, which lets the network model "
                "non-linear patterns:")
    st.latex(r"\mathrm{ReLU}(z)=\max(0,z)")
    st.markdown("**Softmax** (output layer) turns the ten raw scores (logits) into probabilities that sum to 1:")
    st.latex(r"P(i)=\frac{e^{z_i}}{\sum_j e^{z_j}}")
    z = np.linspace(-4, 4, 200)
    f = go.Figure(go.Scatter(x=z, y=np.maximum(0, z), line=dict(color=P["primary"], width=4)))
    _base_layout(f, "ReLU", 230)
    st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
with t3:
    st.markdown("**Categorical cross-entropy** measures how far the predicted probabilities are from the true digit. "
                "With a one-hot label it equals −log of the probability given to the correct class:")
    st.latex(r"L=-\sum_i y_i\log p_i")
    st.markdown("Confident and right → loss near 0. Confident and wrong → very large loss.")
with t4:
    st.markdown("**Backpropagation** uses the chain rule to send the error backwards. For softmax + cross-entropy the "
                "output error is simply")
    st.latex(r"dZ_3=\frac{P-Y}{m},\quad dW_l=A_{l-1}^{T}dZ_l,\quad dZ_{l-1}=(dZ_l W_l^{T})\odot \mathrm{ReLU}'(Z_{l-1})")
    st.markdown("**Gradient descent** then nudges every parameter against its gradient (here with momentum):")
    st.latex(r"v \leftarrow \mu v-\eta\,dW,\qquad W \leftarrow W+v")

if predictor:
    st.markdown("#### Live forward pass")
    try:
        imgs = get_test_images()
    except (DatasetDownloadError, OSError) as exc:
        st.warning(f"MNIST is needed for this demo and could not be loaded: {exc}")
        st.stop()
    idx = st.slider("Choose a test image", 0, len(imgs) - 1, 7)
    x = (imgs[idx].reshape(1, -1) / 255.0).astype(np.float32)
    P_out, cache = net.forward(x)
    a, b = st.columns([1, 3])
    with a:
        st.image(imgs[idx], width=170, clamp=True, caption=f"Test image #{idx}")
        st.markdown(f"**Prediction: {int(P_out.argmax())}**  \nconfidence {P_out.max():.1%}")
    with b:
        for l in range(1, net.n_layers):
            st.plotly_chart(activation_bar(cache[f"A{l}"][0], f"Hidden layer {l} activations after ReLU "
                            f"({np.mean(cache[f'A{l}'][0] > 0):.0%} of neurons active)", P["primary"] if l == 1 else P["lavender"]),
                            use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(activation_bar(P_out[0], "Output probabilities (softmax)", P["peach"]),
                        use_container_width=True, config={"displayModeBar": False})
    st.markdown("#### What the first-layer neurons learned")
    st.caption("Each tile is one hidden neuron's 784 weights drawn as a 28 × 28 image. Bright = excites the neuron, "
               "dark = inhibits it. They emerge purely from gradient descent.")
    st.image(weights_montage(net.params["W1"], 32, 8), use_column_width="always", clamp=True)
