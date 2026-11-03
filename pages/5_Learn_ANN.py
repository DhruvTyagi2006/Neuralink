"""Learn ANN: short, accurate explanations of the core concepts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

from src.ui import page_header, setup_page

setup_page("Learn ANN", "📚")
page_header("Learn ANN", "The ideas behind this project, in the order the network uses them.")


def topic(title, body_fn, expanded=False):
    with st.expander(title, expanded=expanded):
        body_fn()


def t_ann():
    st.write("An **artificial neural network** is a function built from layers of simple units (neurons) whose "
             "behaviour is controlled by numbers called parameters. Instead of programming rules for what a *7* looks "
             "like, we show the network many labelled examples and let an algorithm adjust the parameters until its "
             "outputs match the labels.")
    st.write("Ours is a *fully connected feedforward* network: every neuron in a layer feeds every neuron in the next, "
             "and information flows one way, input → output.")


def t_neuron():
    st.write("A neuron takes numbers in, multiplies each by a weight, adds them up with a bias, and passes the sum "
             "through an activation function.")
    st.latex(r"a=f\Big(\sum_{k} w_k x_k + b\Big)")
    st.write("Example: inputs (0.5, 1.0), weights (2, −1), bias 0.3 → sum = 0.5·2 + 1.0·(−1) + 0.3 = 0.3 → ReLU gives 0.3.")


def t_wb():
    st.write("**Weights** decide how much each input matters; **biases** let a neuron fire even when all inputs are "
             "zero (they shift the decision threshold). Our model has 109,386 of them, and they are all learned.")
    st.latex(r"Z = A_{prev}W + b")


def t_act():
    st.write("Without an activation function, stacked layers collapse into one linear map. **ReLU** adds the "
             "non-linearity that lets the network capture curves and loops in digits.")
    st.latex(r"\mathrm{ReLU}(z)=\max(0,z),\qquad \mathrm{ReLU}'(z)=\begin{cases}1&z>0\\0&z\le 0\end{cases}")
    st.write("The output layer uses **softmax** so the ten outputs are positive and sum to 1:")
    st.latex(r"P(i)=\frac{e^{z_i-\max z}}{\sum_j e^{z_j-\max z}}")


def t_fwd():
    st.write("Forward propagation pushes an image through the layers to produce probabilities:")
    st.latex(r"A_0=X,\; A_1=\mathrm{ReLU}(A_0W_1+b_1),\; A_2=\mathrm{ReLU}(A_1W_2+b_2),\; P=\mathrm{softmax}(A_2W_3+b_3)")
    st.write("Shapes for a batch of 64: (64×784) → (64×128) → (64×64) → (64×10).")


def t_loss():
    st.write("The **loss** is one number saying how wrong the network is. We use categorical cross-entropy:")
    st.latex(r"L=-\frac1m\sum_{n}\sum_{i} y_{n,i}\log p_{n,i}")
    st.write("If the correct digit gets probability 0.9, the loss is −ln 0.9 ≈ 0.105; at 0.1 it is ≈ 2.30.")


def t_bp():
    st.write("**Backpropagation** applies the chain rule layer by layer to find how the loss changes with every weight "
             "and bias. For softmax + cross-entropy the first step is remarkably simple:")
    st.latex(r"dZ_3=\frac{P-Y}{m}")
    st.latex(r"dW_3=A_2^{T}dZ_3,\; db_3=\sum dZ_3,\; dA_2=dZ_3W_3^{T},\; dZ_2=dA_2\odot \mathrm{ReLU}'(Z_2)\;\dots")
    st.write("This project checks its gradients against numerical finite differences in the unit tests.")


def t_gd():
    st.write("**Gradient descent** moves each parameter a small step downhill on the loss surface. The step size is the "
             "**learning rate** η: too small learns slowly, too large overshoots.")
    st.latex(r"W\leftarrow W-\eta\,dW,\qquad b\leftarrow b-\eta\,db")
    st.write("We use **mini-batches** (64 images per update, reshuffled each epoch), **momentum** to smooth the "
             "updates, a **decaying** learning rate, and **L2 weight decay** to keep weights small. One **epoch** = one "
             "pass over all training images.")


def t_over():
    st.write("**Overfitting**: the network memorises training images and does worse on new ones. Signs: training loss keeps "
             "falling while validation loss stops improving or rises. Defences used here: a separate validation set, "
             "early stopping, and L2 regularisation.")


def t_gen():
    st.write("**Generalization** is the ability to perform well on unseen data. That is why the data are split into "
             "training (learn), validation (tune/choose the best epoch) and test (a single final, honest check).")
    st.write("Note: a high MNIST score does not mean the model reads every handwriting style. A fully connected network "
             "ignores pixel neighbourhoods, which is why CNNs do better on images.")


topic("What is an artificial neural network?", t_ann, True)
topic("What is a neuron?", t_neuron)
topic("What are weights and biases?", t_wb)
topic("What is an activation function?", t_act)
topic("What is forward propagation?", t_fwd)
topic("What is a loss function?", t_loss)
topic("What is backpropagation?", t_bp)
topic("What is gradient descent?", t_gd)
topic("What is overfitting?", t_over)
topic("What is generalization?", t_gen)
